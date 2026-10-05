"""REST API + 정적 프론트엔드 서빙.  uvicorn app.api.main:app --port 8000

읽기 전용이 기본. 예측을 수정·삭제하는 엔드포인트는 존재하지 않는다. Secret 은 응답에 절대 포함하지 않는다.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import threading
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import desc, func, select

from ..backtest.engine import BacktestParams, run_backtest
from ..config import settings
from ..db import (BacktestRun, CollectionLog, DailyBar, DataConflict, Disclosure, ErrorLog, Instrument,
                  PerformanceSnapshot, Prediction, PredictionOutcome, PredictionRun, StrategyVersion, engine,
                  get_session, init_db, now_kst, verify_chain)
from ..evaluation.error_analysis import aggregate
from ..evaluation.metrics import overnight_summary, prediction_metrics
from ..features.store import PITData
from ..pipeline import STAGE_ORDER, STAGES, gate_edge, graded_frame, is_final, official_finals, stage_key
from ..prediction.engine import FLAG_LABELS
from ..prediction.model import Model
from ..research.feature_study import study
from ..strategy.manager import champion, load_samples

init_db(engine)
app = FastAPI(title="KR Overnight Predictor", version="0.2.0")
FRONTEND = Path(__file__).resolve().parents[3] / "frontend"
SITE_PASSWORD = os.getenv("SITE_PASSWORD", "")   # 설정 시 사이트 전체에 비밀번호(HTTP Basic, 아이디는 아무거나)


@app.middleware("http")
async def _password_gate(request: Request, call_next):
    if not SITE_PASSWORD or request.url.path == "/api/health":
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    if auth.startswith("Basic "):
        try:
            pw = base64.b64decode(auth[6:]).decode("utf-8").split(":", 1)[1]
        except Exception:  # noqa: BLE001
            pw = ""
        if hmac.compare_digest(pw.encode(), SITE_PASSWORD.encode()):
            return await call_next(request)
    return Response("비밀번호가 필요합니다", status_code=401, headers={"WWW-Authenticate": 'Basic realm="predictor"'},
                    media_type="text/plain; charset=utf-8")


@app.get("/api/health")
def health():
    return {"ok": True}

with get_session() as _s:   # 서버 재시작으로 중단된 백테스트 표시
    for _b in _s.execute(select(BacktestRun)).scalars():
        if _b.metrics.get("status") == "running":
            _b.metrics = {"status": "error", "error": "서버 재시작으로 중단됨"}
    _s.commit()


def _pred(p: Prediction, o: PredictionOutcome | None = None, full: bool = False) -> dict:
    d = {"id": p.id, "stage": p.stage, "ticker": p.ticker, "name": p.name, "trade_date": str(p.trade_date),
         "target_date": str(p.target_date), "price": p.price_at_prediction, "price_source": p.price_source,
         "p_up": p.p_up, "p_down": p.p_down, "p_gap_up": p.p_gap_up, "p_high_hit": p.p_high_hit,
         "expected_return": p.expected_return, "expected_net_return": p.expected_net_return,
         "range_low": p.range_low, "range_high": p.range_high, "up_med": p.up_med, "up_p80": p.up_p80,
         "down_med": p.down_med, "down_p20": p.down_p20, "confidence": p.confidence,
         "confidence_reason": (p.features_snapshot or {}).get("confidence_reason"), "final_score": p.final_score,
         "factor_scores": p.factor_scores,
         "risk_flags": [{"code": f, "label": FLAG_LABELS.get(f, f)} for f in p.risk_flags],
         "eligible": p.eligible, "is_candidate": p.is_candidate, "rank": p.rank, "data_completeness": p.data_completeness,
         "regime": p.regime, "sector": p.sector, "strategy_version": p.strategy_version,
         "created_at": p.created_at.isoformat(), "explanation": p.explanation, "row_hash": p.row_hash}
    if o is not None:
        d["outcome"] = {"entry_price": o.entry_price, "entry_basis": o.entry_basis, "open_return": o.open_return,
                        "high_return": o.high_return, "low_return": o.low_return, "close_return": o.actual_return,
                        "net_close_return": o.net_close_return, "net_open_return": o.net_open_return,
                        "gap_up": o.gap_up, "high_hit": o.high_hit, "result": o.result, "trade_result": o.trade_result,
                        "graded_at": o.graded_at.isoformat(), "source": o.source}
    if full:
        d["features_snapshot"] = p.features_snapshot
    return d


def _run_brief(r: PredictionRun) -> dict:
    return {"run_id": r.run_id, "stage": r.stage, "as_of": r.as_of.isoformat(), "n_universe": r.n_universe,
            "n_eligible": r.n_scored, "n_candidates": len(r.summary.get("candidate_tickers", [])),
            "no_candidate": r.no_candidate, "gate": r.summary.get("gate"), "final": r.summary.get("final")}


@app.get("/api/dashboard")
def dashboard(trade_date: date | None = None):
    with get_session() as s:
        T = trade_date or s.execute(select(func.max(PredictionRun.trade_date))).scalar()
        if not T:
            return {"status": "no_runs", "message": "아직 생성된 예측이 없습니다."}
        runs = s.execute(select(PredictionRun).where(PredictionRun.trade_date == T)).scalars().all()
        runs.sort(key=lambda r: stage_key(r.stage))
        cur = runs[-1]
        rows = s.execute(select(Prediction).where(Prediction.run_id == cur.run_id, Prediction.rank.is_not(None))
                         .order_by(Prediction.rank).limit(10)).scalars().all()
        snap = s.execute(select(PerformanceSnapshot).order_by(desc(PerformanceSnapshot.date)).limit(1)).scalar()
        perf = snap.metrics if snap and snap.metrics.get("n_predictions") else None
        ch = champion(s)
        return {
            "is_synthetic": cur.is_synthetic, "trade_date": str(T), "target_date": str(cur.target_date),
            "stage": cur.stage, "is_final": is_final(cur.stage),
            "stages": [_run_brief(r) for r in runs],
            "stage_plan": [{"stage": k, "time": v["time"].strftime("%H:%M"), "size": v["size"] or "전체"} for k, v in STAGES.items()],
            "run": {"as_of": cur.as_of.isoformat(), "strategy_version": cur.strategy_version,
                    "created_at": cur.created_at.isoformat()},
            "market": {"regime": cur.summary.get("regime"), "values": cur.summary.get("market"),
                       "sources": cur.data_sources.get("market")},
            "n_universe": cur.n_universe, "n_eligible": cur.n_scored, "excluded": cur.summary.get("excluded"),
            "gate": cur.summary.get("gate"), "edge": cur.summary.get("edge"), "model": cur.summary.get("model"),
            "no_candidate_message": cur.summary.get("no_candidate_message"),
            "top10": [_pred(p) for p in rows],
            "performance_as_of": str(snap.date) if snap else None,
            "performance": None if perf is None else {k: perf.get(k) for k in (
                "n_predictions", "n_days", "period", "overall", "recent_20d", "top5", "top10", "candidates",
                "by_confidence", "brier", "base_brier", "bss", "overnight", "daily", "by_stage_top5_net")},
            "champion": None if not ch else {"version": ch.version, "reason": ch.reason, "gate_edge": gate_edge(ch)},
            "costs": {"round_trip": settings.round_trip_cost, "fee_bps": settings.fee_bps, "tax_bps": settings.tax_bps,
                      "slippage_bps": settings.slippage_bps},
        }


@app.get("/api/predictions")
def predictions(trade_date: date | None = None, stage: str | None = None, limit: int = 300):
    with get_session() as s:
        q = select(PredictionRun).order_by(desc(PredictionRun.trade_date))
        if trade_date:
            q = q.where(PredictionRun.trade_date == trade_date)
        if stage:
            q = q.where(PredictionRun.stage == stage)
        runs = s.execute(q.limit(10)).scalars().all()
        if not runs:
            raise HTTPException(404, "해당 예측 없음")
        run = max(runs, key=lambda r: (r.trade_date, stage_key(r.stage)))
        items = s.execute(select(Prediction, PredictionOutcome)
                          .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
                          .where(Prediction.run_id == run.run_id)
                          .order_by(Prediction.rank.is_(None), Prediction.rank).limit(limit)).all()
        return {"run": _run_brief(run), "trade_date": str(run.trade_date), "is_synthetic": run.is_synthetic,
                "items": [_pred(p, o) for p, o in items]}


@app.get("/api/stocks/{ticker}")
def stock(ticker: str):
    with get_session() as s:
        inst = s.get(Instrument, ticker)
        if not inst:
            raise HTTPException(404, "종목 없음")
        bars = pd.read_sql(select(DailyBar.date, DailyBar.open, DailyBar.high, DailyBar.low, DailyBar.close,
                                  DailyBar.volume, DailyBar.source).where(DailyBar.ticker == ticker)
                           .order_by(DailyBar.date), s.bind)
        hist = s.execute(select(Prediction, PredictionOutcome)
                         .outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
                         .where(Prediction.ticker == ticker)
                         .order_by(desc(Prediction.trade_date), desc(Prediction.id)).limit(400)).all()
        errors = s.execute(select(ErrorLog).where(ErrorLog.ticker == ticker).order_by(desc(ErrorLog.trade_date))
                           .limit(50)).scalars().all()
        disc = s.execute(select(Disclosure).where(Disclosure.ticker == ticker).order_by(desc(Disclosure.rcept_dt))
                         .limit(30)).scalars().all()
        final, seen = [], set()    # 거래일별 마지막 확정 분석 1건
        for p, o in sorted(((p, o) for p, o in hist if is_final(p.stage)),
                           key=lambda x: (x[0].trade_date, stage_key(x[0].stage)), reverse=True):
            if p.trade_date not in seen:
                seen.add(p.trade_date)
                final.append((p, o))
        graded = [o for _, o in final if o]
        dec = [o for o in graded if o.result in ("SUCCESS", "FAILURE")]
        return {
            "instrument": {"ticker": inst.ticker, "name": inst.name, "market": inst.market, "sector": inst.sector,
                           "is_halted": inst.is_halted, "is_admin_issue": inst.is_admin_issue,
                           "status_source": inst.status_source},
            "bars": [{**r, "date": str(r["date"])} for r in bars.tail(250).to_dict("records")],
            "latest": _pred(*hist[0], full=True) if hist else None,
            "today_stages": [_pred(p, o) for p, o in hist if hist and p.trade_date == hist[0][0].trade_date][::-1],
            "history": [_pred(p, o) for p, o in final[:120]],
            "record": {"n_graded": len(graded),
                       "hit_rate": (sum(o.result == "SUCCESS" for o in dec) / len(dec)) if dec else None,
                       "avg_net_close": (sum(o.net_close_return for o in graded) / len(graded)) if graded else None},
            "errors": [{"trade_date": str(e.trade_date), "primary_cause": e.primary_cause, "tags": e.tags,
                        "details": e.details, "llm_explanation": e.llm_explanation} for e in errors],
            "disclosures": [{"rcept_no": d.rcept_no, "report_nm": d.report_nm, "rcept_dt": str(d.rcept_dt),
                             "source": d.source} for d in disc],
        }


@app.get("/api/performance")
def performance():
    with get_session() as s:
        g = graded_frame(s)
        if g.empty:
            return {"message": "채점된 예측 없음"}
        fin = official_finals(g)
        m = prediction_metrics(fin if not fin.empty else g)
        m["overnight"] = overnight_summary(fin, settings.max_final_picks, settings.round_trip_cost)
        m["by_stage"] = {}
        launches = g[g.stage.str.startswith("L")]
        for st, gg in [(st, g[g.stage == st]) for st in STAGE_ORDER] + [("LAUNCH", launches)]:
            if gg.empty:
                continue
            top = gg[gg["rank"] <= settings.max_final_picks]
            m["by_stage"][st] = {"hit": prediction_metrics(gg)["overall"], "n": int(len(gg)),
                                 "topk_avg_net_close": round(float((top.close_ret - settings.round_trip_cost).mean()), 5)
                                 if len(top) else None,
                                 "brier": round(float(((gg.p_up - gg.up) ** 2).mean()), 5)}
        for tgt, col, y in (("gap", "p_gap_up", fin.open_ret > 0), ("hit", "p_high_hit", fin.high_ret >= settings.high_hit_pct)):
            from ..prediction.calibration import brier, calibration_table
            ok = fin[col].notna() & y.notna()
            m[f"{tgt}_brier"] = round(brier(fin[col][ok], y[ok].astype(float)), 5) if ok.any() else None
            m[f"{tgt}_calibration"] = calibration_table(fin[col][ok], y[ok].astype(float)) if ok.any() else []
        m["live_only"] = prediction_metrics(fin[~fin.is_synthetic])["overall"] if (~fin.is_synthetic).any() else None
        m["snapshots"] = [{"date": str(p.date), "overall": p.metrics.get("overall"), "bss": p.metrics.get("bss")}
                          for p in s.execute(select(PerformanceSnapshot).order_by(PerformanceSnapshot.date)).scalars()]
        return m


@app.get("/api/research")
def research(live: bool = False):
    """특징별 OOS 검증 리포트 + champion 모델 계수. live=true 면 현재 누적 표본으로 즉시 재계산."""
    with get_session() as s:
        ch = champion(s)
        ev = (ch.evaluation if ch else {}) or {}
        rep = study(load_samples(s)) if live else (ev.get("last_feature_study") or ev.get("feature_study"))
        return {"champion": ch.version if ch else None, "feature_study": rep,
                "coefficients": Model(ch.weights).coef_table() if ch else [],
                "model": {k: (ch.weights or {}).get(k) for k in ("n_train", "train_period", "l2", "base_rates")} if ch else None}


@app.get("/api/errors")
def errors(limit: int = 100):
    with get_session() as s:
        rows = s.execute(select(ErrorLog).order_by(desc(ErrorLog.trade_date)).limit(limit)).scalars().all()
        all_err = pd.read_sql(select(ErrorLog.primary_cause, ErrorLog.tags), s.bind)
        if not all_err.empty:
            all_err["tags"] = all_err.tags.map(lambda x: x if isinstance(x, list) else json.loads(x))
        g = official_finals(graded_frame(s))
        return {"summary": aggregate(all_err, g) if not all_err.empty else {"n_failures": 0},
                "items": [{"trade_date": str(e.trade_date), "ticker": e.ticker, "primary_cause": e.primary_cause,
                           "tags": e.tags, "details": e.details, "llm_explanation": e.llm_explanation} for e in rows]}


@app.get("/api/strategies")
def strategies():
    with get_session() as s:
        vs = s.execute(select(StrategyVersion).order_by(StrategyVersion.created_at)).scalars().all()
        out = []
        for v in vs:
            m = Model(v.weights)
            out.append({"version": v.version, "parent": v.parent_version, "status": v.status,
                        "n_train": (v.weights or {}).get("n_train"), "train_period": (v.weights or {}).get("train_period"),
                        "edge": (v.calibration or {}).get("edge"), "oos_brier": (v.calibration or {}).get("oos_brier"),
                        "top_features": [c for c in m.coef_table()[:6]], "reason": v.reason,
                        "holdout_period": (v.evaluation or {}).get("holdout_period"),
                        "created_at": v.created_at.isoformat(),
                        "activated_at": v.activated_at.isoformat() if v.activated_at else None})
        return out


_analyze_lock = threading.Lock()


@app.post("/api/analyze")
def analyze_now():
    """'지금 다시 분석' 버튼: 밀린 데이터 수집·채점 후 지금 시점까지의 데이터로 오늘 후보 분석 (실데이터 전용)."""
    from ..pipeline import catch_up, run_ondemand
    with get_session() as s:
        last = s.execute(select(PredictionRun).order_by(desc(PredictionRun.created_at)).limit(1)).scalar()
        if last is not None and last.is_synthetic:
            raise HTTPException(400, "데모(합성) 데이터에서는 다시 분석을 지원하지 않습니다")
    if not _analyze_lock.acquire(blocking=False):
        raise HTTPException(409, "이미 분석 중입니다")
    try:
        with get_session() as s:
            catch_up(s)
            run, msg = run_ondemand(s)
            return {"ok": run is not None, "stage": run.stage if run else None, "message": msg}
    finally:
        _analyze_lock.release()


class BacktestReq(BaseModel):
    start: date | None = None
    end: date | None = None
    top_k: int = settings.max_final_picks
    exit: str = "close"
    fee_bps: float = settings.fee_bps
    tax_bps: float = settings.tax_bps
    slippage_bps: float = settings.slippage_bps


_bt_lock = threading.Lock()


@app.post("/api/backtests")
def start_backtest(req: BacktestReq):
    if req.exit not in ("open", "close"):
        raise HTTPException(400, "exit 은 open 또는 close")
    if not _bt_lock.acquire(blocking=False):
        raise HTTPException(409, "다른 백테스트가 실행 중입니다")
    with get_session() as s:
        bt = BacktestRun(created_at=now_kst(), params=req.model_dump(mode="json"), strategy_version="walk-forward",
                         metrics={"status": "running"})
        s.add(bt)
        s.commit()
        bt_id = bt.id

    def work():
        try:
            with get_session() as s2:
                data = PITData.load(s2)
                synth = bool(data.bars_long.source.eq("SYNTHETIC(TEST ONLY)").any())
                metrics, _ = run_backtest(data, BacktestParams(**req.model_dump()))
                row = s2.get(BacktestRun, bt_id)
                row.metrics, row.is_synthetic = metrics | {"status": "done"}, synth
                s2.commit()
        except Exception as e:  # noqa: BLE001
            with get_session() as s3:
                row = s3.get(BacktestRun, bt_id)
                row.metrics = {"status": "error", "error": repr(e)}
                s3.commit()
        finally:
            _bt_lock.release()

    threading.Thread(target=work, daemon=True).start()
    return {"id": bt_id, "status": "running"}


@app.get("/api/backtests")
def list_backtests():
    with get_session() as s:
        rows = s.execute(select(BacktestRun).order_by(desc(BacktestRun.created_at)).limit(20)).scalars().all()
        return [{"id": b.id, "created_at": b.created_at.isoformat(), "params": b.params, "strategy_version": b.strategy_version,
                 "status": b.metrics.get("status", "done"), "is_synthetic": b.is_synthetic} for b in rows]


@app.get("/api/backtests/{bt_id}")
def get_backtest(bt_id: int):
    with get_session() as s:
        b = s.get(BacktestRun, bt_id)
        if not b:
            raise HTTPException(404)
        return {"id": b.id, "params": b.params, "strategy_version": b.strategy_version, "metrics": b.metrics,
                "is_synthetic": b.is_synthetic}


@app.get("/api/integrity")
def integrity():
    with get_session() as s:
        return verify_chain(s)


@app.get("/api/data-status")
def data_status():
    with get_session() as s:
        sub = select(CollectionLog.collector, CollectionLog.target, CollectionLog.status, CollectionLog.message,
                     CollectionLog.finished_at).order_by(desc(CollectionLog.finished_at)).limit(50)
        logs = [dict(r._mapping) for r in s.execute(sub)]
        conflicts = s.execute(select(DataConflict).order_by(desc(DataConflict.detected_at)).limit(50)).scalars().all()
        sources = s.execute(select(DailyBar.source, func.count(), func.max(DailyBar.date)).group_by(DailyBar.source)).all()
        keys = {"KRX": bool(settings.krx_id), "KIS": bool(settings.kis_app_key), "DART": bool(settings.dart_api_key),
                "ECOS": bool(settings.ecos_api_key), "ANTHROPIC": bool(settings.anthropic_api_key)}
        return {"keys_configured": keys, "data_source": settings.data_source,
                "logs": [{**l, "finished_at": str(l["finished_at"])} for l in logs],
                "bar_sources": [{"source": a, "rows": b, "latest_date": str(c)} for a, b, c in sources],
                "conflicts": [{"entity": c.entity, "field": c.field, "date": str(c.data_date), "a": [c.source_a, c.value_a],
                               "b": [c.source_b, c.value_b], "rel_diff": c.rel_diff} for c in conflicts]}


if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(FRONTEND / "index.html")

    @app.get("/{page}.html")
    def page(page: str):
        f = FRONTEND / f"{page}.html"
        if not f.exists():
            raise HTTPException(404)
        return FileResponse(f)
