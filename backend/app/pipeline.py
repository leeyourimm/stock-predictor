"""일일 Overnight 파이프라인.

거래일 T:
  S1400 전체 종목 1차 분석 → S1430 재평가 → S1500 압축 → S1510 최종 분석 → S1520 Overnight 후보 확정
  (각 단계는 해당 시각까지 공개된 데이터만 사용, 각 단계 결과는 불변 저장)
  18:30 확정치 수집 → 전 거래일 예측 채점(시가/고가/저가/종가 수익률, 비용 차감) → 오류분석 → 학습표본 적재
        → champion 의 OOS 성과(그림자 포트폴리오) 누적 → 성과 스냅샷
"""
from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .calendar_kr import is_trading_day, next_trading_day
from .config import settings
from .db import (DailyBar, Disclosure, ErrorLog, IndexBar, Instrument, PerformanceSnapshot, Prediction,
                 PredictionOutcome, PredictionRun, last_prediction_hash, now_kst, prediction_hash)
from .evaluation.error_analysis import analyze_failure
from .evaluation.metrics import overnight_summary, prediction_metrics
from .evaluation.outcomes import overnight
from .explain.claude import explain_failure, explain_prediction, template_explanation
from .features.builder import build_features
from .features.regime import classify
from .features.store import PITData, _pick_primary
from .prediction.engine import NO_CANDIDATE_MSG, key_reasons, predict
from .prediction.model import FEATURE_NAMES, GROUPS, Model
from .strategy.manager import champion, edge_stats, ensure_initial, save_samples

log = logging.getLogger("pipeline")
SYNTH = "SYNTHETIC(TEST ONLY)"
STAGES = {s[0]: {"time": s[1], "snapshot": s[2], "size": s[3]} for s in settings.stages}
STAGE_ORDER = [s[0] for s in settings.stages]
ONDEMAND_PREFIX = "L"                 # 앱 실행 시 분석: 단계명 L{HHMM}
ONDEMAND_CUTOFF = time(15, 30)        # 종가 단일가 주문 마감 — 이후에는 오늘 Overnight 매수 불가


def is_final(stage: str) -> bool:
    """후보를 확정하는 단계: 고정 일정의 최종 단계 또는 앱 실행 시 분석."""
    return stage == settings.final_stage or stage.startswith(ONDEMAND_PREFIX)


def stage_key(stage: str) -> int:
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 100 + int(stage[1:] or 0)


def official_finals(df: pd.DataFrame) -> pd.DataFrame:
    """거래일마다 공식 확정 예측 1회만 남긴다 (같은 날 여러 번 실행했다면 가장 늦은 실행). 성과 이중 집계 방지."""
    if df.empty:
        return df
    f = df[df.stage.map(is_final)]
    if f.empty:
        return f
    last = f.groupby("trade_date").stage.agg(lambda x: max(x, key=stage_key))
    return f[f.stage.values == f.trade_date.map(last).values]


# ------------------------------------------------------------------ 수집
def collect(session: Session, start: date, end: date, snapshot: bool = False) -> None:
    """실데이터 수집 (키/네트워크 없으면 각 항목이 CollectionLog 에 UNAVAILABLE 로 남는다)."""
    from .collectors import dart, ecos, global_markets, kis, krx, secondary
    from .collectors.base import DataUnavailable
    try:
        if not session.query(Instrument).count():
            krx.collect_instruments(session, end)
        days = krx.trading_days(start, end)
    except DataUnavailable as e:
        log.warning("KRX 거래일 조회 불가: %s", e)
        days = [d for d in pd.date_range(start, end).date if is_trading_day(d)]
    for d in days:
        krx.collect_daily(session, d)
    global_markets.collect_global(session, start - timedelta(days=10), end)
    ecos.collect_ecos(session, start - timedelta(days=10), end)
    dart.collect_disclosures(session, start, end)
    try:
        secondary.collect_admin_status(session)
    except Exception as e:  # noqa: BLE001
        log.warning("관리종목 조회 실패: %s", e)
    if snapshot:
        kis.collect_snapshot(session, [i.ticker for i in session.query(Instrument).all()])
    detect_conflicts(session, start, end)


def detect_conflicts(session: Session, start: date, end: date) -> int:
    from .db import DataConflict
    bars = pd.read_sql(select(DailyBar.ticker, DailyBar.date, DailyBar.close, DailyBar.source)
                       .where(DailyBar.date.between(start, end)), session.bind)
    n = 0
    if bars.empty:
        return 0
    for (t, d), g in bars.groupby(["ticker", "date"]):
        if g.source.nunique() < 2:
            continue
        g = g.reset_index(drop=True)
        a, b = g.iloc[0], g.iloc[1]
        rel = abs(a.close - b.close) / max(abs(a.close), 1e-9)
        if rel > settings.conflict_rel_tol:
            exists = session.query(DataConflict).filter_by(entity=t, data_date=d, field="close").first()
            if not exists:
                session.add(DataConflict(entity=t, field="close", data_date=d, source_a=a.source, value_a=a.close,
                                         source_b=b.source, value_b=b.close, rel_diff=rel, detected_at=now_kst()))
                n += 1
    session.commit()
    return n




def stage_universe(session: Session, T: date, stage: str) -> list[str] | None:
    """단계별 분석 대상. size=0 이면 전체, 아니면 직전 단계 순위 상위 N (후보 풀 압축)."""
    size = STAGES[stage]["size"]
    if not size:
        return None
    i = STAGE_ORDER.index(stage)
    for prev in reversed(STAGE_ORDER[:i]):
        run = session.execute(select(PredictionRun).where(PredictionRun.trade_date == T, PredictionRun.stage == prev)).scalar()
        if run:
            rows = session.execute(select(Prediction.ticker).where(Prediction.run_id == run.run_id, Prediction.rank.is_not(None))
                                   .order_by(Prediction.rank).limit(size)).all()
            return [r[0] for r in rows]
    return None


def collect_stage_snapshot(session: Session, T: date, stage: str) -> None:
    """라이브: 단계 시작 직전 KIS 장중 스냅샷 수집 (대상 = 해당 단계 유니버스)."""
    from .collectors import kis
    uni = stage_universe(session, T, stage) or [i.ticker for i in session.query(Instrument).all()]
    kis.collect_index_snapshot(session)
    kis.collect_snapshot(session, uni)


# ------------------------------------------------------------------ 채점 데이터
def graded_frame(session: Session, stage: str | None = None) -> pd.DataFrame:
    q = (select(Prediction.id, Prediction.run_id, Prediction.stage, Prediction.ticker, Prediction.trade_date,
                Prediction.target_date, Prediction.p_up, Prediction.p_gap_up, Prediction.p_high_hit,
                Prediction.expected_return, Prediction.expected_net_return, Prediction.confidence, Prediction.rank,
                Prediction.regime, Prediction.sector, Prediction.final_score, Prediction.eligible, Prediction.is_candidate,
                Prediction.range_low, Prediction.range_high, Prediction.strategy_version, PredictionRun.is_synthetic,
                PredictionOutcome.open_return, PredictionOutcome.high_return, PredictionOutcome.low_return,
                PredictionOutcome.actual_return, PredictionOutcome.net_close_return, PredictionOutcome.result)
         .join(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
         .join(PredictionRun, PredictionRun.run_id == Prediction.run_id))
    if stage:
        q = q.where(Prediction.stage == stage)
    df = pd.read_sql(q, session.bind)
    if df.empty:
        return df
    df = df.rename(columns={"expected_return": "exp_ret", "open_return": "open_ret", "high_return": "high_ret",
                            "low_return": "low_ret"})
    df["close_ret"] = df.actual_return
    df["up"] = (df.actual_return > 0).astype(int)
    df["trend"] = df.regime.str.split("|").str[0]
    df["is_candidate"] = df.is_candidate.astype(bool)
    return df


def gate_edge(ch) -> dict:
    """후보 게이트용 champion OOS 성과: 최근 250 거래일 그림자 포트폴리오 일별 순수익."""
    ed = (ch.calibration or {}).get("edge_daily", {}) if ch else {}
    recent = dict(sorted(ed.items())[-250:])
    return edge_stats(recent)


# ------------------------------------------------------------------ 단계별 분석
def run_stage(session: Session, stage: str, T: date | None = None, data: PITData | None = None,
              use_claude: bool = True, explain_top: int = 10, as_of: datetime | None = None) -> PredictionRun:
    T = T or now_kst().date()
    as_of = as_of or datetime.combine(T, STAGES[stage]["time"])
    existing = session.execute(select(PredictionRun).where(PredictionRun.trade_date == T,
                                                           PredictionRun.stage == stage)).scalar()
    if existing:
        log.info("%s %s 예측은 이미 존재 → 재생성하지 않음(불변)", T, stage)
        return existing
    data = data or PITData.load(session, since=T - timedelta(days=420))
    view = data.as_of(as_of)
    is_synth = bool(view.bar_source.isin([SYNTH]).any().any()) if not view.bar_source.empty else False
    if not is_synth:
        if abs(now_kst() - as_of) > timedelta(minutes=15):
            raise ValueError(f"실데이터 예측은 해당 시각({as_of})에만 생성 가능 (사후 생성 금지, 과거는 백테스트 사용)")
        if not is_trading_day(T):
            raise ValueError(f"{T} 는 휴장일")
    target = next_trading_day(T, data.trading_dates() if is_synth else None)
    final = is_final(stage)

    fs = build_features(view)
    if fs.table.empty:
        raise ValueError("분석 가능한 데이터 없음")
    regime = classify(fs.market)
    ch = champion(session) or ensure_initial(session)
    model, edge = Model(ch.weights), gate_edge(ch)
    universe = stage_universe(session, T, stage) if stage in STAGES else None
    res = predict(fs, regime, model, edge, universe=universe, final=final)
    X = res.attrs["features"]
    gate = res.attrs["gate"]
    cands = res[res.is_candidate].sort_values("rank")
    parent = None
    i = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 0
    if i:
        prev = session.execute(select(PredictionRun.run_id).where(PredictionRun.trade_date == T,
                                                                  PredictionRun.stage == STAGE_ORDER[i - 1])).scalar()
        parent = prev
    created = now_kst()
    run_id = f"{T:%Y%m%d}-{stage}-{uuid.uuid4().hex[:6]}"
    top = res[res["rank"].notna()].sort_values("rank").head(10)
    summary = {
        "stage": stage, "final": final, "regime": regime, "market": fs.market, "last_bar_date": str(fs.last_bar_date),
        "n_universe": int(len(res)), "n_eligible": int(res.eligible.sum()), "universe_from_prev": universe is not None,
        "excluded": {k: int(v) for k, v in pd.Series([f for fl in res[~res.eligible].risk_flags for f in fl]).value_counts().items()},
        "candidate_tickers": list(cands.index),
        "no_candidate_message": (NO_CANDIDATE_MSG if cands.empty else None),
        "gate": gate, "edge": edge,
        "confidence_counts": {k: int(v) for k, v in res.confidence.value_counts().items()},
        "model": {"version": ch.version, "trained": model.trained, "n_train": (ch.weights or {}).get("n_train", 0)},
        "top10": [{"ticker": t, "name": r["name"], "p_up": r.p_up, "exp_net": r.exp_net, "confidence": r.confidence,
                   "is_candidate": bool(r.is_candidate), "reasons": key_reasons(r.factors, True),
                   "risks": key_reasons(r.factors, False)} for t, r in top.iterrows()],
    }
    sources = {"market": fs.market_meta, "as_of": as_of.isoformat(),
               "price_sources": sorted(set(view.bar_source.stack().unique())) if not view.bar_source.empty else [],
               "intraday_snapshot": "available" if not view.snapshot.empty else "데이터 없음"}
    run = PredictionRun(run_id=run_id, stage=stage, parent_run_id=parent, as_of=as_of, trade_date=T, target_date=target,
                        strategy_version=ch.version, regime=regime["label"], n_universe=len(res),
                        n_scored=int(res.eligible.sum()), no_candidate=bool(cands.empty), summary=summary,
                        data_sources=sources, created_at=created, is_synthetic=is_synth)
    session.add(run)
    session.flush()

    explain_set = set(cands.head(explain_top).index) if final else set()
    prev_hash = last_prediction_hash(session)
    for t, r in res.sort_values(["rank"], na_position="last").iterrows():
        snap = fs.snapshots[t]
        snap["values"].update({f: (None if pd.isna(v) else round(float(v), 6)) for f, v in X.loc[t].items()})
        rec = r.to_dict() | {"ticker": t}
        if t in explain_set and use_claude:
            expl = explain_prediction(rec, snap)
        elif r.eligible and not pd.isna(r["rank"]) and r["rank"] <= 20:
            expl = "[자동 템플릿] " + template_explanation(rec)
        else:
            expl = None
        price_src = (f"SNAPSHOT@{str(r.price_ts)[11:16]} {snap['sources']['intraday']['source']}" if r.price_is_intraday
                     else f"CLOSE {snap['sources']['price']['source']} ({snap['sources']['price']['last_bar_date']})")
        fields = dict(
            run_id=run_id, stage=stage, ticker=t, name=r["name"], trade_date=T, target_date=target,
            price_at_prediction=r.price, price_source=price_src,
            price_timestamp=pd.Timestamp(r.price_ts).to_pydatetime() if r.price_is_intraday else None,
            p_up=float(r.p_up), p_down=float(r.p_down), range_low=r.range_low, range_high=r.range_high,
            expected_return=r.exp_ret, expected_net_return=r.exp_net, p_gap_up=r.p_gap_up, p_high_hit=r.p_high_hit,
            up_med=r.up_med, up_p80=r.up_p80, down_med=r.down_med, down_p20=r.down_p20,
            is_candidate=bool(r.is_candidate), confidence=r.confidence, factor_scores=r.factors,
            final_score=float(r.final_score), risk_flags=list(r.risk_flags), eligible=bool(r.eligible),
            rank=None if pd.isna(r["rank"]) else int(r["rank"]), data_completeness=float(r.completeness),
            features_snapshot=snap | {"confidence_reason": r.confidence_reason, "regime": regime},
            regime=regime["label"], sector=r.sector, strategy_version=ch.version, created_at=created, explanation=expl)
        h = prediction_hash(prev_hash, fields)
        session.add(Prediction(**fields, prev_hash=prev_hash, row_hash=h))
        prev_hash = h
    session.commit()
    log.info("%s %s 저장: %s (%d종목, 후보 %d, 게이트 %s)", T, stage, run_id, len(res), len(cands), gate["ok"])
    return run


def run_ondemand(session: Session, now: datetime | None = None, data: PITData | None = None,
                 use_claude: bool = True) -> tuple[PredictionRun | None, str]:
    """앱을 실행한 시점(as_of = 지금)까지 공개된 데이터로 오늘 Overnight 후보를 확정 분석한다.
    거래일 15:30(종가 단일가 주문 마감) 전에만 생성. 같은 날 여러 번 실행하면 모두 불변 기록되고,
    성과 집계에는 그날 마지막 실행만 쓰인다(official_finals)."""
    now = (now or now_kst()).replace(second=0, microsecond=0)
    T = now.date()
    synth = data is not None and bool(data.bars_long.source.eq(SYNTH).any())
    if not synth and not is_trading_day(T):
        return None, f"오늘({T})은 휴장일입니다. 다음 거래일 15:30 전에 실행하면 그날 매수 후보를 분석합니다."
    if now.time() >= ONDEMAND_CUTOFF:
        return None, ("오늘 장 마감(15:30) 이후라 오늘 매수할 후보는 분석하지 않습니다. "
                      "채점·데이터 갱신은 끝났고, 다음 거래일 15:30 전에 실행하면 새 후보를 분석합니다.")
    stage, as_of = f"{ONDEMAND_PREFIX}{now:%H%M}", now
    if not synth:
        if settings.kis_app_key and now.time() >= time(9, 0):
            try:
                from .collectors import kis
                kis.collect_index_snapshot(session)
                kis.collect_snapshot(session, [i.ticker for i in session.query(Instrument).all()])
            except Exception as e:  # noqa: BLE001
                log.warning("장중 시세 수집 실패 (장중 지표 없이 진행): %s", e)
        as_of = now_kst()        # 방금 수집한 시세까지 포함 (그 이후 데이터는 없음)
    run = run_stage(session, stage, T, data, use_claude=use_claude, as_of=as_of)
    return run, run.summary.get("no_candidate_message") or f"후보 {len(run.summary.get('candidate_tickers', []))}종목"


def catch_up(session: Session, now: datetime | None = None) -> dict:
    """앱 실행 시: 마지막 수집일 이후 확정된 데이터를 모으고, 결과가 나온 예측을 채점한다."""
    now = now or now_kst()
    last = session.execute(select(func.max(DailyBar.date))).scalar()
    end = now.date() if now.time() >= time(18, 30) else now.date() - timedelta(days=1)
    start = (last + timedelta(days=1)) if last else now.date() - timedelta(days=400)
    if start <= end:
        collect(session, start, end)
    return run_grading(session, now)


def run_day(session: Session, T: date, data: PITData | None = None, use_claude: bool = False) -> list[PredictionRun]:
    """(재생/테스트용) 하루 전체 단계를 순서대로 실행."""
    return [run_stage(session, s, T, data, use_claude=use_claude) for s in STAGE_ORDER]


# ------------------------------------------------------------------ 채점
def run_grading(session: Session, now: datetime | None = None, use_claude: bool = False, snapshot: bool = True) -> dict:
    now = now or now_kst()
    pending = session.execute(
        select(Prediction).outerjoin(PredictionOutcome, PredictionOutcome.prediction_id == Prediction.id)
        .where(PredictionOutcome.prediction_id.is_(None))).scalars().all()
    if not pending:
        return {"graded": 0}
    min_d = min(p.trade_date for p in pending)
    bars = pd.read_sql(select(DailyBar.ticker, DailyBar.date, DailyBar.open, DailyBar.high, DailyBar.low, DailyBar.close,
                              DailyBar.source, DailyBar.available_at).where(DailyBar.date >= min_d), session.bind)
    bars["available_at"] = pd.to_datetime(bars.available_at)
    bars = _pick_primary(bars[bars.available_at <= now], ["ticker", "date"]).set_index(["ticker", "date"])
    idx = pd.read_sql(select(IndexBar.symbol, IndexBar.date, IndexBar.close, IndexBar.available_at)
                      .where(IndexBar.date >= min_d - timedelta(days=10)), session.bind)
    idx["available_at"] = pd.to_datetime(idx.available_at)
    idx = idx[idx.available_at <= now].drop_duplicates(["symbol", "date"]).set_index(["symbol", "date"]).close.sort_index()
    kospi_days = sorted(d for (s, d) in idx.index if s == "KOSPI")
    disc = pd.read_sql(select(Disclosure.ticker, Disclosure.rcept_dt), session.bind)
    disc_set = set(zip(disc.ticker, disc.rcept_dt)) if not disc.empty else set()
    inst = {i.ticker: i for i in session.query(Instrument).all()}
    runs = {r.run_id: r for r in session.execute(select(PredictionRun).where(PredictionRun.trade_date >= min_d)).scalars()}

    def idx_ret(sym, d0, d1):
        try:
            return float(idx.loc[(sym, d1)] / idx.loc[(sym, d0)] - 1)
        except KeyError:
            return None

    graded, failures, skipped = 0, 0, 0
    samples = []
    for p in pending:
        later = [d for d in kospi_days if d > p.trade_date]
        T1 = later[0] if later else p.target_date
        try:
            c0 = float(bars.loc[(p.ticker, p.trade_date), "close"])
            o1, h1, l1, c1, src = bars.loc[(p.ticker, T1), ["open", "high", "low", "close", "source"]]
        except KeyError:
            skipped += 1
            continue
        intraday = p.price_source.startswith("SNAPSHOT") and p.price_at_prediction
        entry = float(p.price_at_prediction) if intraday else c0
        basis = p.price_source.split(" ")[0] if intraday else "CLOSE"
        oc = overnight(entry, o1, h1, l1, float(c1))
        r = oc["close_ret"]
        up = 1 if r > 0 else 0
        result = "NEUTRAL" if abs(r) < settings.neutral_band else ("SUCCESS" if (p.p_up >= 0.5) == (r > 0) else "FAILURE")
        session.add(PredictionOutcome(
            prediction_id=p.id, entry_price=entry, entry_basis=basis, ref_close=c0, next_open=o1, next_high=h1,
            next_low=l1, target_close=float(c1), open_return=oc["open_ret"], high_return=oc["high_ret"],
            low_return=oc["low_ret"], actual_return=r, net_open_return=oc["net_open"], net_close_return=oc["net_close"],
            gap_up=oc["gap_up"], high_hit=oc["high_hit"],
            in_range=None if p.range_low is None else bool(p.range_low <= r <= p.range_high), result=result,
            trade_result="WIN" if oc["net_close"] > 0 else "LOSS", brier=(p.p_up - up) ** 2,
            source=src + ("" if T1 == p.target_date else f" (실제 다음 거래일 {T1})"), graded_at=now_kst()))
        graded += 1
        vals = (p.features_snapshot or {}).get("values", {})
        samples.append({"trade_date": p.trade_date, "target_date": T1, "ticker": p.ticker, "stage": p.stage,
                        "regime": p.regime, "sector": p.sector, "eligible": p.eligible, "entry_basis": basis,
                        **{f: vals.get(f) for f in FEATURE_NAMES},
                        "open_ret": oc["open_ret"], "high_ret": oc["high_ret"], "low_ret": oc["low_ret"],
                        "close_ret": r, "_synth": runs[p.run_id].is_synthetic if p.run_id in runs else False})
        analyze = is_final(p.stage) and (p.is_candidate or (p.rank and p.rank <= 20)) and \
            (result == "FAILURE" or (p.is_candidate and oc["net_close"] <= 0))
        if analyze:
            i = inst.get(p.ticker)
            ctx = {"index_ret_next": idx_ret("KOSDAQ" if i and i.market == "KOSDAQ" else "KOSPI", p.trade_date, T1),
                   "sector_ret_next": idx_ret(f"SECTOR:{i.sector_index}", p.trade_date, T1) if i and i.sector_index else None,
                   "disclosure_next": (p.ticker, T1) in disc_set or (p.ticker, p.trade_date) in disc_set}
            pred = {"p_up": p.p_up, "factor_scores": p.factor_scores, "features_snapshot": p.features_snapshot,
                    "range_low": p.range_low, "range_high": p.range_high, "ticker": p.ticker}
            a = analyze_failure(pred, {**oc, "net_close": oc["net_close"]}, ctx)
            expl = explain_failure(pred, a) if use_claude and p.is_candidate else None
            session.add(ErrorLog(prediction_id=p.id, trade_date=p.trade_date, ticker=p.ticker,
                                 primary_cause=a["primary_cause"], tags=a["tags"], details=a["details"],
                                 llm_explanation=expl, created_at=now_kst()))
            failures += 1
    session.commit()
    if samples:
        sdf = pd.DataFrame(samples)
        for (stage, synth), g in sdf.groupby(["stage", "_synth"]):
            save_samples(session, g, "LIVE", stage, bool(synth))
        update_live_edge(session, sdf)
    if snapshot:
        snapshot_performance(session, now.date())
    return {"graded": graded, "error_logs": failures, "pending_or_unavailable": skipped}


def update_live_edge(session: Session, sdf: pd.DataFrame) -> None:
    """champion 이 실제로 낸 최종 단계 예측의 그림자 포트폴리오(상위 K, 비용 차감) 일별 수익을 OOS 증거로 누적."""
    ch = champion(session)
    if not ch:
        return
    fin = pd.DataFrame(session.execute(
        select(Prediction.trade_date, Prediction.stage, Prediction.ticker, Prediction.strategy_version)
        .where(Prediction.rank.is_not(None), Prediction.rank <= settings.max_final_picks,
               Prediction.trade_date.in_(sorted(set(sdf.trade_date))))).all(),
        columns=["trade_date", "stage", "ticker", "strategy_version"])
    fin = official_finals(fin)
    picks = {(d, st, t) for d, st, t, v in fin.itertuples(index=False) if v == ch.version}
    if not picks or not (ch.weights or {}).get("n_train"):
        return
    s = sdf[[(d, st, t) in picks for d, st, t in zip(sdf.trade_date, sdf.stage, sdf.ticker)]]
    daily = s.groupby("trade_date").close_ret.mean() - settings.round_trip_cost
    ed = dict(ch.calibration.get("edge_daily", {}))
    ed.update({str(k): float(v) for k, v in daily.items()})
    ch.calibration = {**ch.calibration, "edge_daily": ed, "edge": edge_stats(ed)}
    session.commit()


def snapshot_performance(session: Session, d: date) -> dict:
    g = graded_frame(session)
    if g.empty:
        return {}
    fin = official_finals(g)
    m = prediction_metrics(fin if not fin.empty else g)
    m["overnight"] = overnight_summary(fin, settings.max_final_picks, settings.round_trip_cost)
    m["by_stage"] = {s: prediction_metrics(gg).get("overall") for s, gg in g.groupby("stage")}
    m["by_stage_top5_net"] = {s: round(float((gg[gg["rank"] <= 5].close_ret - settings.round_trip_cost).mean()), 5)
                              for s, gg in g.groupby("stage") if (gg["rank"] <= 5).any()}
    session.merge(PerformanceSnapshot(date=d, metrics=m, created_at=now_kst()))
    session.commit()
    return m
