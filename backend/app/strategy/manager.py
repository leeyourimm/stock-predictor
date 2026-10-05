"""전략(모델) 버전 관리 + champion/challenger 검증. 목표 지표 = 비용 차감 Overnight 위험조정수익.

원칙
- 전략 = 학습된 모델 스펙(strategy_versions.weights) + OOS 성과 기록(calibration). 변경은 새 버전으로만.
- 평가는 '이전 평가 이후 새로 쌓인' 홀드아웃 거래일(전략 개선에 한 번도 쓰이지 않은 OOS)에서만.
- 홀드아웃 표본이 부족하면 보류. 채택 조건:
    (1) 그림자 포트폴리오(일별 상위 K, 비용 차감) 일별 수익 차이의 블록 부트스트랩 95% 하한 > 0
    (2) 상승확률 Brier 가 champion 보다 나빠지지 않을 것 (+0.001 허용)
- 채택/거절 모두 근거와 함께 기록. 거절돼도 그 홀드아웃은 '사용됨'으로 표시해 재사용하지 않는다.
- champion 의 OOS 일별 성과(edge_daily)는 계속 누적되어 '매수 후보' 게이트에 쓰인다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..backtest.engine import shadow_edge
from ..config import settings
from ..db import StrategyVersion, TrainingSample, now_kst
from ..evaluation.outcomes import overnight  # noqa: F401  (문서용: 표본 결과 정의)
from ..prediction.calibration import brier
from ..prediction.model import FEATURE_NAMES, Model
from ..research.feature_study import study

OUTCOME_COLS = ["open_ret", "high_ret", "low_ret", "close_ret"]


def champion(session: Session) -> StrategyVersion | None:
    return session.execute(select(StrategyVersion).where(StrategyVersion.status == "champion")).scalar()


def _next_version(session: Session) -> str:
    return f"v{len(session.execute(select(StrategyVersion.version)).all()) + 1}"


# ------------------------------------------------------------------ 표본
def save_samples(session: Session, rows: pd.DataFrame, source: str, stage: str, is_synthetic: bool) -> int:
    from ..collectors.base import upsert_ignore
    recs = []
    now = now_kst()
    for r in rows.itertuples(index=False):
        feats = {f: (None if pd.isna(getattr(r, f, np.nan)) else float(getattr(r, f))) for f in FEATURE_NAMES}
        feats["_eligible"] = bool(getattr(r, "eligible", True))
        recs.append(dict(source=source, stage=stage, trade_date=r.trade_date, target_date=r.target_date, ticker=r.ticker,
                         regime=getattr(r, "regime", None), sector=getattr(r, "sector", None), features=feats,
                         entry_basis=getattr(r, "entry_basis", "CLOSE"),
                         **{k: (None if pd.isna(getattr(r, k)) else float(getattr(r, k))) for k in OUTCOME_COLS},
                         is_synthetic=is_synthetic, created_at=now))
    n = upsert_ignore(session, TrainingSample, recs, ["trade_date", "stage", "ticker", "source"])
    session.commit()
    return n


def load_samples(session: Session) -> pd.DataFrame:
    rows = session.execute(select(TrainingSample)).scalars().all()
    if not rows:
        return pd.DataFrame()
    base = pd.DataFrame([{"trade_date": r.trade_date, "target_date": r.target_date, "ticker": r.ticker,
                          "stage": r.stage, "source": r.source, "regime": r.regime, "sector": r.sector,
                          "entry_basis": r.entry_basis, **{k: getattr(r, k) for k in OUTCOME_COLS}} for r in rows])
    feats = pd.DataFrame([r.features for r in rows])
    df = pd.concat([base, feats], axis=1)
    df["eligible"] = df.pop("_eligible").fillna(True).astype(bool) if "_eligible" in df else True
    df["trend"] = df.regime.fillna("UNKNOWN").str.split("|").str[0]
    return df


def score_frame(model: Model, df: pd.DataFrame) -> pd.DataFrame:
    """표본에 모델을 적용해 p_up/exp_net 과 (일자·단계별) 기대수익 순위를 붙인다."""
    P = model.predict(df.reindex(columns=FEATURE_NAMES).astype(float))
    out = df.assign(p_up=P.p_up.values, exp_net=P.exp_net.values)
    elig = out[out.eligible]
    out["rank"] = elig.groupby(["trade_date", "stage"]).exp_net.rank(ascending=False, method="first")
    return out


def daily_shadow(scored: pd.DataFrame, top_k: int) -> pd.Series:
    d = scored[scored["rank"] <= top_k]
    return d.groupby("trade_date").close_ret.mean() - settings.round_trip_cost


def edge_stats(daily: dict[str, float]) -> dict:
    if not daily:
        return {"days": 0}
    s = pd.Series(daily).sort_index()
    n, sd = len(s), s.std(ddof=1) if len(s) > 1 else np.nan
    ok = n > 1 and sd > 0
    return {"days": int(n), "mean_net": float(s.mean()), "tstat": float(s.mean() / sd * np.sqrt(n)) if ok else None,
            "sharpe": float(s.mean() / sd * np.sqrt(252)) if ok else None, "period": [s.index[0], s.index[-1]]}


# ------------------------------------------------------------------ 초기 전략
def ensure_initial(session: Session, bt_preds: pd.DataFrame | None = None, source: str = "historical",
                   is_synthetic: bool = False) -> StrategyVersion:
    ch = champion(session)
    if ch:
        return ch
    if bt_preds is None or bt_preds.empty or not bt_preds.oos.any():
        v = StrategyVersion(version="v1", parent_version=None, weights={"n_train": 0},
                            calibration={"edge_daily": {}, "edge": {"days": 0}}, params={}, status="champion",
                            reason="초기 전략: 학습 표본 없음 → 모든 종목 확률 50%, 매수 후보 없음", evaluation={},
                            created_at=now_kst(), activated_at=now_kst())
        session.add(v)
        session.commit()
        return v
    save_samples(session, bt_preds, "BACKTEST", settings.final_stage, is_synthetic)
    oos = bt_preds[bt_preds.oos]
    daily = (oos[oos.eligible & (oos["rank"] <= settings.max_final_picks)].groupby("trade_date").close_ret.mean()
             - settings.round_trip_cost)
    edge_daily = {str(k): float(v) for k, v in daily.items()}
    model = Model.fit(bt_preds)
    y = (oos.close_ret > 0).astype(float)
    fs = study(bt_preds)
    calibration = {"edge_daily": edge_daily, "edge": edge_stats(edge_daily),
                   "oos_brier": brier(oos.p_up, y), "oos_base_brier": brier(np.full(len(oos), y.mean()), y),
                   "source": source}
    e = calibration["edge"]
    v = StrategyVersion(
        version="v1", parent_version=None, weights=model.spec, calibration=calibration, params={"source": source},
        status="champion", evaluation={"data_cutoff": str(max(bt_preds.trade_date)), "feature_study": fs,
                                       "coefficients": model.coef_table()},
        reason=(f"초기 전략: {source} 워크포워드 {oos.trade_date.nunique()}일 OOS → 그림자 상위{settings.max_final_picks} "
                f"일평균 순수익 {e.get('mean_net', 0) * 100:+.3f}% (t={_fmt(e.get('tstat'))}), 최종 모델은 전체 {model.spec['n_train']}건으로 학습"),
        created_at=now_kst(), activated_at=now_kst())
    session.add(v)
    session.commit()
    return v


def _fmt(x):
    return "-" if x is None else f"{x:.2f}"


def block_bootstrap(diff: pd.Series, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v = diff.values
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    means = v[idx].mean(1)
    return {"mean": float(v.mean()), "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)), "n_days": int(len(v))}


# ------------------------------------------------------------------ 재평가
def reevaluate(session: Session) -> dict:
    ch = champion(session)
    if ch is None:
        return {"status": "skipped", "reason": "champion 없음"}
    df = load_samples(session)
    if df.empty:
        return {"status": "deferred", "reason": "표본 없음"}
    cutoff = ch.evaluation.get("data_cutoff")
    dates = sorted(df.trade_date.unique())
    fresh = [d for d in dates if cutoff is None or str(d) > cutoff]
    hold = fresh[-settings.holdout_days:]
    hold_df = df[df.trade_date.isin(hold)]
    train = df[df.trade_date < (hold[0] if hold else dates[-1])]
    if len(hold) < 20 or len(hold_df) < settings.min_holdout_n:
        return {"status": "deferred", "reason": f"새 OOS 홀드아웃 부족 ({len(hold)}일, {len(hold_df)}건; "
                                                f"필요 ≥20일 & ≥{settings.min_holdout_n}건)"}
    if train.trade_date.nunique() < 40:
        return {"status": "deferred", "reason": "학습 표본 부족"}
    ch_end = ((ch.weights or {}).get("train_period") or [None, None])[1]
    if ch_end and str(train.trade_date.max()) <= ch_end:
        return {"status": "deferred", "reason": f"도전자 학습에 쓸 새 데이터 없음 (champion 학습 종료 {ch_end}, "
                                                f"홀드아웃 시작 {hold[0]}) — 홀드아웃은 소모하지 않음"}
    champ_m, chall_m = Model(ch.weights), Model.fit(train)
    sc, sx = score_frame(champ_m, hold_df), score_frame(chall_m, hold_df)
    k = settings.max_final_picks
    dc, dx = daily_shadow(sc, k), daily_shadow(sx, k)
    idx = dc.index.union(dx.index)
    diff = dx.reindex(idx).fillna(0) - dc.reindex(idx).fillna(0)
    bs = block_bootstrap(diff)
    y = (hold_df.close_ret > 0).astype(float)
    b_c, b_x = brier(sc.p_up, y), brier(sx.p_up, y)
    adopted = bs["ci95_low"] > 0 and b_x <= b_c + 0.001
    fs = study(df, oos_start=hold[0])
    evaluation = {"holdout_period": [str(hold[0]), str(hold[-1])], "n_holdout": int(len(hold_df)),
                  "n_train": int(len(train)), "data_cutoff": str(dates[-1]),
                  "champion": {"version": ch.version, "brier": b_c, "shadow": edge_stats({str(a): float(b) for a, b in dc.items()})},
                  "challenger": {"brier": b_x, "shadow": edge_stats({str(a): float(b) for a, b in dx.items()})},
                  "bootstrap_daily_net_diff": bs, "feature_study": fs, "coefficients": chall_m.coef_table()}
    now = now_kst()
    new_edge_daily = {str(a): float(b) for a, b in dx.items()}
    reason = (f"홀드아웃 {len(hold)}일: 그림자 상위{k} 일평균 순수익 champion {dc.mean() * 100:+.3f}% vs 도전자 "
              f"{dx.mean() * 100:+.3f}% (차이 95% CI {bs['ci95_low'] * 100:+.3f}~{bs['ci95_high'] * 100:+.3f}%p), "
              f"Brier {b_c:.4f}→{b_x:.4f} → " + ("채택" if adopted else "유의한 개선 아님, 거절"))
    session.add(StrategyVersion(
        version=_next_version(session), parent_version=ch.version, weights=chall_m.spec,
        calibration={"edge_daily": new_edge_daily, "edge": edge_stats(new_edge_daily), "oos_brier": b_x},
        params={"challenger": "retrained"}, status="champion" if adopted else "rejected", reason=reason,
        evaluation=evaluation, created_at=now, activated_at=now if adopted else None))
    if adopted:
        ch.status = "retired"
    else:
        # champion 의 OOS 증거 누적 (이 홀드아웃은 champion 학습에 쓰이지 않았음)
        ed = dict(ch.calibration.get("edge_daily", {}))
        ed.update({str(a): float(b) for a, b in dc.items()})
        ch.calibration = {**ch.calibration, "edge_daily": ed, "edge": edge_stats(ed)}
        ch.evaluation = {**ch.evaluation, "data_cutoff": str(dates[-1]), "last_feature_study": fs,
                         "last_monitoring": {"period": evaluation["holdout_period"], "brier": b_c}}
    session.commit()
    return {"status": "adopted" if adopted else "rejected", "reason": reason,
            **{k_: v for k_, v in evaluation.items() if k_ not in ("feature_study", "coefficients")}}
