"""워크포워드 Overnight 백테스트.

- 각 거래일 T 에 as_of = T 15:20(최종 단계) 뷰를 만들어 라이브와 동일한 피처/모델 코드를 실행한다.
  15:20 이후 데이터(종가 단일가, 당일 일봉 확정치 등)는 PITData 가 차단한다.
- 장중 스냅샷이 저장돼 있으면 사용, 없으면 장중 피처는 '데이터 없음'(전일 확정 데이터만 사용).
- 모델은 T 15:20 이전에 결과가 확정된 표본(target_date < T)으로만 RETRAIN_EVERY 일마다 재학습.
- 전략 우위 게이트도 T 이전의 OOS 그림자 포트폴리오 성과로만 판단.
- 진입: 15:20 가격(있으면) 기준 종가 단일가 매수 / 청산: 다음 거래일 시가 또는 종가. 수수료·세금·슬리피지 반영.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from ..config import settings
from ..evaluation.metrics import prediction_metrics, trading_metrics
from ..evaluation.outcomes import overnight
from ..features.builder import build_features
from ..features.regime import classify
from ..features.store import PITData
from ..prediction.calibration import brier, calibration_table
from ..prediction.engine import predict
from ..prediction.model import FEATURE_NAMES, Model

FINAL_TIME = dict((s[0], s[1]) for s in settings.stages)[settings.final_stage]


@dataclass
class BacktestParams:
    start: date | None = None
    end: date | None = None
    top_k: int = settings.max_final_picks
    exit: str = "close"                       # close | open
    fee_bps: float = settings.fee_bps
    tax_bps: float = settings.tax_bps
    slippage_bps: float = settings.slippage_bps
    retrain_every: int = settings.retrain_every
    warmup_days: int = 60
    min_train_days: int = 40


def shadow_edge(rows: pd.DataFrame, top_k: int, cost: float | None = None) -> dict:
    """그림자 포트폴리오: 매일 기대수익 상위 K(후보 자격 종목) 매수 → 다음날 종가 매도, 비용 차감. OOS 우위 판단용."""
    cost = settings.round_trip_cost if cost is None else cost
    d = rows[rows.eligible.astype(bool) & rows["rank"].notna() & (rows["rank"] <= top_k)]
    if d.empty:
        return {"days": 0}
    daily = d.groupby("trade_date").close_ret.mean() - cost
    n = len(daily)
    sd = daily.std(ddof=1) if n > 1 else np.nan
    ok = n > 1 and sd > 0
    return {"days": int(n), "mean_net": float(daily.mean()),
            "tstat": float(daily.mean() / sd * np.sqrt(n)) if ok else None,
            "sharpe": float(daily.mean() / sd * np.sqrt(252)) if ok else None,
            "win_rate": float((d.close_ret - cost > 0).mean()), "period": [str(daily.index[0]), str(daily.index[-1])]}


def walk_forward(data: PITData, params: BacktestParams, progress=None) -> pd.DataFrame:
    bars = data.bars_long
    nxt = {k: bars.pivot(index="date", columns="ticker", values=k).sort_index() for k in ("open", "high", "low", "close")}
    all_days = data.trading_dates()
    days = [d for d in all_days if (params.start is None or d >= params.start) and (params.end is None or d <= params.end)]
    rows: list[pd.DataFrame] = []
    hist: pd.DataFrame | None = None
    model, last_fit = Model(None), None
    for i, T in enumerate(days):
        pos = all_days.index(T)
        if pos < params.warmup_days or pos + 1 >= len(all_days):
            continue
        T1 = all_days[pos + 1]
        if hist is not None and (last_fit is None or (pos - last_fit) >= params.retrain_every):
            train = hist[hist.target_date < T]
            if train.trade_date.nunique() >= params.min_train_days:
                model, last_fit = Model.fit(train), pos
        edge = shadow_edge(hist[(hist.target_date < T) & hist.oos], params.top_k) if hist is not None else None
        fs = build_features(data.as_of(datetime.combine(T, FINAL_TIME)))
        if fs.table.empty or T not in nxt["close"].index or T1 not in nxt["close"].index:
            continue
        regime = classify(fs.market)
        res = predict(fs, regime, model, edge, final=True, explain=False)
        X = res.attrs["features"]
        res = res.join(X[[c for c in FEATURE_NAMES if c not in res.columns]])
        ref_close = nxt["close"].loc[T].reindex(res.index)
        entry = res.price.where(res.price_is_intraday, ref_close)
        o, h, l, c = (nxt[k].loc[T1].reindex(res.index) for k in ("open", "high", "low", "close"))
        oc = [overnight(e, a, b, cc, d_) if e == e and d_ == d_ else None for e, a, b, cc, d_ in zip(entry, o, h, l, c)]
        res["entry_basis"] = np.where(res.price_is_intraday, f"SNAPSHOT@{FINAL_TIME:%H:%M}", "CLOSE")
        for k in ("open_ret", "high_ret", "low_ret", "close_ret"):
            res[k] = [x[k] if x else np.nan for x in oc]
        res = res[res.close_ret.notna()].copy()
        res["trade_date"], res["target_date"] = T, T1
        res["regime"], res["trend"] = regime["label"], regime["trend"]
        res["oos"] = model.trained
        res["up"] = (res.close_ret > 0).astype(int)
        res["actual_return"] = res.close_ret
        res["result"] = np.where(res.close_ret.abs() < settings.neutral_band, "NEUTRAL",
                                 np.where((res.p_up >= 0.5) == (res.close_ret > 0), "SUCCESS", "FAILURE"))
        res = res.reset_index().rename(columns={"index": "ticker"})
        rows.append(res)
        hist = pd.concat([hist, res], ignore_index=True) if hist is not None else res
        if progress:
            progress(i, len(days), T)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _m(s):
    return None if len(s) == 0 or s.isna().all() else round(float(s.mean()), 5)


def simulate(preds: pd.DataFrame, params: BacktestParams, mode: str = "strategy") -> dict:
    """mode=strategy: 실제 후보 규칙(게이트 포함) / shadow: 게이트 무시, 매일 상위 K."""
    cost = (2 * params.fee_bps + params.tax_bps + 2 * params.slippage_bps) / 1e4
    oos = preds[preds.oos]
    days = sorted(oos.trade_date.unique())
    if mode == "strategy":
        picks = oos[oos.is_candidate]
    else:
        picks = oos[oos.eligible & oos["rank"].notna() & (oos["rank"] <= params.top_k)]
    picks = picks.sort_values(["trade_date", "rank"]).groupby("trade_date").head(params.top_k).copy()
    gross = picks.open_ret if params.exit == "open" else picks.close_ret
    picks["net_return"] = gross - cost
    daily = picks.groupby("trade_date").net_return.mean().reindex(days).fillna(0.0)
    picks["regime"] = picks.trend
    m = trading_metrics(daily, picks, oos.groupby("trade_date").trend.first())
    m.update(cost_per_trade=cost, exit=params.exit, mode=mode,
             avg_open_return=_m(picks.open_ret), avg_max_intraday_return=_m(picks.high_ret),
             avg_max_intraday_loss=_m(picks.low_ret), avg_close_return=_m(picks.close_ret),
             high_hit_rate=_m((picks.high_ret >= settings.high_hit_pct).astype(float)),
             gap_up_rate=_m((picks.open_ret > 0).astype(float)))
    return m


def run_backtest(data: PITData, params: BacktestParams, progress=None) -> tuple[dict, pd.DataFrame]:
    preds = walk_forward(data, params, progress)
    if preds.empty or not preds.oos.any():
        return {"message": "백테스트 가능한 기간 없음 (데이터 부족 또는 학습 표본 부족)"}, preds
    oos = preds[preds.oos]
    pm = prediction_metrics(oos.assign(exp_ret=oos.exp_ret.astype(float)))
    for tgt, col, y in (("gap", "p_gap_up", oos.open_ret > 0), ("hit", "p_high_hit", oos.high_ret >= settings.high_hit_pct)):
        y = y.astype(float)
        pm[f"{tgt}_brier"] = round(brier(oos[col], y), 5)
        pm[f"{tgt}_base_brier"] = round(brier(np.full(len(oos), y.mean()), y), 5)
        pm[f"{tgt}_calibration"] = calibration_table(oos[col], y)
    no_cand = int((oos.groupby("trade_date").is_candidate.sum() == 0).sum())
    return {"prediction": pm, "trading": simulate(preds, params, "strategy"),
            "shadow": simulate(preds, params, "shadow"), "no_candidate_days": no_cand,
            "oos_days": int(oos.trade_date.nunique()),
            "params": {k: (str(v) if isinstance(v, date) else v) for k, v in params.__dict__.items()},
            "note": (f"워크포워드 OOS 구간만 집계. 진입: {FINAL_TIME:%H:%M} 가격(없으면 당일 종가)으로 종가 단일가 매수, "
                     f"청산: 다음 거래일 {'시가' if params.exit == 'open' else '종가'}. "
                     "strategy = 실제 후보 규칙(우위 게이트 포함), shadow = 게이트 없이 매일 상위 K (참고용).")}, preds
