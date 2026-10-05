"""성과 지표 (라이브 채점 결과와 백테스트 결과에 공통 사용).

입력 DataFrame 컬럼: trade_date, p_up, up(0/1), actual_return, exp_ret, confidence, rank, regime, sector, result,
                    is_candidate
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..prediction.calibration import brier, calibration_table


def _hit(df: pd.DataFrame) -> dict:
    d = df[df.result.isin(["SUCCESS", "FAILURE"])]
    return {"n": int(len(d)), "hit_rate": None if d.empty else round(float((d.result == "SUCCESS").mean()), 4)}


def _bss(df: pd.DataFrame, base_rate: float | None = None) -> dict:
    if df.empty:
        return {"brier": None, "base_brier": None, "bss": None}
    b = brier(df.p_up, df.up)
    base = base_rate if base_rate is not None else float(df.up.mean())
    bb = brier(np.full(len(df), base), df.up)
    return {"brier": round(b, 5), "base_brier": round(bb, 5), "bss": round(1 - b / bb, 4) if bb > 0 else None}


def prediction_metrics(df: pd.DataFrame, recent_days: int = 20) -> dict:
    if df.empty:
        return {"n": 0, "message": "채점된 예측 없음"}
    df = df.sort_values("trade_date")
    dates = sorted(df.trade_date.unique())
    recent = df[df.trade_date.isin(dates[-recent_days:])]
    ranked = df[df["rank"].notna()]
    cand = df[df.is_candidate.fillna(False).astype(bool)] if "is_candidate" in df else df.iloc[0:0]
    # 기저율 Brier 는 '그 시점까지 알 수 있었던' 비율 대신 전체 비율을 쓰면 낙관적 → 보수적으로 0.5 와 실제 비율 중 나쁜 쪽을 보고
    out = {
        "n_predictions": int(len(df)), "n_days": len(dates),
        "period": [str(dates[0]), str(dates[-1])],
        "overall": _hit(df),
        f"recent_{recent_days}d": _hit(recent),
        "top5": _hit(ranked[ranked["rank"] <= 5]),
        "top10": _hit(ranked[ranked["rank"] <= 10]),
        "candidates": _hit(cand),
        "by_confidence": {c: _hit(g) for c, g in df.groupby("confidence")},
        "avg_expected_return": None if df.exp_ret.isna().all() else round(float(df.exp_ret.mean()), 5),
        "avg_actual_return": round(float(df.actual_return.mean()), 5),
        "top10_avg_expected_return": None if ranked[ranked["rank"] <= 10].empty else
        round(float(ranked[ranked["rank"] <= 10].exp_ret.mean()), 5),
        "top10_avg_actual_return": None if ranked[ranked["rank"] <= 10].empty else
        round(float(ranked[ranked["rank"] <= 10].actual_return.mean()), 5),
        "max_loss_top10": None if ranked[ranked["rank"] <= 10].empty else
        round(float(ranked[ranked["rank"] <= 10].actual_return.min()), 5),
        "max_loss_candidates": None if cand.empty else round(float(cand.actual_return.min()), 5),
        **_bss(df),
        "brier_vs_coinflip": round(brier(df.p_up, df.up) - 0.25, 5),
        "calibration": calibration_table(df.p_up, df.up),
        "by_regime": {r: {**_hit(g), **_bss(g)} for r, g in df.groupby("regime")},
    }
    if "sector" in df:
        out["by_sector"] = {s: _hit(g) for s, g in df.groupby(df.sector.fillna("미분류"))}
    daily = df.groupby("trade_date").apply(
        lambda g: pd.Series({"hit_rate": (g.result == "SUCCESS").sum() / max(1, g.result.isin(["SUCCESS", "FAILURE"]).sum()),
                             "brier": brier(g.p_up, g.up), "n": len(g)}), include_groups=False)
    out["daily"] = [{"date": str(d), **{k: round(float(v), 4) for k, v in r.items()}} for d, r in daily.tail(60).iterrows()]
    return out


def trading_metrics(daily_ret: pd.Series, trades: pd.DataFrame, regimes: pd.Series | None = None) -> dict:
    """daily_ret: 일별 포트폴리오 순수익률(비용 반영, 무포지션일 0). trades: 개별 거래(net_return 컬럼)."""
    if daily_ret.empty:
        return {"message": "거래 없음"}
    eq = (1 + daily_ret).cumprod()
    dd = eq / eq.cummax() - 1
    gains, losses = trades.net_return[trades.net_return > 0].sum(), -trades.net_return[trades.net_return < 0].sum()
    active = daily_ret[daily_ret != 0]
    sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else None
    idx = pd.to_datetime(pd.Series(daily_ret.index))
    monthly = (1 + daily_ret).groupby(idx.dt.to_period("M").astype(str).values).prod() - 1
    out = {
        "n_trades": int(len(trades)), "n_days": int(len(daily_ret)), "days_in_market": int((daily_ret != 0).sum()),
        "avg_trade_return": round(float(trades.net_return.mean()), 5) if len(trades) else None,
        "win_rate": round(float((trades.net_return > 0).mean()), 4) if len(trades) else None,
        "cumulative_return": round(float(eq.iloc[-1] - 1), 5),
        "mdd": round(float(dd.min()), 5),
        "profit_factor": round(float(gains / losses), 3) if losses > 0 else None,
        "sharpe": None if sharpe is None else round(sharpe, 3),
        "avg_daily_return_when_active": round(float(active.mean()), 5) if len(active) else None,
        "monthly": {k: round(float(v), 5) for k, v in monthly.items()},
        "equity_curve": [{"date": str(d), "equity": round(float(v), 5)} for d, v in eq.items()],
    }
    if regimes is not None:
        by = {}
        for reg, g in daily_ret.groupby(regimes.reindex(daily_ret.index).fillna("UNKNOWN")):
            t = trades[trades.regime == reg] if "regime" in trades else trades.iloc[0:0]
            by[reg] = {"days": int(len(g)), "cum_return": round(float((1 + g).prod() - 1), 5),
                       "n_trades": int(len(t)),
                       "win_rate": round(float((t.net_return > 0).mean()), 4) if len(t) else None}
        out["by_regime"] = by
    return out


def overnight_summary(df: pd.DataFrame, top_k: int, cost: float) -> dict:
    """최종 단계 예측의 Overnight 실거래 기준 성과. df: trade_date, rank, is_candidate, open/high/low/close_ret, trend."""
    out = {}
    if df.empty:
        return out
    days = sorted(df.trade_date.unique())
    regimes = df.groupby("trade_date").trend.first() if "trend" in df else None
    for name, picks in (("candidates", df[df.is_candidate.astype(bool)]),
                        ("shadow_topk", df[df["rank"].notna() & (df["rank"] <= top_k)])):
        rec = {"n_trades": int(len(picks))}
        if len(picks):
            for col, lbl in (("open_ret", "avg_open_return"), ("high_ret", "avg_max_intraday_return"),
                             ("low_ret", "avg_max_intraday_loss"), ("close_ret", "avg_close_return")):
                rec[lbl] = round(float(picks[col].mean()), 5) if col in picks and picks[col].notna().any() else None
            rec["avg_net_close"] = round(float((picks.close_ret - cost).mean()), 5)
            rec["avg_net_open"] = round(float((picks.open_ret - cost).mean()), 5) if picks.open_ret.notna().any() else None
            for exit_col in ("close_ret", "open_ret"):
                p = picks.assign(net_return=picks[exit_col] - cost, regime=picks.get("trend"))
                daily = p.groupby("trade_date").net_return.mean().reindex(days).fillna(0.0)
                tm = trading_metrics(daily, p, regimes)
                tm.pop("equity_curve", None)
                rec["exit_" + exit_col.split("_")[0]] = tm
        out[name] = rec
    return out
