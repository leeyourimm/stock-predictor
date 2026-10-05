"""Overnight 결과 계산 (사후 채점·백테스트 공통). 진입가 대비 다음 거래일 시가/고가/저가/종가 수익률과 비용 차감 수익."""
from __future__ import annotations

from ..config import settings


def overnight(entry: float, o: float | None, h: float | None, l: float | None, c: float) -> dict:
    cost = settings.round_trip_cost

    def r(x):
        return None if x is None or x != x or not entry else x / entry - 1

    out = {"open_ret": r(o), "high_ret": r(h), "low_ret": r(l), "close_ret": r(c)}
    out["net_open"] = None if out["open_ret"] is None else out["open_ret"] - cost
    out["net_close"] = out["close_ret"] - cost
    out["gap_up"] = None if out["open_ret"] is None else out["open_ret"] > 0
    out["high_hit"] = None if out["high_ret"] is None else out["high_ret"] >= settings.high_hit_pct
    return out
