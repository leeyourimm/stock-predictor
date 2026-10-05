"""시장 Regime 분류 (KOSPI 기준, 코드 계산).

추세: 60일 수익률과 60일 이동평균 위치 → BULL / BEAR / SIDEWAYS
변동성: 20일 실현변동성의 과거 분위수 → HIGH_VOL(>=70%) / LOW_VOL(<=30%) / NORMAL_VOL
"""
from __future__ import annotations

TREND_BAND = 0.05


def classify(market: dict) -> dict:
    r60, above = market.get("kospi_ret60"), market.get("kospi_above_ma60")
    if r60 is None or above is None:
        trend = "UNKNOWN"
    elif r60 > TREND_BAND and above:
        trend = "BULL"
    elif r60 < -TREND_BAND and not above:
        trend = "BEAR"
    else:
        trend = "SIDEWAYS"
    pct = market.get("kospi_vol_pctile")
    if pct is None:
        vol = "UNKNOWN_VOL"
    elif pct >= 0.7:
        vol = "HIGH_VOL"
    elif pct <= 0.3:
        vol = "LOW_VOL"
    else:
        vol = "NORMAL_VOL"
    return {"trend": trend, "vol": vol, "label": f"{trend}|{vol}",
            "inputs": {"kospi_ret60": r60, "kospi_above_ma60": above, "kospi_vol_pctile": pct,
                       "kospi_vol20": market.get("kospi_vol20")}}
