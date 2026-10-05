"""KRX 거래일 캘린더.

과거 거래일은 KOSPI 지수 일봉이 존재하는 날로 판정한다(가장 정확).
미래(다음 거래일)는 주말 + 아래 휴장일 목록으로 계산한다. 목록은 매년 KRX 공지로 갱신해야 하며,
KRX_HOLIDAYS 환경변수(쉼표구분 YYYY-MM-DD)로 추가할 수 있다. 예측 대상일이 실제와 다르면 채점은 실제 다음 거래일 기준으로 한다.
"""
from __future__ import annotations

import os
from datetime import date, timedelta

# KRX 휴장일 (대체공휴일 포함, 연말 휴장 포함). 출처: KRX 휴장일 공지 기준으로 매년 확인 필요.
KRX_HOLIDAYS = {
    # 2025
    "2025-01-01", "2025-01-27", "2025-01-28", "2025-01-29", "2025-01-30", "2025-03-03", "2025-05-01", "2025-05-05",
    "2025-05-06", "2025-06-03", "2025-06-06", "2025-08-15", "2025-10-03", "2025-10-06", "2025-10-07", "2025-10-08",
    "2025-10-09", "2025-12-25", "2025-12-31",
    # 2026
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18", "2026-03-02", "2026-05-01", "2026-05-05", "2026-05-25",
    "2026-06-03", "2026-08-17", "2026-09-24", "2026-09-25", "2026-10-05", "2026-10-09", "2026-12-25", "2026-12-31",
}
KRX_HOLIDAYS |= {x.strip() for x in os.getenv("KRX_HOLIDAYS", "").split(",") if x.strip()}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in KRX_HOLIDAYS


def next_trading_day(d: date, known_days: list[date] | None = None) -> date:
    if known_days:
        later = [x for x in known_days if x > d]
        if later:
            return later[0]
    n = d + timedelta(days=1)
    while not is_trading_day(n):
        n += timedelta(days=1)
    return n


def previous_trading_day(d: date) -> date:
    n = d - timedelta(days=1)
    while not is_trading_day(n):
        n -= timedelta(days=1)
    return n
