"""해외 지수·환율·미국 금리 (yfinance).

미국장 d 일 종가는 한국시간 d+1 일 새벽(05:00~06:00)에 확정되므로 available_at = d+1 07:00 KST 로 보수적으로 둔다.
"""
from __future__ import annotations

from datetime import date, time, timedelta

from sqlalchemy.orm import Session

from ..db import IndexBar, now_kst
from .base import DataUnavailable, at, collection, num, upsert_ignore

SOURCE = "Yahoo Finance(yfinance)"
SYMBOLS = {
    "US:SPX": "^GSPC", "US:NASDAQ": "^IXIC", "US:SOX": "^SOX", "US:VIX": "^VIX",
    "US:10Y": "^TNX", "FX:USDKRW:YF": "KRW=X",
}
AVAILABLE_NEXT_DAY = time(7, 0)


def collect_global(session: Session, start: date, end: date) -> None:
    try:
        import yfinance as yf
    except ImportError as e:
        raise DataUnavailable(f"yfinance 미설치: {e}")
    retrieved = now_kst()
    for symbol, yf_sym in SYMBOLS.items():
        with collection(session, "yfinance", f"{symbol} {start}~{end}") as c:
            df = yf.download(yf_sym, start=start, end=end + timedelta(days=1), progress=False, auto_adjust=False)
            if df is None or df.empty:
                raise DataUnavailable(f"{yf_sym} 응답 없음")
            if hasattr(df.columns, "levels"):
                df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            rows = []
            for ts, r in df.iterrows():
                d = ts.date()
                avail = at(d, AVAILABLE_NEXT_DAY, plus_days=1)
                if retrieved < avail:
                    continue
                rows.append(dict(symbol=symbol, date=d, open=num(r.get("Open")), high=num(r.get("High")),
                                 low=num(r.get("Low")), close=num(r.get("Close")), volume=num(r.get("Volume")),
                                 source=SOURCE, retrieved_at=retrieved, data_timestamp=avail, available_at=avail))
            c.rows = upsert_ignore(session, IndexBar, rows, ["symbol", "date", "source"])
