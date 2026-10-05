"""보조 출처: FinanceDataReader(가격 교차검증, 관리종목 목록)."""
from __future__ import annotations

from datetime import date

from sqlalchemy.orm import Session

from ..config import settings
from ..db import DailyBar, Instrument, now_kst
from .base import DataUnavailable, at, collection, num, upsert_ignore

SOURCE = "FinanceDataReader"


def _fdr():
    try:
        import FinanceDataReader as fdr
    except ImportError as e:
        raise DataUnavailable(f"FinanceDataReader 미설치: {e}")
    return fdr


def collect_crosscheck_bars(session: Session, tickers: list[str], start: date, end: date) -> None:
    """일부 종목(예: 상위 후보)에 대해 2차 출처 일봉을 받아 data_conflicts 탐지에 사용."""
    fdr = _fdr()
    retrieved = now_kst()
    with collection(session, "fdr", f"crosscheck bars {len(tickers)} tickers {start}~{end}") as c:
        rows = []
        for t in tickers:
            df = fdr.DataReader(t, start, end)
            for ts, r in df.iterrows():
                d = ts.date()
                avail = at(d, settings.bar_available_time)
                if retrieved < avail:
                    continue
                rows.append(dict(ticker=t, date=d, open=num(r.get("Open")), high=num(r.get("High")),
                                 low=num(r.get("Low")), close=num(r.get("Close")), volume=num(r.get("Volume")),
                                 value=None, change_pct=None, market_cap=None, source=SOURCE,
                                 retrieved_at=retrieved, data_timestamp=avail, available_at=avail))
        c.rows = upsert_ignore(session, DailyBar, rows, ["ticker", "date", "source"])


def collect_admin_status(session: Session) -> None:
    """관리종목 목록. 조회 실패 시 상태를 '확인 불가(None)'로 남긴다 (False 로 간주하지 않음)."""
    fdr = _fdr()
    with collection(session, "fdr", "administrative issues") as c:
        df = fdr.StockListing("KRX-ADMINISTRATIVE")
        if df is None or df.empty:
            raise DataUnavailable("관리종목 목록 비어 있음")
        code_col = next(col for col in df.columns if col.lower() in ("code", "symbol", "종목코드"))
        admin = set(df[code_col].astype(str).str.zfill(6))
        checked = now_kst()
        for inst in session.query(Instrument).all():
            inst.is_admin_issue = inst.ticker in admin
            inst.status_source = SOURCE
            inst.status_checked_at = checked
        c.rows = len(admin)
