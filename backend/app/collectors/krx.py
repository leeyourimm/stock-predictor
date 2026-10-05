"""KRX 정보데이터시스템 수집기 (pykrx 경유).

pykrx 최신 버전은 KRX 데이터 포털 로그인(KRX_ID / KRX_PW 환경변수)이 필요하다.
- 일봉(전 종목, 일자별 1회 호출), 시가총액
- 투자자별 순매수(외국인/기관합계/개인)
- 공매도 거래량/잔고
- KOSPI/KOSDAQ/업종 지수
"""
from __future__ import annotations

import os
from datetime import date, datetime

from sqlalchemy.orm import Session

from ..config import settings
from ..db import DailyBar, IndexBar, Instrument, InvestorFlow, ShortData, now_kst
from .base import DataUnavailable, at, collection, num, upsert_ignore

SOURCE = "KRX(pykrx)"

INDEX_CODES = {"KOSPI": "1001", "KOSDAQ": "2001"}
# KOSPI 업종지수(전기전자, 화학 등). 구성종목으로 종목→업종 매핑에 사용.
KOSPI_SECTOR_INDEXES = [f"10{n:02d}" for n in range(5, 29)]
KOSDAQ_SECTOR_INDEXES = [f"20{n:02d}" for n in range(12, 40)]


def _stock():
    if settings.krx_id and settings.krx_pw:
        os.environ.setdefault("KRX_ID", settings.krx_id)
        os.environ.setdefault("KRX_PW", settings.krx_pw)
    try:
        from pykrx import stock
    except ImportError as e:
        raise DataUnavailable(f"pykrx 미설치: {e}")
    return stock


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


def _guard_final(d: date, available: datetime) -> None:
    """장중에 당일 일봉(미확정치)을 저장하지 않는다."""
    if now_kst() < available:
        raise DataUnavailable(f"{d} 데이터는 {available} 이후 확정됨 (현재 {now_kst()})")


def trading_days(start: date, end: date) -> list[date]:
    stock = _stock()
    try:
        idx = stock.get_index_ohlcv_by_date(_ymd(start), _ymd(end), INDEX_CODES["KOSPI"])
    except Exception as e:  # noqa: BLE001
        raise DataUnavailable(f"KRX 거래일 조회 실패: {e}")
    if idx is None or idx.empty:
        raise DataUnavailable("KRX 거래일 데이터 없음")
    return [ts.date() for ts in idx.index]


def collect_instruments(session: Session, d: date) -> None:
    stock = _stock()
    with collection(session, "krx", f"instruments {d}") as c:
        rows = []
        for market in settings.universe_markets:
            for t in stock.get_market_ticker_list(_ymd(d), market=market):
                rows.append({"ticker": t, "name": stock.get_market_ticker_name(t), "market": market})
        if not rows:
            raise DataUnavailable("상장종목 목록 없음")
        existing = {i.ticker: i for i in session.query(Instrument).all()}
        for r in rows:
            inst = existing.get(r["ticker"])
            if inst:
                inst.name, inst.market = r["name"], r["market"]
            else:
                session.add(Instrument(**r))
        c.rows = len(rows)
    _collect_sector_map(session, d)


def _collect_sector_map(session: Session, d: date) -> None:
    stock = _stock()
    with collection(session, "krx", f"sector map {d}") as c:
        mapping: dict[str, tuple[str, str]] = {}
        for code in KOSPI_SECTOR_INDEXES + KOSDAQ_SECTOR_INDEXES:
            try:
                members = stock.get_index_portfolio_deposit_file(code, _ymd(d))
                name = stock.get_index_ticker_name(code)
            except Exception:  # noqa: BLE001 - 존재하지 않는 업종코드는 건너뜀
                continue
            for t in members or []:
                mapping.setdefault(t, (code, name))
        if not mapping:
            raise DataUnavailable("업종 구성종목 조회 결과 없음")
        for inst in session.query(Instrument).all():
            if inst.ticker in mapping:
                inst.sector_index, inst.sector = mapping[inst.ticker]
        c.rows = len(mapping)


def collect_daily(session: Session, d: date) -> None:
    """d 일자의 전 종목 일봉 + 시총, 투자자별 순매수, 공매도, 지수."""
    stock = _stock()
    ymd = _ymd(d)
    retrieved = now_kst()
    bar_avail = at(d, settings.bar_available_time)
    flow_avail = at(d, settings.flow_available_time)

    with collection(session, "krx", f"ohlcv {d}") as c:
        _guard_final(d, bar_avail)
        rows = []
        for market in settings.universe_markets:
            df = stock.get_market_ohlcv_by_ticker(ymd, market=market)
            cap = stock.get_market_cap_by_ticker(ymd, market=market)
            if df is None or df.empty:
                continue
            for t, r in df.iterrows():
                if num(r.get("거래량")) in (None, 0.0) and num(r.get("시가")) in (None, 0.0):
                    continue  # 거래정지 등으로 체결 없음 → 행을 만들지 않음
                rows.append(dict(
                    ticker=t, date=d, open=num(r.get("시가")), high=num(r.get("고가")), low=num(r.get("저가")),
                    close=num(r.get("종가")), volume=num(r.get("거래량")), value=num(r.get("거래대금")),
                    change_pct=num(r.get("등락률")),
                    market_cap=num(cap.loc[t, "시가총액"]) if cap is not None and t in cap.index else None,
                    source=SOURCE, retrieved_at=retrieved, data_timestamp=at(d, settings.bar_available_time),
                    available_at=bar_avail))
        if not rows:
            raise DataUnavailable(f"{d} 일봉 없음(휴장일?)")
        c.rows = upsert_ignore(session, DailyBar, rows, ["ticker", "date", "source"])

    with collection(session, "krx", f"investor flows {d}") as c:
        _guard_final(d, flow_avail)
        acc: dict[str, dict] = {}
        for market in settings.universe_markets:
            for inv, col in (("외국인", "foreign_net"), ("기관합계", "institution_net"), ("개인", "individual_net")):
                df = stock.get_market_net_purchases_of_equities_by_ticker(ymd, ymd, market, inv)
                if df is None or df.empty:
                    continue
                for t, r in df.iterrows():
                    acc.setdefault(t, {})[col] = num(r.get("순매수거래대금"))
        rows = [dict(ticker=t, date=d, foreign_net=v.get("foreign_net"), institution_net=v.get("institution_net"),
                     individual_net=v.get("individual_net"), source=SOURCE, retrieved_at=retrieved,
                     data_timestamp=flow_avail, available_at=flow_avail) for t, v in acc.items()]
        if not rows:
            raise DataUnavailable(f"{d} 투자자별 매매 데이터 없음")
        c.rows = upsert_ignore(session, InvestorFlow, rows, ["ticker", "date", "source"])

    with collection(session, "krx", f"short selling {d}") as c:
        _guard_final(d, flow_avail)
        acc = {}
        for market in settings.universe_markets:
            vol = stock.get_shorting_volume_by_ticker(ymd, market=market)
            if vol is not None and not vol.empty:
                vcol = next((x for x in vol.columns if "공매도" in str(x)), vol.columns[0])
                for t, r in vol.iterrows():
                    acc.setdefault(t, {})["short_volume"] = num(r[vcol])
        rows = [dict(ticker=t, date=d, short_volume=v.get("short_volume"), short_value=None, short_balance=None,
                     source=SOURCE, retrieved_at=retrieved, data_timestamp=flow_avail, available_at=flow_avail)
                for t, v in acc.items()]
        if not rows:
            raise DataUnavailable(f"{d} 공매도 거래량 데이터 없음")
        c.rows = upsert_ignore(session, ShortData, rows, ["ticker", "date", "source"])

    collect_indexes(session, d, d)


def collect_indexes(session: Session, start: date, end: date) -> None:
    stock = _stock()
    retrieved = now_kst()
    codes = dict(INDEX_CODES)
    sectors = {i for (i,) in session.query(Instrument.sector_index).distinct() if i}
    codes.update({f"SECTOR:{s}": s for s in sectors})
    for symbol, code in codes.items():
        with collection(session, "krx", f"index {symbol} {start}~{end}") as c:
            df = stock.get_index_ohlcv_by_date(_ymd(start), _ymd(end), code, name_display=False)
            if df is None or df.empty:
                raise DataUnavailable(f"지수 {code} 데이터 없음")
            rows = []
            for ts, r in df.iterrows():
                d = ts.date()
                avail = at(d, settings.bar_available_time)
                if retrieved < avail:
                    continue  # 장중 미확정치 저장 금지
                rows.append(dict(symbol=symbol, date=d, open=num(r.get("시가")), high=num(r.get("고가")),
                                 low=num(r.get("저가")), close=num(r.get("종가")), volume=num(r.get("거래량")),
                                 source=SOURCE, retrieved_at=retrieved, data_timestamp=avail, available_at=avail))
            c.rows = upsert_ignore(session, IndexBar, rows, ["symbol", "date", "source"])
