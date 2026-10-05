"""네이버 금융 수집기 (로그인 불필요). KRX 데이터 포털이 로그인 필수가 되면서 기본 데이터 소스로 사용.

- 종목 목록: 시가총액 순위 페이지 (KOSPI/KOSDAQ) 상위 N
- 일봉: api.finance.naver.com/siseJson (시가/고가/저가/종가/거래량). 거래대금은 제공되지 않아 종가×거래량 근사치를 쓴다.
- 외국인/기관 순매매: finance.naver.com/item/frgn (일별 순매매 '수량') → 그날 종가를 곱해 금액 근사. 개인은 제공되지 않음(데이터 없음).
- 지수: siseJson symbol=KOSPI/KOSDAQ
- 실행 시점 현재가: polling.finance.naver.com 실시간 시세 (장중 스냅샷)

공개 시각 규칙은 KRX 와 동일하게 적용한다(일봉 16:00, 수급 18:00). 당일 미확정 값은 저장하지 않는다.
"""
from __future__ import annotations

import re
import time as _time
from datetime import date, datetime, timedelta

import requests
from sqlalchemy.orm import Session

from ..config import settings
from ..db import DailyBar, IndexBar, Instrument, IntradaySnapshot, InvestorFlow, now_kst
from .base import DataUnavailable, at, collection, num, upsert_ignore

SOURCE = "NAVER 금융"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126.0 Safari/537.36", "Referer": "https://finance.naver.com/"}
MARKET_SOSOK = {"KOSPI": 0, "KOSDAQ": 1}
DEFAULT_TOP_N = 300
_session = requests.Session()
_session.headers.update(UA)


def _get(url: str, params: dict | None = None, encoding: str | None = None, pause: float = 0.15) -> str:
    for attempt in range(3):
        try:
            r = _session.get(url, params=params, timeout=15)
            if r.status_code == 200:
                if encoding:
                    r.encoding = encoding
                _time.sleep(pause)
                return r.text
        except requests.RequestException:
            pass
        _time.sleep(1.5 * (attempt + 1))
    raise DataUnavailable(f"네이버 응답 실패: {url}")


def _ymd(d: date) -> str:
    return d.strftime("%Y%m%d")


# ------------------------------------------------------------------ 파서 (테스트 가능하도록 분리)
_SISE_ROW = re.compile(r'\[\s*"(\d{8})"\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)')


def parse_sise_json(text: str) -> list[dict]:
    """siseJson 응답 → [{date, open, high, low, close, volume}]"""
    out = []
    for m in _SISE_ROW.finditer(text):
        d = datetime.strptime(m.group(1), "%Y%m%d").date()
        o, h, l, c, v = (float(m.group(i)) for i in range(2, 7))
        out.append({"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return out


_CODE_ROW = re.compile(r'/item/main\.naver\?code=(\d{6})"\s+class="tltle">([^<]+)</a>')


def parse_market_sum(html: str) -> list[tuple[str, str]]:
    """시가총액 순위 페이지 → [(종목코드, 종목명)] (페이지 내 순위 순서)"""
    seen, out = set(), []
    for code, name in _CODE_ROW.findall(html):
        if code not in seen:
            seen.add(code)
            out.append((code, name.strip()))
    return out


_TAG = re.compile(r"<[^>]+>")


def _cells(tr: str) -> list[str]:
    return [re.sub(r"\s+", " ", _TAG.sub("", c)).replace("&nbsp;", "").strip()
            for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]


def parse_frgn(html: str) -> list[dict]:
    """외국인·기관 순매매 페이지 → [{date, close, volume, inst_net_shares, foreign_net_shares}]
    열: 날짜 | 종가 | 전일비 | 등락률 | 거래량 | 기관 순매매량 | 외국인 순매매량 | 보유주수 | 보유율"""
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S):
        c = _cells(tr)
        if len(c) < 7 or not re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", c[0]):
            continue
        out.append({"date": datetime.strptime(c[0], "%Y.%m.%d").date(), "close": num(c[1]), "volume": num(c[4]),
                    "inst_net_shares": num(c[5].replace("+", "")), "foreign_net_shares": num(c[6].replace("+", ""))})
    return out


# ------------------------------------------------------------------ 수집
def collect_instruments(session: Session) -> list[str]:
    top_n = settings.universe_top_n or DEFAULT_TOP_N
    with collection(session, "naver", f"instruments top {top_n}") as c:
        ranked: list[tuple[str, str, str]] = []
        for market in settings.universe_markets:
            sosok = MARKET_SOSOK.get(market.strip().upper())
            if sosok is None:
                continue
            for page in range(1, 41):
                rows = parse_market_sum(_get("https://finance.naver.com/sise/sise_market_sum.naver",
                                             {"sosok": sosok, "page": page}, encoding="euc-kr"))
                if not rows:
                    break
                ranked += [(code, name, market) for code, name in rows]
                if sum(1 for r in ranked if r[2] == market) >= top_n:
                    break
        if not ranked:
            raise DataUnavailable("네이버 시가총액 순위 페이지에서 종목을 찾지 못함")
        # 시장별 순위를 번갈아 합쳐 시총 상위 top_n (정확한 통합 순위는 시총 값이 필요하지만 근사로 충분)
        per = {m: [r for r in ranked if r[2] == m] for m in settings.universe_markets}
        share = {m: max(1, round(top_n * (0.7 if m == "KOSPI" else 0.3))) for m in per} if len(per) > 1 else {m: top_n for m in per}
        chosen = [r for m, rs in per.items() for r in rs[:share[m]]]
        existing = {i.ticker: i for i in session.query(Instrument).all()}
        for code, name, market in chosen:
            if code in existing:
                existing[code].name, existing[code].market = name, market
            else:
                session.add(Instrument(ticker=code, name=name, market=market))
        c.rows = len(chosen)
    return [r[0] for r in chosen]


def _bars_rows(ticker: str, start: date, end: date, retrieved: datetime) -> list[dict]:
    rows = []
    for r in parse_sise_json(_get("https://api.finance.naver.com/siseJson.naver",
                                  {"symbol": ticker, "requestType": 1, "startTime": _ymd(start),
                                   "endTime": _ymd(end), "timeframe": "day"})):
        avail = at(r["date"], settings.bar_available_time)
        if retrieved < avail or not r["volume"]:
            continue  # 장중 미확정 / 체결 없음(거래정지 등)
        rows.append(dict(ticker=ticker, date=r["date"], open=r["open"], high=r["high"], low=r["low"], close=r["close"],
                         volume=r["volume"], value=r["close"] * r["volume"], change_pct=None, market_cap=None,
                         source=SOURCE, retrieved_at=retrieved, data_timestamp=avail, available_at=avail))
    return rows


def _flow_rows(ticker: str, start: date, retrieved: datetime, max_pages: int = 25) -> list[dict]:
    rows = []
    for page in range(1, max_pages + 1):
        recs = parse_frgn(_get("https://finance.naver.com/item/frgn.naver", {"code": ticker, "page": page},
                               encoding="euc-kr"))
        if not recs:
            break
        for r in recs:
            avail = at(r["date"], settings.flow_available_time)
            if r["date"] < start or retrieved < avail or r["close"] is None:
                continue
            rows.append(dict(ticker=ticker, date=r["date"],
                             foreign_net=None if r["foreign_net_shares"] is None else r["foreign_net_shares"] * r["close"],
                             institution_net=None if r["inst_net_shares"] is None else r["inst_net_shares"] * r["close"],
                             individual_net=None, source=SOURCE, retrieved_at=retrieved, data_timestamp=avail,
                             available_at=avail))
        if min(r["date"] for r in recs) < start:
            break
    return rows


def collect_range(session: Session, start: date, end: date) -> None:
    """start~end 기간의 일봉·수급(종목별) + 지수. 종목 목록이 비어 있거나 30일 넘게 지났으면 갱신."""
    tickers = [i.ticker for i in session.query(Instrument).filter(~Instrument.ticker.like("IDX:%")).all()]
    if not tickers:
        tickers = collect_instruments(session)
    if not tickers:
        return
    retrieved = now_kst()
    collect_indexes(session, start, end)
    with collection(session, "naver", f"daily bars {len(tickers)} tickers {start}~{end}") as c:
        n = 0
        for t in tickers:
            n += upsert_ignore(session, DailyBar, _bars_rows(t, start, end, retrieved), ["ticker", "date", "source"])
        session.commit()
        if n == 0 and (end - start).days > 5:
            raise DataUnavailable("네이버 일봉 응답 없음")
        c.rows = n
    with collection(session, "naver", f"investor flows {len(tickers)} tickers {start}~{end}") as c:
        n = 0
        pages = max(1, min(25, (end - start).days // 25 + 2))
        for t in tickers:
            n += upsert_ignore(session, InvestorFlow, _flow_rows(t, start, retrieved, pages), ["ticker", "date", "source"])
        session.commit()
        c.rows = n


def collect_indexes(session: Session, start: date, end: date) -> None:
    retrieved = now_kst()
    for sym in ("KOSPI", "KOSDAQ"):
        with collection(session, "naver", f"index {sym} {start}~{end}") as c:
            recs = parse_sise_json(_get("https://api.finance.naver.com/siseJson.naver",
                                        {"symbol": sym, "requestType": 1, "startTime": _ymd(start - timedelta(days=5)),
                                         "endTime": _ymd(end), "timeframe": "day"}))
            rows = []
            for r in recs:
                avail = at(r["date"], settings.bar_available_time)
                if retrieved < avail:
                    continue
                rows.append(dict(symbol=sym, date=r["date"], open=r["open"], high=r["high"], low=r["low"],
                                 close=r["close"], volume=r["volume"], source=SOURCE, retrieved_at=retrieved,
                                 data_timestamp=avail, available_at=avail))
            if not rows:
                raise DataUnavailable(f"네이버 지수 {sym} 데이터 없음")
            c.rows = upsert_ignore(session, IndexBar, rows, ["symbol", "date", "source"])


def trading_days(start: date, end: date) -> list[date]:
    recs = parse_sise_json(_get("https://api.finance.naver.com/siseJson.naver",
                                {"symbol": "KOSPI", "requestType": 1, "startTime": _ymd(start), "endTime": _ymd(end),
                                 "timeframe": "day"}))
    if not recs:
        raise DataUnavailable("네이버 거래일 조회 실패")
    return [r["date"] for r in recs]


# ------------------------------------------------------------------ 실행 시점 현재가 (장중 스냅샷)
def _rt_num(d: dict, *keys):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return num(d[k])
    return None


def collect_snapshot(session: Session, tickers: list[str]) -> None:
    with collection(session, "naver", f"realtime snapshot {len(tickers)} tickers") as c:
        rows = []
        for t in tickers:
            try:
                js = _session.get(f"https://polling.finance.naver.com/api/realtime/domestic/stock/{t}", timeout=10).json()
            except Exception:  # noqa: BLE001
                continue
            ts = now_kst()
            for d in js.get("datas", [])[:1]:
                price = _rt_num(d, "closePriceRaw", "closePrice")
                if price is None:
                    continue
                rows.append(dict(ticker=t, ts=ts, price=price, open=_rt_num(d, "openPriceRaw", "openPrice"),
                                 high=_rt_num(d, "highPriceRaw", "highPrice"), low=_rt_num(d, "lowPriceRaw", "lowPrice"),
                                 cum_volume=_rt_num(d, "accumulatedTradingVolumeRaw", "accumulatedTradingVolume"),
                                 cum_value=_rt_num(d, "accumulatedTradingValueRaw"),
                                 source=SOURCE, retrieved_at=ts, data_timestamp=ts, available_at=ts))
            _time.sleep(0.1)
        if not rows:
            raise DataUnavailable("네이버 실시간 시세 응답 없음")
        c.rows = upsert_ignore(session, IntradaySnapshot, rows, ["ticker", "ts", "source"])


def collect_index_snapshot(session: Session) -> None:
    with collection(session, "naver", "realtime index snapshot") as c:
        rows = []
        for sym in ("KOSPI", "KOSDAQ"):
            try:
                js = _session.get(f"https://polling.finance.naver.com/api/realtime/domestic/index/{sym}", timeout=10).json()
            except Exception:  # noqa: BLE001
                continue
            ts = now_kst()
            for d in js.get("datas", [])[:1]:
                price = _rt_num(d, "closePriceRaw", "closePrice")
                if price is not None:
                    rows.append(dict(ticker=f"IDX:{sym}", ts=ts, price=price, open=_rt_num(d, "openPriceRaw", "openPrice"),
                                     high=_rt_num(d, "highPriceRaw", "highPrice"), low=_rt_num(d, "lowPriceRaw", "lowPrice"),
                                     cum_volume=None, cum_value=None, source=SOURCE, retrieved_at=ts, data_timestamp=ts,
                                     available_at=ts))
        if not rows:
            raise DataUnavailable("네이버 실시간 지수 응답 없음")
        c.rows = upsert_ignore(session, IntradaySnapshot, rows, ["ticker", "ts", "source"])
