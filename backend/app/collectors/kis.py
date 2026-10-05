"""한국투자증권 KIS Open API: 14:00 장중 현재가 스냅샷 + 거래정지/관리종목 상태.

키(KIS_APP_KEY / KIS_APP_SECRET)가 없으면 스냅샷은 '데이터 없음'이 되고, 예측은 T-1 확정 데이터만으로 수행된다.
"""
from __future__ import annotations

import time as _time

import requests
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Instrument, IntradaySnapshot, now_kst
from .base import DataUnavailable, collection, num, upsert_ignore

SOURCE = "KIS Open API"
# iscd_stat_cls_code: 51 관리종목, 52 투자위험, 53 투자경고, 54 투자주의, 58 거래정지, 59 단기과열
ADMIN_CODES, HALT_CODES = {"51"}, {"58"}


def _token() -> str:
    if not (settings.kis_app_key and settings.kis_app_secret):
        raise DataUnavailable("KIS_APP_KEY/KIS_APP_SECRET 미설정")
    r = requests.post(f"{settings.kis_base_url}/oauth2/tokenP", json={
        "grant_type": "client_credentials", "appkey": settings.kis_app_key, "appsecret": settings.kis_app_secret},
        timeout=10)
    r.raise_for_status()
    return r.json()["access_token"]


def collect_snapshot(session: Session, tickers: list[str], max_per_sec: float = 15.0) -> None:
    with collection(session, "kis", f"intraday snapshot {len(tickers)} tickers") as c:
        token = _token()
        headers = {"authorization": f"Bearer {token}", "appkey": settings.kis_app_key,
                   "appsecret": settings.kis_app_secret, "tr_id": "FHKST01010100"}
        rows, inst = [], {i.ticker: i for i in session.query(Instrument).all()}
        for t in tickers:
            r = requests.get(f"{settings.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-price",
                             headers=headers, params={"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": t}, timeout=10)
            ts = now_kst()
            if r.ok and (out := r.json().get("output")):
                rows.append(dict(ticker=t, ts=ts, price=num(out.get("stck_prpr")), open=num(out.get("stck_oprc")),
                                 high=num(out.get("stck_hgpr")), low=num(out.get("stck_lwpr")),
                                 cum_volume=num(out.get("acml_vol")),
                                 cum_value=num(out.get("acml_tr_pbmn")), source=SOURCE, retrieved_at=ts,
                                 data_timestamp=ts, available_at=ts))
                if t in inst:
                    code = str(out.get("iscd_stat_cls_code", ""))
                    inst[t].is_admin_issue = code in ADMIN_CODES
                    inst[t].is_halted = code in HALT_CODES or out.get("temp_stop_yn") == "Y"
                    inst[t].status_source, inst[t].status_checked_at = SOURCE, ts
            _time.sleep(1.0 / max_per_sec)
        if not rows:
            raise DataUnavailable("KIS 스냅샷 응답 없음")
        c.rows = upsert_ignore(session, IntradaySnapshot, rows, ["ticker", "ts", "source"])


INDEX_CODES = {"IDX:KOSPI": "0001", "IDX:KOSDAQ": "1001"}


def collect_index_snapshot(session: Session) -> None:
    """KOSPI/KOSDAQ 장중 지수 (시장 대비 상대강도, Risk-on/off 판단용)."""
    with collection(session, "kis", "intraday index snapshot") as c:
        token = _token()
        headers = {"authorization": f"Bearer {token}", "appkey": settings.kis_app_key,
                   "appsecret": settings.kis_app_secret, "tr_id": "FHPUP02100000"}
        rows = []
        for sym, code in INDEX_CODES.items():
            r = requests.get(f"{settings.kis_base_url}/uapi/domestic-stock/v1/quotations/inquire-index-price",
                             headers=headers, params={"FID_COND_MRKT_DIV_CODE": "U", "FID_INPUT_ISCD": code}, timeout=10)
            ts = now_kst()
            if r.ok and (out := r.json().get("output")):
                rows.append(dict(ticker=sym, ts=ts, price=num(out.get("bstp_nmix_prpr")), open=num(out.get("bstp_nmix_oprc")),
                                 high=num(out.get("bstp_nmix_hgpr")), low=num(out.get("bstp_nmix_lwpr")),
                                 cum_volume=num(out.get("acml_vol")), cum_value=num(out.get("acml_tr_pbmn")),
                                 source=SOURCE, retrieved_at=ts, data_timestamp=ts, available_at=ts))
        if not rows:
            raise DataUnavailable("KIS 지수 스냅샷 응답 없음")
        c.rows = upsert_ignore(session, IntradaySnapshot, rows, ["ticker", "ts", "source"])


def import_minute_csv(session: Session, path: str, checkpoints: list) -> None:
    """외부(유료) 분봉 데이터로 과거 장중 스냅샷 백필.
    CSV 컬럼: ticker,datetime,open,high,low,close,volume,value (1분봉, KST). 각 체크포인트 시각 '이전에 끝난' 봉까지만 집계.
    available_at = 마지막 포함 봉의 종료 시각 → look-ahead 없음."""
    import pandas as pd
    with collection(session, "import", f"minute csv {path}") as c:
        df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["datetime"])
        df["date"] = df.datetime.dt.date
        rows = []
        for (t, d), g in df.sort_values("datetime").groupby(["ticker", "date"]):
            for cp in checkpoints:
                end = pd.Timestamp.combine(d, cp)
                gg = g[g.datetime + pd.Timedelta(minutes=1) <= end]
                if gg.empty:
                    continue
                ts = (gg.datetime.iloc[-1] + pd.Timedelta(minutes=1)).to_pydatetime()
                rows.append(dict(ticker=t, ts=ts, price=float(gg.close.iloc[-1]), open=float(gg.open.iloc[0]),
                                 high=float(gg.high.max()), low=float(gg.low.min()), cum_volume=float(gg.volume.sum()),
                                 cum_value=float(gg.value.sum()) if "value" in gg else None,
                                 source=f"IMPORT:{path.split('/')[-1]}", retrieved_at=now_kst(), data_timestamp=ts,
                                 available_at=ts))
        c.rows = upsert_ignore(session, IntradaySnapshot, rows, ["ticker", "ts", "source"])
