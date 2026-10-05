"""DART 전자공시 OpenAPI: 공시 목록.

DART 목록 API 는 접수 '일자'만 제공한다. 따라서
- 실시간 수집(조회 시각에 이미 목록에 있음): available_at = 조회 시각
- 사후 백필: available_at = 접수일 23:59 (같은 날 14:00 예측에는 쓰지 않음 → look-ahead 방지)
"""
from __future__ import annotations

from datetime import date, datetime, time

import requests
from sqlalchemy.orm import Session

from ..config import settings
from ..db import Disclosure, now_kst
from .base import DataUnavailable, at, collection, upsert_ignore

SOURCE = "DART OpenAPI"
URL = "https://opendart.fss.or.kr/api/list.json"


def collect_disclosures(session: Session, start: date, end: date) -> None:
    with collection(session, "dart", f"disclosures {start}~{end}") as c:
        if not settings.dart_api_key:
            raise DataUnavailable("DART_API_KEY 미설정")
        retrieved = now_kst()
        rows, page = [], 1
        while True:
            r = requests.get(URL, params=dict(crtfc_key=settings.dart_api_key, bgn_de=start.strftime("%Y%m%d"),
                                              end_de=end.strftime("%Y%m%d"), page_no=page, page_count=100),
                             timeout=20)
            r.raise_for_status()
            body = r.json()
            if body.get("status") == "013":   # 조회된 데이터 없음
                break
            if body.get("status") != "000":
                raise DataUnavailable(f"DART 오류 {body.get('status')}: {body.get('message')}")
            for it in body.get("list", []):
                d = datetime.strptime(it["rcept_dt"], "%Y%m%d").date()
                end_of_day = at(d, time(23, 59))
                avail = retrieved if d == retrieved.date() else end_of_day
                rows.append(dict(rcept_no=it["rcept_no"], ticker=(it.get("stock_code") or None),
                                 corp_name=it.get("corp_name", ""), report_nm=it.get("report_nm", ""), rcept_dt=d,
                                 source=SOURCE, retrieved_at=retrieved, data_timestamp=at(d, time(0, 0)),
                                 available_at=avail))
            if page >= int(body.get("total_page", 1)):
                break
            page += 1
        c.rows = upsert_ignore(session, Disclosure, rows, ["rcept_no"])
