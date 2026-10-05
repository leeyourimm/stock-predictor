"""한국은행 ECOS OpenAPI: 원/달러 환율(매매기준율), 기준금리, 국고채 3년.

공표 시각을 정확히 알 수 없는 시계열은 data_date 다음날 09:00 을 available_at 으로 둔다(보수적).
"""
from __future__ import annotations

from datetime import date, datetime, time

import requests
from sqlalchemy.orm import Session

from ..config import settings
from ..db import IndexBar, now_kst
from .base import DataUnavailable, at, collection, num, upsert_ignore

SOURCE = "한국은행 ECOS"
URL = "https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/10000/{stat}/D/{start}/{end}/{item}"
SERIES = {
    "FX:USDKRW": ("731Y001", "0000001"),     # 원/달러 매매기준율
    "KR:BASE_RATE": ("722Y001", "0101000"),  # 한국은행 기준금리
    "KR:KTB3Y": ("817Y002", "010200000"),    # 국고채 3년
}


def collect_ecos(session: Session, start: date, end: date) -> None:
    retrieved = now_kst()
    for symbol, (stat, item) in SERIES.items():
        with collection(session, "ecos", f"{symbol} {start}~{end}") as c:
            if not settings.ecos_api_key:
                raise DataUnavailable("ECOS_API_KEY 미설정")
            r = requests.get(URL.format(key=settings.ecos_api_key, stat=stat, start=start.strftime("%Y%m%d"),
                                        end=end.strftime("%Y%m%d"), item=item), timeout=20)
            r.raise_for_status()
            body = r.json()
            data = body.get("StatisticSearch", {}).get("row")
            if not data:
                raise DataUnavailable(f"ECOS 응답에 데이터 없음: {str(body)[:300]}")
            rows = []
            for row in data:
                d = datetime.strptime(row["TIME"], "%Y%m%d").date()
                avail = at(d, time(9, 0), plus_days=1)
                if retrieved < avail:
                    continue
                rows.append(dict(symbol=symbol, date=d, close=num(row.get("DATA_VALUE")), source=SOURCE,
                                 retrieved_at=retrieved, data_timestamp=at(d, time(0, 0)), available_at=avail))
            c.rows = upsert_ignore(session, IndexBar, rows, ["symbol", "date", "source"])
