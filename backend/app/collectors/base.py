"""수집기 공통 유틸.

원칙
- 값이 없으면 만들지 않는다: 실패/미설정은 CollectionLog 에 UNAVAILABLE/ERROR 로만 남긴다.
- 모든 행에 source, retrieved_at, data_timestamp, available_at 을 기록한다.
- available_at = min(공개 규칙상 시각, ...)이 아니라 **max(공개 규칙상 시각, 데이터 기준 시각)** 으로 보수적으로 잡는다.
  (라이브 수집에서 규칙보다 일찍 받은 값이라도 규칙 시각 이전에는 사용하지 않는다.)
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import date, datetime, time, timedelta

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from ..db import CollectionLog, now_kst

log = logging.getLogger("collectors")


class DataUnavailable(Exception):
    """데이터를 확보할 수 없음 (키 미설정, API 실패, 응답에 값 없음). 절대 추정값으로 대체하지 않는다."""


def at(d: date, t: time, plus_days: int = 0) -> datetime:
    return datetime.combine(d + timedelta(days=plus_days), t)


def upsert_ignore(session: Session, model, rows: list[dict], keys: list[str]) -> int:
    """이미 있는 (keys) 행은 건드리지 않는다. 원천 데이터의 사후 덮어쓰기를 막기 위함."""
    if not rows:
        return 0
    dialect = session.bind.dialect.name
    ins = sqlite_insert if dialect == "sqlite" else pg_insert
    n = 0
    for i in range(0, len(rows), 500):
        stmt = ins(model).values(rows[i:i + 500]).on_conflict_do_nothing(index_elements=keys)
        n += session.execute(stmt).rowcount or 0
    return n


@contextmanager
def collection(session: Session, collector: str, target: str):
    """with collection(s, 'krx', 'ohlcv 2025-01-02') as c: c.rows = n"""
    class _C:
        rows = 0
    c = _C()
    started = now_kst()
    try:
        yield c
        session.add(CollectionLog(collector=collector, target=target, status="OK", rows=c.rows,
                                  started_at=started, finished_at=now_kst()))
        session.commit()
    except DataUnavailable as e:
        session.rollback()
        session.add(CollectionLog(collector=collector, target=target, status="UNAVAILABLE", message=str(e)[:2000],
                                  started_at=started, finished_at=now_kst()))
        session.commit()
        log.warning("[%s] %s: DATA UNAVAILABLE (%s)", collector, target, e)
    except Exception as e:  # noqa: BLE001 - 어떤 실패든 기록 후 계속
        session.rollback()
        session.add(CollectionLog(collector=collector, target=target, status="ERROR", message=repr(e)[:2000],
                                  started_at=started, finished_at=now_kst()))
        session.commit()
        log.exception("[%s] %s: ERROR", collector, target)


def num(v):
    """pandas/문자열 값을 float 로. 변환 불가·NaN 은 None (0으로 채우지 않는다)."""
    try:
        f = float(str(v).replace(",", "")) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f
