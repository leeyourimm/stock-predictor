"""데이터베이스 모델.

시각 규칙: 모든 datetime 컬럼은 KST 기준 naive datetime 으로 저장한다 (`now_kst()` 사용).
불변성: predictions / prediction_runs / prediction_outcomes 는 DB 트리거로 UPDATE·DELETE 를 금지하고,
predictions 는 해시 체인(prev_hash → row_hash)으로 사후 변조를 탐지할 수 있다.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path

from sqlalchemy import (JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text,
                        UniqueConstraint, create_engine, event, select, text)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import KST, settings


def now_kst() -> datetime:
    return datetime.now(KST).replace(tzinfo=None, microsecond=0)


class Base(DeclarativeBase):
    pass


class SourceMixin:
    """모든 외부 데이터 행이 갖는 출처/시각 메타데이터."""
    source: Mapped[str] = mapped_column(String(64))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
    data_timestamp: Mapped[datetime] = mapped_column(DateTime)
    available_at: Mapped[datetime] = mapped_column(DateTime, index=True)  # 이 시각 이후에만 분석에 사용 가능


# ---------------------------------------------------------------- 원천 데이터
class Instrument(Base):
    __tablename__ = "instruments"
    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    market: Mapped[str] = mapped_column(String(16))
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sector_index: Mapped[str | None] = mapped_column(String(16), nullable=True)  # KRX 업종지수 코드
    is_halted: Mapped[bool | None] = mapped_column(Boolean, nullable=True)       # None = 확인 불가
    is_admin_issue: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # 관리종목
    status_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status_checked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class DailyBar(SourceMixin, Base):
    __tablename__ = "daily_bars"
    __table_args__ = (UniqueConstraint("ticker", "date", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    value: Mapped[float | None] = mapped_column(Float)        # 거래대금(원)
    change_pct: Mapped[float | None] = mapped_column(Float)
    market_cap: Mapped[float | None] = mapped_column(Float, nullable=True)


class InvestorFlow(SourceMixin, Base):
    __tablename__ = "investor_flows"
    __table_args__ = (UniqueConstraint("ticker", "date", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    foreign_net: Mapped[float | None] = mapped_column(Float)       # 순매수 거래대금(원)
    institution_net: Mapped[float | None] = mapped_column(Float)
    individual_net: Mapped[float | None] = mapped_column(Float)


class ShortData(SourceMixin, Base):
    __tablename__ = "short_data"
    __table_args__ = (UniqueConstraint("ticker", "date", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    short_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    short_balance: Mapped[float | None] = mapped_column(Float, nullable=True)


class IndexBar(SourceMixin, Base):
    """KOSPI/KOSDAQ/업종지수/해외지수/환율/금리 등 시계열."""
    __tablename__ = "index_bars"
    __table_args__ = (UniqueConstraint("symbol", "date", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)
    high: Mapped[float | None] = mapped_column(Float, nullable=True)
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float, nullable=True)


class Disclosure(SourceMixin, Base):
    __tablename__ = "disclosures"
    rcept_no: Mapped[str] = mapped_column(String(32), primary_key=True)
    ticker: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    corp_name: Mapped[str] = mapped_column(String(128))
    report_nm: Mapped[str] = mapped_column(String(512))
    rcept_dt: Mapped[date] = mapped_column(Date, index=True)


class IntradaySnapshot(SourceMixin, Base):
    __tablename__ = "intraday_snapshots"
    __table_args__ = (UniqueConstraint("ticker", "ts", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, index=True)
    price: Mapped[float | None] = mapped_column(Float)
    open: Mapped[float | None] = mapped_column(Float, nullable=True)   # 당일 시가
    high: Mapped[float | None] = mapped_column(Float, nullable=True)   # 스냅샷 시각까지의 당일 고가
    low: Mapped[float | None] = mapped_column(Float, nullable=True)
    cum_volume: Mapped[float | None] = mapped_column(Float, nullable=True)
    cum_value: Mapped[float | None] = mapped_column(Float, nullable=True)


class CollectionLog(Base):
    """수집 시도 기록. 실패/미설정은 값을 만들지 않고 여기 UNAVAILABLE 로 남긴다."""
    __tablename__ = "collection_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    collector: Mapped[str] = mapped_column(String(64))
    target: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(16))   # OK / UNAVAILABLE / ERROR
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    rows: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime)
    finished_at: Mapped[datetime] = mapped_column(DateTime)


class DataConflict(Base):
    __tablename__ = "data_conflicts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    entity: Mapped[str] = mapped_column(String(32), index=True)
    field: Mapped[str] = mapped_column(String(32))
    data_date: Mapped[date] = mapped_column(Date, index=True)
    source_a: Mapped[str] = mapped_column(String(64))
    value_a: Mapped[float] = mapped_column(Float)
    source_b: Mapped[str] = mapped_column(String(64))
    value_b: Mapped[float] = mapped_column(Float)
    rel_diff: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime)


# ---------------------------------------------------------------- 전략 / 예측
class StrategyVersion(Base):
    __tablename__ = "strategy_versions"
    version: Mapped[str] = mapped_column(String(32), primary_key=True)
    parent_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    weights: Mapped[dict] = mapped_column(JSON)
    calibration: Mapped[dict] = mapped_column(JSON)        # {"a":..,"b":..,"n":..,"base_rate":..,"oos_brier":..}
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16))        # champion / candidate / rejected / retired
    reason: Mapped[str] = mapped_column(Text)
    evaluation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class PredictionRun(Base):
    """하루에 단계(S1400~S1520)별로 1개씩. 각 단계 실행은 불변."""
    __tablename__ = "prediction_runs"
    __table_args__ = (UniqueConstraint("trade_date", "stage"),)
    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    stage: Mapped[str] = mapped_column(String(8))
    parent_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    as_of: Mapped[datetime] = mapped_column(DateTime)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    target_date: Mapped[date] = mapped_column(Date)
    strategy_version: Mapped[str] = mapped_column(String(32))
    regime: Mapped[str] = mapped_column(String(32))
    n_universe: Mapped[int] = mapped_column(Integer)
    n_scored: Mapped[int] = mapped_column(Integer)
    no_candidate: Mapped[bool] = mapped_column(Boolean)
    summary: Mapped[dict] = mapped_column(JSON)
    data_sources: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)


class Prediction(Base):
    __tablename__ = "predictions"
    __table_args__ = (UniqueConstraint("run_id", "ticker"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("prediction_runs.run_id"), index=True)
    stage: Mapped[str] = mapped_column(String(8), index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    name: Mapped[str] = mapped_column(String(128))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    target_date: Mapped[date] = mapped_column(Date, index=True)
    price_at_prediction: Mapped[float | None] = mapped_column(Float)
    price_source: Mapped[str] = mapped_column(String(64))
    price_timestamp: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    p_up: Mapped[float] = mapped_column(Float)
    p_down: Mapped[float] = mapped_column(Float)
    range_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    range_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_return: Mapped[float | None] = mapped_column(Float, nullable=True)      # 다음날 종가 기대수익률(총)
    expected_net_return: Mapped[float | None] = mapped_column(Float, nullable=True)  # 비용 차감
    p_gap_up: Mapped[float | None] = mapped_column(Float, nullable=True)             # 다음날 시가 > 진입가
    p_high_hit: Mapped[float | None] = mapped_column(Float, nullable=True)           # 다음날 고가 ≥ 진입가×(1+HIGH_HIT_PCT)
    up_med: Mapped[float | None] = mapped_column(Float, nullable=True)               # 예상 상승 범위: 장중 최대수익 중앙값
    up_p80: Mapped[float | None] = mapped_column(Float, nullable=True)               #   ~ 80분위
    down_med: Mapped[float | None] = mapped_column(Float, nullable=True)             # 예상 하락 범위: 장중 최대손실 중앙값
    down_p20: Mapped[float | None] = mapped_column(Float, nullable=True)             #   ~ 20분위
    is_candidate: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[str] = mapped_column(String(8))
    factor_scores: Mapped[dict] = mapped_column(JSON)
    final_score: Mapped[float] = mapped_column(Float)
    risk_flags: Mapped[list] = mapped_column(JSON)
    eligible: Mapped[bool] = mapped_column(Boolean)       # 후보 자격 (정지/관리/데이터부족/고변동 아님)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    data_completeness: Mapped[float] = mapped_column(Float)
    features_snapshot: Mapped[dict] = mapped_column(JSON)  # 예측 당시 사용한 데이터(값+출처+시각)
    regime: Mapped[str] = mapped_column(String(32))
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    strategy_version: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)  # Claude 설명(생성 시점에만 기록)
    prev_hash: Mapped[str] = mapped_column(String(64))
    row_hash: Mapped[str] = mapped_column(String(64), unique=True)


class PredictionOutcome(Base):
    """Overnight 실제 결과. 진입가 = 예측 시점 가격(15:20 등 장중 스냅샷), 없으면 당일 종가(종가 단일가 체결)."""
    __tablename__ = "prediction_outcomes"
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), primary_key=True)
    entry_price: Mapped[float] = mapped_column(Float)
    entry_basis: Mapped[str] = mapped_column(String(32))      # SNAPSHOT@HH:MM / CLOSE
    ref_close: Mapped[float] = mapped_column(Float)          # close(T)
    next_open: Mapped[float | None] = mapped_column(Float, nullable=True)
    next_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    next_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_close: Mapped[float] = mapped_column(Float)       # close(T+1)
    open_return: Mapped[float | None] = mapped_column(Float, nullable=True)   # Overnight 시가 수익률
    high_return: Mapped[float | None] = mapped_column(Float, nullable=True)   # 다음날 장중 최대수익률
    low_return: Mapped[float | None] = mapped_column(Float, nullable=True)    # 다음날 장중 최대손실률
    actual_return: Mapped[float] = mapped_column(Float)                       # 다음날 종가 수익률
    net_open_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    net_close_return: Mapped[float] = mapped_column(Float)
    gap_up: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    high_hit: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    in_range: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    result: Mapped[str] = mapped_column(String(8))           # 방향: SUCCESS / FAILURE / NEUTRAL
    trade_result: Mapped[str] = mapped_column(String(8))     # 비용 차감 종가 매도 기준: WIN / LOSS
    brier: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(64))
    graded_at: Mapped[datetime] = mapped_column(DateTime)


class ErrorLog(Base):
    __tablename__ = "error_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True)
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    ticker: Mapped[str] = mapped_column(String(16), index=True)
    primary_cause: Mapped[str] = mapped_column(String(64))
    tags: Mapped[list] = mapped_column(JSON)
    details: Mapped[dict] = mapped_column(JSON)
    llm_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class TrainingSample(Base):
    """모델 학습/특징 검증용 표본: (예측 시점 특징값, 이후 실제 Overnight 결과). source = BACKTEST / LIVE."""
    __tablename__ = "training_samples"
    __table_args__ = (UniqueConstraint("trade_date", "stage", "ticker", "source"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))
    stage: Mapped[str] = mapped_column(String(8))
    trade_date: Mapped[date] = mapped_column(Date, index=True)
    target_date: Mapped[date] = mapped_column(Date)
    ticker: Mapped[str] = mapped_column(String(16))
    regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(128), nullable=True)
    features: Mapped[dict] = mapped_column(JSON)
    entry_basis: Mapped[str] = mapped_column(String(32))
    open_ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    low_ret: Mapped[float | None] = mapped_column(Float, nullable=True)
    close_ret: Mapped[float] = mapped_column(Float)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class PerformanceSnapshot(Base):
    __tablename__ = "performance_snapshots"
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    metrics: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    params: Mapped[dict] = mapped_column(JSON)
    strategy_version: Mapped[str] = mapped_column(String(32))
    metrics: Mapped[dict] = mapped_column(JSON)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)


# ---------------------------------------------------------------- 불변성
IMMUTABLE_TABLES = ("predictions", "prediction_runs", "prediction_outcomes")

_SQLITE_TRIGGERS = [
    f"CREATE TRIGGER IF NOT EXISTS {t}_no_{op.lower()} BEFORE {op} ON {t} "
    f"BEGIN SELECT RAISE(ABORT, '{t} is append-only: {op} forbidden'); END;"
    for t in IMMUTABLE_TABLES for op in ("UPDATE", "DELETE")
]
_PG_TRIGGERS = [
    """CREATE OR REPLACE FUNCTION forbid_mutation() RETURNS trigger AS $$
       BEGIN RAISE EXCEPTION '% is append-only', TG_TABLE_NAME; END; $$ LANGUAGE plpgsql;""",
] + [
    f"DROP TRIGGER IF EXISTS {t}_immutable ON {t}; CREATE TRIGGER {t}_immutable BEFORE UPDATE OR DELETE ON {t} "
    f"FOR EACH ROW EXECUTE FUNCTION forbid_mutation();"
    for t in IMMUTABLE_TABLES
]


def canonical(obj) -> str:
    def default(o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()
        raise TypeError(type(o))
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=default, separators=(",", ":"))


HASHED_FIELDS = ("run_id", "stage", "ticker", "trade_date", "target_date", "price_at_prediction", "p_up", "p_down",
                 "range_low", "range_high", "expected_return", "expected_net_return", "p_gap_up", "p_high_hit",
                 "up_med", "up_p80", "down_med", "down_p20", "is_candidate", "confidence", "factor_scores", "final_score", "risk_flags",
                 "features_snapshot", "strategy_version", "created_at", "explanation")


def _norm(o):
    """해시 안정화: -0.0 → 0.0 (SQLite 가 -0.0 을 0.0 으로 저장), float 하위 타입 → float."""
    if isinstance(o, dict):
        return {k: _norm(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_norm(v) for v in o]
    if isinstance(o, float):
        return None if o != o else float(o) + 0.0
    return o


def prediction_hash(prev_hash: str, fields: dict) -> str:
    payload = _norm({k: fields.get(k) for k in HASHED_FIELDS})
    return hashlib.sha256((prev_hash + canonical(payload)).encode()).hexdigest()


GENESIS_HASH = "0" * 64


def last_prediction_hash(session: Session) -> str:
    h = session.execute(select(Prediction.row_hash).order_by(Prediction.id.desc()).limit(1)).scalar()
    return h or GENESIS_HASH


def verify_chain(session: Session) -> dict:
    prev = GENESIS_HASH
    n = 0
    for p in session.execute(select(Prediction).order_by(Prediction.id)).scalars():
        fields = {k: getattr(p, k) for k in HASHED_FIELDS}
        if p.prev_hash != prev or prediction_hash(prev, fields) != p.row_hash:
            return {"ok": False, "checked": n, "broken_at_id": p.id}
        prev = p.row_hash
        n += 1
    return {"ok": True, "checked": n, "head": prev}


# ---------------------------------------------------------------- 엔진
def make_engine(url: str | None = None):
    url = url or settings.database_url
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, future=True)
    if eng.dialect.name == "sqlite":
        @event.listens_for(eng, "connect")
        def _fk(dbapi_conn, _):
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    return eng


def init_db(eng) -> None:
    Base.metadata.create_all(eng)
    with eng.begin() as conn:
        stmts = _SQLITE_TRIGGERS if eng.dialect.name == "sqlite" else _PG_TRIGGERS
        for s in stmts:
            conn.execute(text(s))


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def get_session() -> Session:
    return SessionLocal()
