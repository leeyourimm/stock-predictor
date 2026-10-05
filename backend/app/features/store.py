"""Point-in-Time 데이터 저장소.

`PITData.load(session)` 으로 원천 테이블을 한 번 읽고, `.as_of(ts)` 로 **available_at <= ts** 인 값만 남긴 뷰를 만든다.
피처/예측/백테스트 코드는 오직 이 뷰를 통해서만 데이터에 접근한다 → look-ahead 를 구조적으로 차단.
같은 (종목, 일자)에 여러 출처가 있으면 SOURCE_PRIORITY 순으로 하나를 고르고, 선택된 출처를 기록한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db import DailyBar, DataConflict, Disclosure, IndexBar, Instrument, IntradaySnapshot, InvestorFlow, ShortData

SOURCE_PRIORITY = ["KRX(pykrx)", "KIS Open API", "NAVER 금융", "한국은행 ECOS", "FinanceDataReader", "Yahoo Finance(yfinance)",
                   "SYNTHETIC(TEST ONLY)"]


def _prio(src: pd.Series) -> pd.Series:
    order = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    return src.map(lambda s: order.get(s, len(order)))


def _read(session: Session, stmt) -> pd.DataFrame:
    return pd.read_sql(stmt, session.bind)


def _pick_primary(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.assign(_p=_prio(df["source"])).sort_values(keys + ["_p"])
    return df.drop_duplicates(keys, keep="first").drop(columns="_p")


@dataclass
class PITView:
    as_of: datetime
    bars: dict[str, pd.DataFrame]         # field -> (date × ticker)
    bar_source: pd.DataFrame
    flows: dict[str, pd.DataFrame]
    shorts: pd.DataFrame
    index: dict[str, pd.Series]           # symbol -> close series
    index_meta: dict[str, dict]           # symbol -> {source, date, available_at}
    instruments: pd.DataFrame
    disclosures: pd.DataFrame
    snapshot: pd.DataFrame                # ticker -> 당일 최신 장중 스냅샷 (available_at <= as_of)
    snapshots_today: pd.DataFrame         # 당일 모든 스냅샷 (as_of 이전)
    conflicts: set[str] = field(default_factory=set)

    @property
    def last_bar_date(self) -> date | None:
        c = self.bars["close"]
        return c.index[-1] if len(c) else None


class PITData:
    def __init__(self, bars, flows, shorts, index, instruments, disclosures, snapshots, conflicts):
        self.bars_long, self.flows_long, self.shorts_long = bars, flows, shorts
        self.index_long, self.instruments = index, instruments
        self.disclosures, self.snapshots, self.conflicts = disclosures, snapshots, conflicts
        self._bar_wide = self._wide(bars, "ticker", ["open", "high", "low", "close", "volume", "value", "market_cap"])
        self._flow_wide = self._wide(flows, "ticker", ["foreign_net", "institution_net", "individual_net"])
        self._short_wide = self._wide(shorts, "ticker", ["short_volume"])

    @staticmethod
    def _wide(df: pd.DataFrame, col: str, fields: list[str]) -> dict[str, pd.DataFrame]:
        if df.empty:
            return {f: pd.DataFrame() for f in fields + ["available_at", "source"]}
        out = {f: df.pivot(index="date", columns=col, values=f).sort_index() for f in fields}
        out["available_at"] = df.pivot(index="date", columns=col, values="available_at").sort_index()
        out["source"] = df.pivot(index="date", columns=col, values="source").sort_index()
        return out

    @classmethod
    def load(cls, session: Session, since: date | None = None, tickers: list[str] | None = None) -> "PITData":
        def q(model, *cols, date_col="date"):
            stmt = select(*[getattr(model, c) for c in cols])
            if since is not None:
                stmt = stmt.where(getattr(model, date_col) >= since)
            if tickers is not None and hasattr(model, "ticker"):
                stmt = stmt.where(model.ticker.in_(tickers))
            return _read(session, stmt)

        meta = ["source", "available_at"]
        bars = q(DailyBar, "ticker", "date", "open", "high", "low", "close", "volume", "value", "market_cap", *meta)
        flows = q(InvestorFlow, "ticker", "date", "foreign_net", "institution_net", "individual_net", *meta)
        shorts = q(ShortData, "ticker", "date", "short_volume", *meta)
        index = q(IndexBar, "symbol", "date", "close", *meta)
        disc = q(Disclosure, "rcept_no", "ticker", "corp_name", "report_nm", "rcept_dt", *meta, date_col="rcept_dt")
        snaps = q(IntradaySnapshot, "ticker", "ts", "price", "open", "high", "low", "cum_volume", "cum_value", *meta,
                  date_col="ts")
        if not snaps.empty:
            snaps["ts"] = pd.to_datetime(snaps["ts"])
            snaps["day"] = snaps["ts"].dt.date
        conf = _read(session, select(DataConflict.entity, DataConflict.data_date, DataConflict.field))
        inst = _read(session, select(Instrument)).set_index("ticker")
        for df in (bars, flows, shorts, index, disc, snaps):
            if "available_at" in df:
                df["available_at"] = pd.to_datetime(df["available_at"])
        for df in (bars, flows, shorts, index):
            df["date"] = pd.to_datetime(df["date"]).dt.date
        # 다중 출처 → 우선순위 출처 선택 (불일치는 data_conflicts 로 별도 관리)
        bars = _pick_primary(bars, ["ticker", "date"])
        flows = _pick_primary(flows, ["ticker", "date"])
        shorts = _pick_primary(shorts, ["ticker", "date"])
        return cls(bars, flows, shorts, index, inst, disc, snaps, conf)

    def trading_dates(self) -> list[date]:
        k = self.index_long[self.index_long.symbol == "KOSPI"]
        return sorted(set(k["date"]))

    def as_of(self, ts: datetime, lookback: int = 260) -> PITView:
        """ts 시각에 실제로 이용 가능했던 데이터만 포함한 뷰."""
        ts64 = np.datetime64(ts)

        def cut(wide: dict[str, pd.DataFrame], fields: list[str]) -> dict[str, pd.DataFrame]:
            av = wide["available_at"]
            if av.empty:
                return {f: pd.DataFrame() for f in fields}
            mask = av.le(ts64) & av.notna()
            rows = mask.any(axis=1)
            mask = mask[rows].tail(lookback)
            return {f: wide[f].loc[mask.index].where(mask) for f in fields + ["source"]}

        bars = cut(self._bar_wide, ["open", "high", "low", "close", "volume", "value", "market_cap"])
        flows = cut(self._flow_wide, ["foreign_net", "institution_net", "individual_net"])
        shorts = cut(self._short_wide, ["short_volume"])["short_volume"]

        idx = self.index_long[self.index_long.available_at <= ts]
        idx = _pick_primary(idx, ["symbol", "date"])
        index, index_meta = {}, {}
        for sym, g in idx.groupby("symbol"):
            g = g.sort_values("date").tail(lookback)
            index[sym] = pd.Series(g["close"].values, index=g["date"].values, name=sym)
            last = g.iloc[-1]
            index_meta[sym] = {"source": last["source"], "date": last["date"].isoformat(),
                               "available_at": last["available_at"].isoformat()}

        disc = self.disclosures[self.disclosures.available_at <= ts] if not self.disclosures.empty else self.disclosures
        snap = today = self.snapshots
        if not snap.empty:
            if not hasattr(self, "_snap_by_day"):
                self._snap_by_day = {d: g for d, g in self.snapshots.groupby("day")}
            today = self._snap_by_day.get(ts.date(), self.snapshots.iloc[0:0])
            today = today[today.available_at <= ts].sort_values("ts")
            snap = today.drop_duplicates("ticker", keep="last").set_index("ticker")
        conflicts = set()
        if not self.conflicts.empty:
            recent = self.conflicts[pd.to_datetime(self.conflicts.data_date) <= pd.Timestamp(ts)]
            conflicts = set(recent.entity)
        return PITView(as_of=ts, bars={k: v for k, v in bars.items() if k != "source"}, bar_source=bars["source"],
                       flows={k: v for k, v in flows.items() if k != "source"}, shorts=shorts, index=index,
                       index_meta=index_meta, instruments=self.instruments, disclosures=disc, snapshot=snap, snapshots_today=today,
                       conflicts=conflicts)
