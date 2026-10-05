"""PITView → 종목별 피처 테이블 + 시장 컨텍스트.

반환되는 모든 값은 view 안의 데이터(= as_of 시점에 공개된 데이터)만으로 계산된다.
없는 값은 NaN/None 으로 두고 절대 채우지 않는다. 대신 unavailable 목록과 data_completeness 에 반영한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .store import PITView
from .technical import atr, bollinger, macd, rsi, sma

# 데이터 그룹별 가중치 (data_completeness 계산용)
GROUP_WEIGHTS = {
    "price_history": 3.0, "investor_flows": 2.0, "market_index": 1.0, "us_markets": 1.0, "fx": 0.5,
    "sector_index": 1.0, "short_selling": 0.5, "intraday": 2.0, "us_futures": 0.5, "disclosures": 0.5,
    "news": 0.5, "consensus": 0.5, "program_trading": 0.5,
}
# Phase 1 에서 소스가 없는 항목 — 값을 만들지 않고 항상 '데이터 없음'으로 명시
ALWAYS_UNAVAILABLE = {"news": "뉴스 수집 Phase 3 예정", "consensus": "무료 공식 컨센서스 소스 없음",
                      "us_futures": "장중 미국 선물 실시간 소스 미연동 (Phase 2)",
                      "program_trading": "종목별 프로그램매매 무료 API 없음 (Phase 2: KIS)"}
EARNINGS_KEYWORDS = ("잠정", "실적", "영업실적", "매출액또는손익구조")


def _r(x, nd=4):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, nd)


@dataclass
class FeatureSet:
    table: pd.DataFrame          # index=ticker
    market: dict                 # 시장 컨텍스트 값
    market_meta: dict            # 출처/기준일
    snapshots: dict              # ticker -> features_snapshot(json)
    last_bar_date: object


def _ret(s: pd.Series, n: int):
    s = s.dropna()
    return s.iloc[-1] / s.iloc[-1 - n] - 1 if len(s) > n else np.nan


def market_context(view: PITView) -> tuple[dict, dict]:
    m, meta = {}, {}
    for sym, key in (("KOSPI", "kospi"), ("KOSDAQ", "kosdaq")):
        s = view.index.get(sym)
        if s is None or len(s) < 2:
            continue
        for n in (1, 5, 20, 60):
            m[f"{key}_ret{n}"] = _ret(s, n)
        if len(s) >= 60:
            m[f"{key}_above_ma60"] = float(s.iloc[-1] > s.tail(60).mean())
        lr = np.log(s).diff().dropna()
        if len(lr) >= 20:
            m[f"{key}_vol20"] = lr.tail(20).std() * np.sqrt(252)
            if len(lr) >= 120:
                roll = lr.rolling(20).std().dropna() * np.sqrt(252)
                m[f"{key}_vol_pctile"] = float((roll <= roll.iloc[-1]).mean())
        meta[sym] = view.index_meta[sym]
    for sym, key in (("US:SPX", "spx"), ("US:NASDAQ", "nasdaq"), ("US:SOX", "sox")):
        s = view.index.get(sym)
        if s is not None and len(s) >= 2:
            m[f"{key}_ret1"] = _ret(s, 1)
            m[f"{key}_ret5"] = _ret(s, 5)
            meta[sym] = view.index_meta[sym]
    if (s := view.index.get("US:VIX")) is not None and len(s):
        m["vix"] = float(s.iloc[-1])
        m["vix_chg5"] = _ret(s, 5)
        meta["US:VIX"] = view.index_meta["US:VIX"]
    for sym in ("FX:USDKRW", "FX:USDKRW:YF"):
        if (s := view.index.get(sym)) is not None and len(s) > 5:
            m["usdkrw"] = float(s.iloc[-1])
            m["usdkrw_chg5"] = _ret(s, 5)
            meta["FX:USDKRW"] = view.index_meta[sym]
            break
    for sym, key in (("KR:KTB3Y", "ktb3y"), ("KR:BASE_RATE", "base_rate"), ("US:10Y", "us10y")):
        if (s := view.index.get(sym)) is not None and len(s):
            m[key] = float(s.iloc[-1])
            meta[sym] = view.index_meta[sym]
    # Risk-on/off 합성 지표: 미국 전일 수익률, VIX 변화, 원화 변화 (부호만 정의, 유효성은 OOS 검증으로 판단)
    parts = [m.get("spx_ret1"), -(m.get("vix_chg5") or 0) / 5 if "vix_chg5" in m else None,
             -(m.get("usdkrw_chg5") or 0) if "usdkrw_chg5" in m else None]
    parts = [p for p in parts if p is not None and not (isinstance(p, float) and math.isnan(p))]
    if parts:
        m["risk_on"] = float(np.mean(parts))
    return {k: _r(v) for k, v in m.items()}, meta


def build_features(view: PITView) -> FeatureSet:
    b = view.bars
    close = b["close"]
    if close.empty:
        return FeatureSet(pd.DataFrame(), *market_context(view), {}, None)
    high, low, vol, value = b["high"], b["low"], b["volume"], b["value"]
    last = close.index[-1]
    ret = close.pct_change(fill_method=None)

    f = pd.DataFrame(index=close.columns)
    f["history_days"] = close.notna().sum()
    f["has_last_bar"] = close.iloc[-1].notna()
    f["close"] = close.ffill().iloc[-1]
    for n in (1, 5, 20, 60, 120):
        f[f"ret{n}"] = close.ffill().iloc[-1] / close.ffill().shift(n).iloc[-1] - 1
    for n in (5, 20, 60, 120):
        f[f"ma{n}"] = sma(close, n).iloc[-1]
    f["dist_ma20"] = f["close"] / f["ma20"] - 1
    f["ma_aligned_up"] = ((f["ma5"] > f["ma20"]) & (f["ma20"] > f["ma60"])).astype(float).where(f["ma60"].notna())
    f["ma_aligned_dn"] = ((f["ma5"] < f["ma20"]) & (f["ma20"] < f["ma60"])).astype(float).where(f["ma60"].notna())
    f["rsi14"] = rsi(close).iloc[-1]
    line, sig, hist = macd(close)
    f["macd"], f["macd_signal"], f["macd_hist"] = line.iloc[-1], sig.iloc[-1], hist.iloc[-1]
    f["macd_hist_chg"] = (hist - hist.shift(1)).iloc[-1]
    pb, bw = bollinger(close)
    f["bb_pctb"], f["bb_width"] = pb.iloc[-1], bw.iloc[-1]
    a = atr(high, low, close)
    f["atr14"] = a.iloc[-1]
    f["atr_pct"] = f["atr14"] / f["close"]
    f["vol20"] = ret.rolling(20, min_periods=15).std().iloc[-1]
    f["vol60"] = ret.rolling(60, min_periods=40).std().iloc[-1]
    f["volume_ratio"] = vol.iloc[-1] / vol.rolling(20, min_periods=15).mean().shift(1).iloc[-1]
    f["value_avg20"] = value.rolling(20, min_periods=10).mean().iloc[-1]
    f["value_last"] = value.iloc[-1]
    f["market_cap"] = b["market_cap"].ffill().iloc[-1] if not b["market_cap"].empty else np.nan
    f["intraday_ret_last"] = (close.iloc[-1] / b["open"].iloc[-1] - 1)
    f["upper_wick_last"] = (high.iloc[-1] - np.maximum(close.iloc[-1], b["open"].iloc[-1])) / f["close"]

    # 수급: 최근 5일/20일 순매수 합 / 20일 평균 거래대금
    fl = view.flows
    if fl.get("foreign_net") is not None and not fl["foreign_net"].empty:
        fn, inn, ind = fl["foreign_net"], fl["institution_net"], fl["individual_net"]
        denom = f["value_avg20"].replace(0, np.nan)
        f["flow_last_date_ok"] = fn.index[-1] == last
        f["foreign_1"] = fn.iloc[-1].reindex(f.index) / denom
        f["foreign_5"] = fn.tail(5).sum(min_count=3).reindex(f.index) / denom
        f["foreign_20"] = fn.tail(20).sum(min_count=10).reindex(f.index) / denom
        f["inst_5"] = inn.tail(5).sum(min_count=3).reindex(f.index) / denom
        f["indiv_5"] = ind.tail(5).sum(min_count=3).reindex(f.index) / denom
        f["foreign_streak"] = (fn.tail(5) > 0).sum().reindex(f.index).where(fn.tail(5).notna().sum() >= 3)
        f["flow_date"] = str(fn.index[-1])
    else:
        for c in ("foreign_1", "foreign_5", "foreign_20", "inst_5", "indiv_5", "foreign_streak"):
            f[c] = np.nan
        f["flow_last_date_ok"] = False
        f["flow_date"] = None

    sv = view.shorts
    if sv is not None and not sv.empty:
        ratio = (sv.reindex(columns=vol.columns) / vol.reindex(sv.index)).replace([np.inf, -np.inf], np.nan)
        f["short_ratio_5"] = ratio.tail(5).mean().reindex(f.index)
        f["short_ratio_60"] = ratio.tail(60).mean().reindex(f.index)
    else:
        f["short_ratio_5"] = f["short_ratio_60"] = np.nan

    # 업종 상대강도
    inst = view.instruments.reindex(f.index)
    f["sector"] = inst["sector"]
    f["sector_index"] = inst["sector_index"]
    f["market"] = inst["market"]
    f["name"] = inst["name"].fillna(pd.Series(f.index, index=f.index))
    f["is_halted"] = inst["is_halted"]
    f["is_admin_issue"] = inst["is_admin_issue"]
    kospi = view.index.get("KOSPI")
    k5 = _ret(kospi, 5) if kospi is not None else np.nan
    sec5 = {}
    for code in f["sector_index"].dropna().unique():
        s = view.index.get(f"SECTOR:{code}")
        sec5[code] = (_ret(s, 5), _ret(s, 20), _ret(s, 1)) if s is not None else (np.nan, np.nan, np.nan)
    f["sector_ret5"] = f["sector_index"].map(lambda c: sec5.get(c, (np.nan,) * 3)[0])
    f["sector_ret20"] = f["sector_index"].map(lambda c: sec5.get(c, (np.nan,) * 3)[1])
    f["sector_rel5"] = f["sector_ret5"] - k5
    f["stock_rel_sector5"] = f["ret5"] - f["sector_ret5"]

    # 연속 상승/하락 일수 (+n / -n)
    sgn = np.sign(ret.tail(10))
    streak = pd.Series(0.0, index=close.columns)
    alive = pd.Series(True, index=close.columns)
    first = sgn.iloc[-1] if len(sgn) else pd.Series(0.0, index=close.columns)
    for k in range(1, len(sgn) + 1):
        same = (sgn.iloc[-k] == first) & alive & (first != 0)
        streak[same] += first[same]
        alive &= same
    f["streak"] = streak
    f["prev_high"] = high.iloc[-1]

    # 장중 스냅샷 (as_of 이전에 공개된 당일 값만): VWAP, 14시 이후 모멘텀, 장후반 거래량, 고가 대비 위치
    snap, today = view.snapshot, view.snapshots_today
    stock_snap = snap[~snap.index.astype(str).str.startswith("IDX:")] if snap is not None and not snap.empty else None
    if stock_snap is not None and not stock_snap.empty:
        sp = stock_snap.reindex(f.index)
        f["snap_price"], f["snap_ts"], f["snap_source"] = sp["price"], sp["ts"].astype(str), sp["source"]
        minutes = sp["ts"].map(lambda x: (x.hour * 60 + x.minute - 540) if pd.notna(x) else np.nan).clip(1, 390)
        f["intraday_ret"] = sp["price"] / f["close"] - 1                          # 전일 종가 대비
        f["ret_since_open"] = sp["price"] / sp["open"] - 1
        f["gap_today"] = sp["open"] / f["close"] - 1
        vwap = sp["cum_value"] / sp["cum_volume"]
        f["vwap_gap"] = sp["price"] / vwap - 1
        f["dist_day_high"] = sp["price"] / sp["high"] - 1
        f["day_range_pos"] = (sp["price"] - sp["low"]) / (sp["high"] - sp["low"]).replace(0, np.nan)
        f["broke_prev_high"] = (sp["high"] > f["prev_high"]).astype(float).where(sp["high"].notna())
        f["value_pace"] = (sp["cum_value"] / f["value_avg20"]) / (minutes / 390)   # 시간 보정 거래대금 페이스
        first = today[~today.ticker.str.startswith("IDX:")].drop_duplicates("ticker", keep="first").set_index("ticker")
        fp = first.reindex(f.index)
        later = sp["ts"] > fp["ts"]
        f["mom_since_first"] = (sp["price"] / fp["price"] - 1).where(later)        # 첫 스냅샷(≈14:00) 이후 모멘텀
        fm = fp["ts"].map(lambda x: (x.hour * 60 + x.minute - 540) if pd.notna(x) else np.nan)
        rate_late = (sp["cum_value"] - fp["cum_value"]) / (minutes - fm)
        rate_early = fp["cum_value"] / fm
        f["late_volume_accel"] = (rate_late / rate_early).where(later)              # 장 후반 거래 강도 변화
    else:
        for c in ("snap_price", "intraday_ret", "ret_since_open", "gap_today", "vwap_gap", "dist_day_high",
                  "day_range_pos", "broke_prev_high", "value_pace", "mom_since_first", "late_volume_accel"):
            f[c] = np.nan
        f["snap_ts"] = f["snap_source"] = None
    idx_snap = snap.loc["IDX:KOSPI"] if snap is not None and not snap.empty and "IDX:KOSPI" in snap.index else None
    kospi_intraday = np.nan
    if idx_snap is not None and kospi is not None and len(kospi):
        kospi_intraday = idx_snap["price"] / kospi.iloc[-1] - 1
    f["rel_market_intraday"] = f["intraday_ret"] - kospi_intraday

    # 공시 (as_of 까지 공개된 것, 최근 3일)
    disc = view.disclosures
    f["disclosures_3d"] = 0.0
    f["earnings_disclosure_3d"] = 0.0
    disclosures_available = disc is not None and not disc.empty
    if disclosures_available:
        recent = disc[pd.to_datetime(disc.rcept_dt) >= pd.Timestamp(view.as_of.date()) - pd.Timedelta(days=4)]
        cnt = recent.groupby("ticker").size()
        earn = recent[recent.report_nm.str.contains("|".join(EARNINGS_KEYWORDS))].groupby("ticker").size()
        f["disclosures_3d"] = cnt.reindex(f.index).fillna(0)
        f["earnings_disclosure_3d"] = earn.reindex(f.index).fillna(0)
    else:
        f["disclosures_3d"] = f["earnings_disclosure_3d"] = np.nan

    f["conflict"] = f.index.isin(list(view.conflicts))

    market, market_meta = market_context(view)
    if not (isinstance(kospi_intraday, float) and math.isnan(kospi_intraday)):
        market["kospi_intraday"] = _r(kospi_intraday)
        market_meta["IDX:KOSPI(intraday)"] = {"source": str(idx_snap["source"]), "ts": str(idx_snap["ts"])}

    # 데이터 완전성 & 스냅샷
    snapshots = {}
    for t, row in f.iterrows():
        avail = {
            "price_history": row["history_days"] >= 60 and bool(row["has_last_bar"]),
            "investor_flows": not pd.isna(row["foreign_5"]),
            "market_index": "kospi_ret1" in market,
            "us_markets": "spx_ret1" in market,
            "fx": "usdkrw_chg5" in market,
            "sector_index": not pd.isna(row["sector_ret5"]),
            "short_selling": not pd.isna(row["short_ratio_5"]),
            "intraday": not pd.isna(row["snap_price"]),
            "disclosures": disclosures_available,
            "news": False, "consensus": False, "program_trading": False, "us_futures": False,
        }
        comp = sum(GROUP_WEIGHTS[g] for g, ok in avail.items() if ok) / sum(GROUP_WEIGHTS.values())
        f.loc[t, "data_completeness"] = comp
        unavailable = {g: ALWAYS_UNAVAILABLE.get(g, "데이터 없음") for g, ok in avail.items() if not ok}
        price_src = view.bar_source[t].dropna().iloc[-1] if t in view.bar_source and view.bar_source[t].notna().any() else None
        snapshots[t] = {
            "as_of": view.as_of.isoformat(),
            "values": {**{k: _r(row[k]) for k in FEATURE_COLUMNS if k in row}, "data_completeness": _r(comp)},
            "sources": {
                "price": {"source": price_src, "last_bar_date": str(last)},
                "investor_flows": {"date": row["flow_date"]},
                "intraday": {"source": row["snap_source"], "ts": row["snap_ts"]} if avail["intraday"] else None,
                "market": "prediction_runs.data_sources 참조",
            },
            "unavailable": unavailable,
        }
    return FeatureSet(f, market, market_meta, snapshots, last)


FEATURE_COLUMNS = [
    "close", "history_days", "ret1", "ret5", "ret20", "ret60", "ret120", "ma5", "ma20", "ma60", "ma120", "dist_ma20",
    "ma_aligned_up", "ma_aligned_dn", "rsi14", "macd", "macd_signal", "macd_hist", "macd_hist_chg", "bb_pctb",
    "bb_width", "atr14", "atr_pct", "vol20", "vol60", "volume_ratio", "value_avg20", "value_last", "market_cap",
    "intraday_ret_last", "upper_wick_last", "foreign_1", "foreign_5", "foreign_20", "inst_5", "indiv_5",
    "foreign_streak", "short_ratio_5", "short_ratio_60", "sector_ret5", "sector_ret20", "sector_rel5",
    "stock_rel_sector5", "streak", "snap_price", "intraday_ret", "ret_since_open", "gap_today", "vwap_gap",
    "dist_day_high", "day_range_pos", "broke_prev_high", "value_pace", "mom_since_first", "late_volume_accel",
    "rel_market_intraday", "disclosures_3d", "earnings_disclosure_3d", "data_completeness",
]
