"""예측 엔진: 피처 → 학습 모델(상승/갭/고가도달/기대수익) → 범위 → 위험 플래그 → Confidence → 순위/Overnight 후보.

후보 조건 (모두 만족):
  1) 전략 우위 게이트: 이 전략의 Out-of-Sample 그림자 포트폴리오(일별 상위 K 매수)가 비용 차감 후
     통계적으로 양(+)의 수익(t ≥ EDGE_MIN_TSTAT, 표본 ≥ EDGE_MIN_DAYS)을 보였을 것
  2) 종목: 후보 자격(정지/관리/유동성/고변동/데이터부족 아님), 상승확률 ≥ CANDIDATE_MIN_P,
     비용 차감 기대수익 ≥ MIN_EDGE_NET, Confidence ≥ Medium
  하나도 없으면 "오늘은 매수 후보 없음".
"""
from __future__ import annotations

import math

import pandas as pd

from ..config import settings
from ..features.builder import FeatureSet
from .model import Model, feature_frame

WARNING_FLAGS = {"HIGH_VOLATILITY", "RECENT_SURGE", "EARNINGS_EVENT", "DATA_CONFLICT", "MARKET_RISK", "NEWS_EVENT",
                 "NO_INTRADAY"}
EXCLUDE_FLAGS = {"HALTED", "ADMIN_ISSUE", "INSUFFICIENT_HISTORY", "LOW_LIQUIDITY", "HIGH_VOLATILITY",
                 "INCOMPLETE_DATA", "DATA_CONFLICT", "NO_CURRENT_BAR"}
FLAG_LABELS = {
    "HALTED": "거래정지", "ADMIN_ISSUE": "관리종목", "STATUS_UNKNOWN": "거래정지/관리 여부 확인 불가",
    "INSUFFICIENT_HISTORY": "가격 이력 부족", "LOW_LIQUIDITY": "거래량(대금) 부족", "HIGH_VOLATILITY": "변동성 과다",
    "RECENT_SURGE": "최근 급등", "EARNINGS_EVENT": "실적 발표 이벤트", "NEWS_EVENT": "공시/뉴스 이벤트",
    "MARKET_RISK": "시장 위험(고변동/하락장/VIX)", "DATA_CONFLICT": "출처 간 데이터 불일치",
    "INCOMPLETE_DATA": "데이터 부족", "NO_CURRENT_BAR": "최근 거래일 시세 없음",
    "NO_INTRADAY": "장중 시세 없음(전일 데이터만 사용)",
}
NO_CANDIDATE_MSG = "오늘은 매수 후보 없음"


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _r(x, nd=5):
    return None if _nan(x) else round(float(x), nd)


def risk_flags(f: dict, regime: dict, market: dict) -> list[str]:
    flags = []
    if f.get("is_halted") is True:
        flags.append("HALTED")
    if f.get("is_admin_issue") is True:
        flags.append("ADMIN_ISSUE")
    if f.get("is_halted") is None and f.get("is_admin_issue") is None:
        flags.append("STATUS_UNKNOWN")
    if not f.get("has_last_bar"):
        flags.append("NO_CURRENT_BAR")
    if (f.get("history_days") or 0) < settings.min_history_days:
        flags.append("INSUFFICIENT_HISTORY")
    if _nan(f.get("value_avg20")) or f["value_avg20"] < settings.min_avg_value_krw:
        flags.append("LOW_LIQUIDITY")
    if not _nan(f.get("atr_pct")) and f["atr_pct"] > settings.max_atr_pct_candidate:
        flags.append("HIGH_VOLATILITY")
    if (not _nan(f.get("ret5")) and f["ret5"] > 0.2) or (not _nan(f.get("intraday_ret")) and f["intraday_ret"] > 0.15):
        flags.append("RECENT_SURGE")
    if not _nan(f.get("earnings_disclosure_3d")) and f["earnings_disclosure_3d"] > 0:
        flags.append("EARNINGS_EVENT")
    elif not _nan(f.get("disclosures_3d")) and f["disclosures_3d"] > 0:
        flags.append("NEWS_EVENT")
    if regime.get("trend") == "BEAR" or regime.get("vol") == "HIGH_VOL" or (market.get("vix") or 0) > 30:
        flags.append("MARKET_RISK")
    if f.get("conflict"):
        flags.append("DATA_CONFLICT")
    if _nan(f.get("snap_price")):
        flags.append("NO_INTRADAY")
    if (f.get("data_completeness") or 0) < 0.55:
        flags.append("INCOMPLETE_DATA")
    return flags


def edge_gate(edge: dict | None) -> tuple[bool, str]:
    if not edge or not edge.get("days"):
        return False, "전략의 Out-of-Sample 성과 기록 없음"
    if edge["days"] < settings.edge_min_days:
        return False, f"OOS 표본 부족 ({edge['days']}일 < {settings.edge_min_days}일)"
    t = edge.get("tstat")
    if t is None or edge.get("mean_net", 0) <= 0 or t < settings.edge_min_tstat:
        return False, (f"비용 차감 OOS 우위 미확인 (일평균 {edge.get('mean_net', 0) * 100:+.3f}%, "
                       f"t={'-' if t is None else round(t, 2)} < {settings.edge_min_tstat})")
    return True, f"OOS {edge['days']}일 일평균 {edge['mean_net'] * 100:+.3f}% (t={t:.2f})"


def confidence(r: dict, flags: list[str], gate_ok: bool, gate_reason: str, trained: bool) -> tuple[str, str]:
    if not trained:
        return "Low", "학습된 모델 없음 (표본 부족)"
    if not gate_ok:
        return "Low", gate_reason
    comp = r["completeness"]
    if (r["p_up"] >= settings.candidate_min_p + 0.03 and r["exp_net"] >= 2 * settings.min_edge_net and comp >= 0.75
            and not (set(flags) & WARNING_FLAGS)):
        return "High", f"{gate_reason}; 확률·기대수익 여유, 데이터 완전성 {comp:.2f}"
    if r["p_up"] >= settings.candidate_min_p and r["exp_net"] >= settings.min_edge_net and comp >= 0.6:
        return "Medium", f"{gate_reason}; 기준 충족"
    return "Low", f"{gate_reason}; 이 종목은 확률/기대수익 기준 미달"


def predict(fs: FeatureSet, regime: dict, model: Model, edge: dict | None, universe: list[str] | None = None,
            final: bool = False, explain: bool = True) -> pd.DataFrame:
    if fs.table.empty:
        return pd.DataFrame()
    table = fs.table if universe is None else fs.table.loc[[t for t in universe if t in fs.table.index]]
    if table.empty:
        return pd.DataFrame()
    X = feature_frame(table, fs.market)
    P = model.predict(X)
    gate_ok, gate_reason = edge_gate(edge)
    out = []
    for t, row in table.iterrows():
        f = row.to_dict()
        p = P.loc[t]
        flags = risk_flags(f, regime, fs.market)
        rec = {"p_up": float(p.p_up), "exp_net": float(p.exp_net), "completeness": float(f.get("data_completeness") or 0)}
        conf, why = confidence(rec, flags, gate_ok, gate_reason, model.trained)
        groups = model.explain(p.contrib, X.loc[t].to_dict()) if explain else {}
        price = f.get("snap_price") if not _nan(f.get("snap_price")) else f.get("close")
        out.append(dict(
            ticker=t, name=f.get("name") or t, sector=f.get("sector") if isinstance(f.get("sector"), str) else None,
            price=None if _nan(price) else float(price), price_is_intraday=not _nan(f.get("snap_price")),
            price_ts=f.get("snap_ts"), p_up=round(float(p.p_up), 4), p_down=round(1 - float(p.p_up), 4),
            p_gap_up=round(float(p.p_gap), 4), p_high_hit=round(float(p.p_hit), 4),
            exp_ret=_r(p.exp_close), exp_net=_r(p.exp_net), range_low=_r(p.range_low), range_high=_r(p.range_high),
            up_med=_r(p.up_med), up_p80=_r(p.up_p80), down_med=_r(p.down_med), down_p20=_r(p.down_p20),
            confidence=conf, confidence_reason=why, factors=groups,
            final_score=float(sum(g["score"] for g in groups.values())) if groups else 0.0,
            risk_flags=flags, eligible=not (set(flags) & EXCLUDE_FLAGS), completeness=round(rec["completeness"], 3)))
    res = pd.DataFrame(out).set_index("ticker")
    ranked = res[res.eligible].sort_values(["exp_net", "p_up"], ascending=False)
    res["rank"] = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    ok = (res.eligible & (res.p_up >= settings.candidate_min_p) & (res.exp_net >= settings.min_edge_net)
          & res.confidence.isin(["High", "Medium"]))
    res["is_candidate"] = False
    if final:
        top = res[ok].sort_values("rank").head(settings.max_final_picks).index
        res.loc[top, "is_candidate"] = True
    else:
        res["is_candidate"] = ok
    res.attrs["gate"] = {"ok": gate_ok, "reason": gate_reason}
    res.attrs["features"] = X
    return res


def key_reasons(factors: dict, positive: bool = True, k: int = 3) -> list[str]:
    items = [(i["points"], i["label"]) for g in factors.values() for i in g["items"] if i.get("points")]
    items = [x for x in items if (x[0] > 0) == positive]
    items.sort(key=lambda x: -abs(x[0]))
    return [lbl for _, lbl in items[:k]]
