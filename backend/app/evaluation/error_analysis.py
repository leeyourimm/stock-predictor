"""틀린 예측의 원인 분석 (규칙 기반, 코드 계산). Claude 는 이 결과를 받아 서술만 덧붙인다.

채점 이후의 사후 분석이므로 T+1 의 실제 시장 데이터(지수·업종 수익률, 고가)를 사용해도 된다.
(예측 자체에는 절대 사용되지 않음)
"""
from __future__ import annotations

from collections import Counter

import pandas as pd

CAUSES = {
    "MARKET_WIDE_MOVE": "시장 전체 움직임을 충분히 반영하지 못함",
    "SECTOR_MOVE": "업종 전체 움직임에 휩쓸림",
    "GAP_DOWN": "다음날 갭하락 (Overnight 위험)",
    "FADE_AFTER_GAP": "갭상승 후 장중 밀림 (시가 매도가 유리했던 경우)",
    "INTRADAY_REVERSAL": "장중 목표가 도달 후 종가 하락",
    "LATE_MOMENTUM_FADE": "장 후반 강세가 다음날 이어지지 않음",
    "INTRADAY_OVERWEIGHTED": "장중 흐름 신호를 과대평가",
    "FLOW_OVERWEIGHTED": "수급(외국인/기관) 신호를 과대평가",
    "TECHNICAL_OVERWEIGHTED": "기술적 신호를 과대평가",
    "MARKET_FACTOR_WRONG": "시장환경 판단이 빗나감",
    "RELATIVE_OVERWEIGHTED": "상대강도 신호를 과대평가",
    "VOLUME_MISREAD": "거래량 급증을 잘못 해석",
    "OVERBOUGHT_IGNORED": "과매수 상태를 과소평가",
    "COST_EATEN": "방향은 맞았으나 거래비용 차감 후 손실",
    "EVENT_DRIVEN": "공시/이벤트 발생",
    "NOISE": "예상 변동 범위 내 잡음 수준의 실패",
    "OUT_OF_RANGE": "예상 변동 범위를 벗어난 큰 움직임",
    "REGIME_MISMATCH": "현재 시장 국면에서 전략 성능 저하",
}
PRIORITY = ["EVENT_DRIVEN", "MARKET_WIDE_MOVE", "SECTOR_MOVE", "COST_EATEN", "GAP_DOWN", "FADE_AFTER_GAP",
            "INTRADAY_REVERSAL", "LATE_MOMENTUM_FADE", "OVERBOUGHT_IGNORED", "VOLUME_MISREAD", "INTRADAY_OVERWEIGHTED",
            "FLOW_OVERWEIGHTED", "TECHNICAL_OVERWEIGHTED", "RELATIVE_OVERWEIGHTED", "MARKET_FACTOR_WRONG",
            "OUT_OF_RANGE", "REGIME_MISMATCH", "NOISE"]
GROUP_TAG = {"intraday": "INTRADAY_OVERWEIGHTED", "flow": "FLOW_OVERWEIGHTED", "technical": "TECHNICAL_OVERWEIGHTED",
             "market": "MARKET_FACTOR_WRONG", "relative": "RELATIVE_OVERWEIGHTED"}


def analyze_failure(pred: dict, outcome: dict, ctx: dict) -> dict:
    """pred: p_up, factor_scores(그룹 기여), features_snapshot.values, range_low/high
    outcome: close_ret, open_ret, high_ret, net_close
    ctx: index_ret_next, sector_ret_next, disclosure_next, regime_hit_rate"""
    from ..config import settings
    up_call = pred["p_up"] >= 0.5
    sign = 1 if up_call else -1
    r, o, h = outcome["close_ret"], outcome.get("open_ret"), outcome.get("high_ret")
    v = (pred.get("features_snapshot") or {}).get("values", {})
    fac = pred.get("factor_scores") or {}
    tags, details = [], {"close_ret": r, "open_ret": o, "high_ret": h, "p_up": pred["p_up"]}
    mret = ctx.get("index_ret_next")
    if mret is not None and sign * mret < -0.01:
        tags.append("MARKET_WIDE_MOVE")
        details["market_ret_next"] = mret
    sret = ctx.get("sector_ret_next")
    if sret is not None and sign * sret < -0.015:
        tags.append("SECTOR_MOVE")
        details["sector_ret_next"] = sret
    if ctx.get("disclosure_next"):
        tags.append("EVENT_DRIVEN")
    if up_call:
        if r > 0 and outcome.get("net_close", r) <= 0:
            tags.append("COST_EATEN")
        if o is not None and o < 0:
            tags.append("GAP_DOWN")
        if o is not None and o > 0 and r < 0:
            tags.append("FADE_AFTER_GAP")
        if h is not None and h >= settings.high_hit_pct and r < 0:
            tags.append("INTRADAY_REVERSAL")
        if (v.get("mom_since_first") or 0) > 0.01 and (o is None or o < 0):
            tags.append("LATE_MOMENTUM_FADE")
        if (v.get("value_pace") or 0) > 2 or (v.get("volume_ratio") or 0) > 2:
            tags.append("VOLUME_MISREAD")
        if (v.get("rsi14") or 0) > 70:
            tags.append("OVERBOUGHT_IGNORED")
    contrib = {g: x.get("logit", x.get("score", 0)) * sign for g, x in fac.items() if g in GROUP_TAG}
    if contrib:
        top = max(contrib, key=contrib.get)
        if contrib[top] > 0:
            tags.append(GROUP_TAG[top])
            details["top_contributor"] = top
    lo, hi = pred.get("range_low"), pred.get("range_high")
    if lo is not None and hi is not None:
        if r < lo or r > hi:
            tags.append("OUT_OF_RANGE")
        elif abs(r) < 0.5 * (hi - lo) / 2.5632:
            tags.append("NOISE")
    if ctx.get("regime_hit_rate") is not None and ctx["regime_hit_rate"] < 0.5:
        tags.append("REGIME_MISMATCH")
        details["regime_hit_rate"] = ctx["regime_hit_rate"]
    tags = list(dict.fromkeys(tags)) or ["NOISE"]
    primary = next(t for t in PRIORITY if t in tags)
    return {"primary_cause": primary, "tags": tags, "details": details, "summary": CAUSES[primary]}


def aggregate(errors: pd.DataFrame, graded: pd.DataFrame) -> dict:
    """원인 태그 빈도, 업종/국면별 실패율."""
    if errors.empty:
        return {"n_failures": 0}
    tag_counts = Counter(t for ts in errors.tags for t in ts)
    out = {"n_failures": int(len(errors)),
           "primary_causes": {k: {"count": int(v), "label": CAUSES.get(k, k)}
                              for k, v in errors.primary_cause.value_counts().items()},
           "tags": {k: {"count": int(v), "label": CAUSES.get(k, k)} for k, v in tag_counts.most_common()}}
    d = graded[graded.result.isin(["SUCCESS", "FAILURE"])]
    if not d.empty:
        for key in ("sector", "regime"):
            if key in d:
                g = d.groupby(d[key].fillna("미분류")).result.agg(
                    n="size", failure_rate=lambda s: float((s == "FAILURE").mean()))
                out[f"failure_rate_by_{key}"] = {k: {"n": int(r.n), "failure_rate": round(r.failure_rate, 4)}
                                                 for k, r in g.iterrows() if r.n >= 10}
    return out
