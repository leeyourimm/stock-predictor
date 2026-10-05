"""Claude 설명 생성 (선택). 숫자는 이미 코드가 계산한 값만 전달하고, Claude 의 출력은 설명 텍스트로만 쓰인다.

ANTHROPIC_API_KEY 가 없으면 코드 템플릿 설명(template_explanation)을 사용한다.
"""
from __future__ import annotations

import json
import logging

import requests

from ..config import settings
from ..prediction.engine import FLAG_LABELS, key_reasons

log = logging.getLogger("claude")
API_URL = "https://api.anthropic.com/v1/messages"

SYSTEM = (
    "당신은 한국 주식 예측 시스템의 설명 담당이다. 입력 JSON 의 숫자는 코드가 계산한 확정값이다. "
    "규칙: (1) 입력에 없는 숫자·뉴스·공시·사실을 절대 만들지 마라. (2) '데이터 없음' 항목은 그대로 데이터 없음이라고 말하라. "
    "(3) 확률을 수익 보장처럼 표현하지 마라. 이 시스템은 장 마감 전 매수 → 다음 거래일 매도(Overnight) 후보를 평가한다. (4) 한국어 3~5문장, 핵심 근거와 주요 위험을 균형 있게."
)


def template_explanation(p: dict) -> str:
    pos = key_reasons(p["factors"], True) or ["뚜렷한 상승 요인 없음"]
    neg = key_reasons(p["factors"], False) or ["뚜렷한 하락 요인 없음"]
    flags = [FLAG_LABELS.get(f, f) for f in p["risk_flags"]]

    def pc(x):
        return "데이터 없음" if x is None or x != x else f"{x * 100:+.1f}%"
    return (f"다음 거래일 상승확률 {p['p_up'] * 100:.0f}%, 갭상승 {p['p_gap_up'] * 100:.0f}%, "
            f"장중 고가 +{settings.high_hit_pct * 100:.0f}% 도달 {p['p_high_hit'] * 100:.0f}%. "
            f"비용 차감 기대수익 {pc(p.get('exp_net'))}, 예상 상승폭(장중 최대) {pc(p.get('up_med'))}~{pc(p.get('up_p80'))}, "
            f"예상 하락폭(장중 최대) {pc(p.get('down_med'))}~{pc(p.get('down_p20'))}. "
            f"Confidence {p['confidence']} ({p.get('confidence_reason', '')}). "
            f"상승 근거: {', '.join(pos)}. 하락 위험: {', '.join(neg)}."
            + (f" 위험 플래그: {', '.join(flags)}." if flags else "")
            + " 확률·범위는 과거 실제 결과로 학습한 통계치이며 수익을 보장하지 않습니다.")


def _call(prompt: str, max_tokens: int = 600) -> str | None:
    if not settings.anthropic_api_key:
        return None
    try:
        r = requests.post(API_URL, timeout=60, headers={
            "x-api-key": settings.anthropic_api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": settings.claude_model, "max_tokens": max_tokens, "system": SYSTEM,
                  "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []) if b.get("type") == "text").strip()
    except Exception as e:  # noqa: BLE001
        log.warning("Claude 호출 실패: %s", e)
        return None


def explain_prediction(p: dict, snapshot: dict) -> str:
    payload = {k: p.get(k) for k in ("ticker", "name", "p_up", "p_down", "p_gap_up", "p_high_hit", "exp_ret", "exp_net",
                                     "range_low", "range_high", "up_med", "up_p80", "down_med", "down_p20",
                                     "confidence", "confidence_reason", "final_score", "risk_flags")}
    payload["factor_contributions"] = {g: {"score": v["score"], "items": v["items"]} for g, v in p["factors"].items()}
    payload["unavailable_data"] = snapshot.get("unavailable")
    payload["key_features"] = {k: v for k, v in (snapshot.get("values") or {}).items() if v is not None}
    text = _call("다음 계산 결과를 사용자에게 설명하라:\n" + json.dumps(payload, ensure_ascii=False, default=str))
    return f"[Claude] {text}" if text else f"[자동 템플릿] {template_explanation(p)}"


def explain_failure(pred: dict, analysis: dict) -> str | None:
    payload = {"prediction": {k: pred.get(k) for k in ("ticker", "p_up", "range_low", "range_high", "factor_scores")},
               "analysis": analysis}
    return _call("다음 예측 실패에 대한 규칙 기반 원인 분석 결과를 2~3문장으로 해석하라. 새로운 사실은 추가하지 마라:\n"
                 + json.dumps(payload, ensure_ascii=False, default=str), max_tokens=300)
