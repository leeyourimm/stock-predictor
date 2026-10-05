"""확률 보정: 최종 Score → 실제 상승 확률.

p = sigmoid(a + b·S) 를 과거 (Score, 실제결과) 표본에 L2 로지스틱 회귀(Platt scaling)로 적합하고,
표본 수 n 이 작을수록 기저율(base rate)로 수축한다: p = base + (p_fit − base) · n/(n + n0).
표본이 0이면 p = 기저율(사실상 50% 근처) → '모른다'를 그대로 표현.
"""
from __future__ import annotations

import numpy as np

from ..config import settings


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logistic_fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """Newton-Raphson L2 로지스틱 회귀. X 에 절편 열 포함 가정, 절편은 규제하지 않음."""
    n, k = X.shape
    w = np.zeros(k)
    reg = np.full(k, l2)
    reg[0] = 0.0
    for _ in range(iters):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + reg * w
        H = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg) + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def fit(scores, ups, returns=None, prior: dict | None = None) -> dict:
    s = np.asarray(scores, float)
    y = np.asarray(ups, float)
    n = len(s)
    if n == 0:
        return {"a": 0.0, "b": 0.0, "n": 0, "base_rate": 0.5, "c": 0.0, "d": 0.0}
    base = float(np.clip(y.mean(), 0.05, 0.95))
    a, b = logistic_fit(np.c_[np.ones(n), s], y, l2=max(1.0, n * 0.001))
    out = {"a": float(a), "b": float(b), "n": int(n), "base_rate": base}
    if returns is not None:
        r = np.asarray(returns, float)
        X = np.c_[np.ones(n), s]
        c, d = np.linalg.lstsq(X, r, rcond=None)[0]
        out.update(c=float(c), d=float(d))
    else:
        out.update(c=0.0, d=0.0)
    return out


def predict_proba(cal: dict, scores) -> np.ndarray:
    s = np.asarray(scores, float)
    base, n = cal.get("base_rate", 0.5), cal.get("n", 0)
    p_fit = sigmoid(cal.get("a", 0.0) + cal.get("b", 0.0) * s)
    shrink = n / (n + settings.calib_shrink_n0) if n else 0.0
    return np.clip(base + (p_fit - base) * shrink, 0.01, 0.99)


def expected_return(cal: dict, scores) -> np.ndarray:
    s = np.asarray(scores, float)
    n = cal.get("n", 0)
    shrink = n / (n + settings.calib_shrink_n0) if n else 0.0
    return (cal.get("c", 0.0) + cal.get("d", 0.0) * s) * shrink


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def calibration_table(p, y, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum():
            out.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                        "actual_up_rate": float(y[m].mean())})
    return out
