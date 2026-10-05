"""학습형 Overnight 예측 모델 (코드 계산, 블랙박스 라이브러리 없이 numpy 로 구현).

원칙: 어떤 지표가 도움이 된다고 미리 정하지 않는다.
- 모든 후보 특징을 강하게 정규화된(L2) 선형 모델에 넣고, 과거 실제 결과로 계수(방향·크기)를 학습한다.
- 효과 없는 특징은 계수가 0 근처로 수축된다. 유효성은 OOS 성과(feature_study, strategy.manager)로 따로 검증한다.
- 타깃 4개를 따로 학습: 다음날 종가 상승(up), 시가 갭상승(gap), 장중 고가 +HIGH_HIT_PCT 도달(hit), 비용 차감 종가수익(ret).
- 전처리: 학습 데이터의 중앙값/IQR 로 robust z-score(±3 clip), 결측은 0(=중앙값) — 단계·유니버스와 무관하게 동일 변환.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..config import settings
from .calibration import logistic_fit, sigmoid

# (특징, 그룹, 한글 라벨). m_ 접두사는 시장 전체 값(모든 종목 동일).
FEATURES: list[tuple[str, str, str]] = [
    ("ret1", "technical", "전일 수익률"), ("ret5", "technical", "5일 수익률"), ("ret20", "technical", "20일 수익률"),
    ("ret60", "technical", "60일 수익률"), ("dist_ma20", "technical", "20일선 이격"),
    ("ma_aligned_up", "technical", "이동평균 정배열"), ("rsi14", "technical", "RSI(14)"),
    ("macd_hist", "technical", "MACD 히스토그램"), ("bb_pctb", "technical", "볼린저 %B"),
    ("volume_ratio", "technical", "전일 거래량 급증"), ("streak", "technical", "연속 상승/하락"),
    ("upper_wick_last", "technical", "전일 윗꼬리"),
    ("intraday_ret", "intraday", "당일 등락률(현재)"), ("ret_since_open", "intraday", "시가 대비"),
    ("gap_today", "intraday", "당일 시가 갭"), ("vwap_gap", "intraday", "VWAP 대비 현재가"),
    ("dist_day_high", "intraday", "당일 고가 대비 현재가"), ("day_range_pos", "intraday", "당일 범위 내 위치"),
    ("broke_prev_high", "intraday", "전일 고가 돌파"), ("value_pace", "intraday", "거래대금 페이스"),
    ("mom_since_first", "intraday", "14:00 이후 모멘텀"), ("late_volume_accel", "intraday", "장후반 거래 강도 변화"),
    ("foreign_1", "flow", "외국인 전일 순매수"), ("foreign_5", "flow", "외국인 5일 순매수"),
    ("foreign_20", "flow", "외국인 20일 순매수"), ("inst_5", "flow", "기관 5일 순매수"),
    ("indiv_5", "flow", "개인 5일 순매수"), ("short_ratio_5", "flow", "공매도 비중"),
    ("sector_rel5", "relative", "업종 상대강도(5일)"), ("stock_rel_sector5", "relative", "업종 대비 종목 강도"),
    ("rel_market_intraday", "relative", "당일 시장 대비 강도"),
    ("vol20", "risk", "20일 변동성"), ("atr_pct", "risk", "ATR%"), ("value_avg20", "risk", "평균 거래대금"),
    ("disclosures_3d", "event", "최근 공시"), ("earnings_disclosure_3d", "event", "실적 공시"),
    ("m_spx_ret1", "market", "미국 S&P500 전일"), ("m_nasdaq_ret1", "market", "나스닥 전일"),
    ("m_sox_ret1", "market", "필라델피아 반도체 전일"), ("m_vix", "market", "VIX"),
    ("m_usdkrw_chg5", "market", "원/달러 5일 변화"), ("m_kospi_ret5", "market", "KOSPI 5일"),
    ("m_kospi_intraday", "market", "KOSPI 당일(현재)"), ("m_risk_on", "market", "Risk-on 지표"),
]
FEATURE_NAMES = [f for f, _, _ in FEATURES]
FEATURE_GROUP = {f: g for f, g, _ in FEATURES}
FEATURE_LABEL = {f: lbl for f, _, lbl in FEATURES}
GROUPS = ["technical", "intraday", "flow", "relative", "market", "event", "news", "risk"]
GROUP_LABELS = {"technical": "기술적 분석", "intraday": "장중 흐름", "flow": "수급", "relative": "상대강도",
                "market": "시장환경", "event": "공시/실적", "news": "뉴스", "risk": "변동성/유동성"}
TARGETS = ("up", "gap", "hit", "ret")
POINT = 0.05   # logit 0.05 = 1점 (상승확률 약 +1.2%p)


def feature_frame(table: pd.DataFrame, market: dict) -> pd.DataFrame:
    """FeatureSet.table + 시장값 → 모델 입력 원값 (index=ticker)."""
    X = pd.DataFrame(index=table.index)
    for f in FEATURE_NAMES:
        if f.startswith("m_"):
            v = market.get(f[2:])
            X[f] = np.nan if v is None else float(v)
        else:
            X[f] = pd.to_numeric(table[f], errors="coerce") if f in table else np.nan
    return X


def outcome_targets(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {"up": (df.close_ret > 0).astype(float).values, "gap": (df.open_ret > 0).astype(float).values,
            "hit": (df.high_ret >= settings.high_hit_pct).astype(float).values,
            "ret": np.clip(df.close_ret.values - settings.round_trip_cost, -0.15, 0.15)}


class Model:
    def __init__(self, spec: dict | None):
        self.spec = spec

    @property
    def trained(self) -> bool:
        return bool(self.spec) and self.spec.get("n_train", 0) > 0

    # ---------------------------------------------------------------- 학습
    @classmethod
    def fit(cls, rows: pd.DataFrame, l2_per_sample: float = 0.02) -> "Model":
        """rows: FEATURE_NAMES 컬럼 + close_ret/open_ret/high_ret/low_ret/vol20."""
        rows = rows[rows.close_ret.notna()]
        n = len(rows)
        if n < 50:
            return cls({"n_train": 0})
        X = rows.reindex(columns=FEATURE_NAMES).astype(float)
        med = X.median()
        iqr = (X.quantile(0.75) - X.quantile(0.25)).replace(0, np.nan)
        usable = iqr.notna() & (X.notna().mean() > 0.2)
        Z = cls._z(X, med, iqr, usable)
        A = np.c_[np.ones(n), Z.values]
        tg = outcome_targets(rows)
        coefs = {}
        lam = max(5.0, l2_per_sample * n)
        for t in ("up", "gap", "hit"):
            y = tg[t]
            if y.min() == y.max():
                w = np.r_[math.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)), np.zeros(Z.shape[1])]
            else:
                w = logistic_fit(A, y, l2=lam)
            coefs[t] = {"b0": float(w[0]), "w": dict(zip(FEATURE_NAMES, map(float, w[1:])))}
        y = tg["ret"]
        reg = np.full(A.shape[1], lam * 0.25)
        reg[0] = 0
        w = np.linalg.solve(A.T @ A + np.diag(reg), A.T @ y)
        coefs["ret"] = {"b0": float(w[0]), "w": dict(zip(FEATURE_NAMES, map(float, w[1:])))}
        resid = y - A @ w
        vol = rows.vol20.clip(lower=0.003).fillna(rows.vol20.median()).values
        rng = {"resid_scale": float(np.nanstd(resid / vol)),
               "up_q50": float(np.nanquantile(rows.high_ret / vol, 0.5)),
               "up_q80": float(np.nanquantile(rows.high_ret / vol, 0.8)),
               "dn_q50": float(np.nanquantile(rows.low_ret / vol, 0.5)),
               "dn_q20": float(np.nanquantile(rows.low_ret / vol, 0.2))}
        dates = sorted(rows.trade_date.unique()) if "trade_date" in rows else []
        return cls({"n_train": int(n), "train_period": [str(dates[0]), str(dates[-1])] if dates else None,
                    "median": {k: (None if pd.isna(v) else float(v)) for k, v in med.items()},
                    "iqr": {k: (None if pd.isna(v) else float(v)) for k, v in iqr.items()},
                    "usable": [k for k, u in usable.items() if u], "coefs": coefs, "range": rng, "l2": lam,
                    "base_rates": {t: float(np.mean(tg[t])) for t in ("up", "gap", "hit")}})

    @staticmethod
    def _z(X: pd.DataFrame, med, iqr, usable) -> pd.DataFrame:
        Z = ((X - med) / iqr).clip(-3, 3).fillna(0.0)
        for c in Z.columns:
            if not usable.get(c, False):
                Z[c] = 0.0
        return Z

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        s = self.spec
        med = pd.Series(s["median"], dtype=float).reindex(FEATURE_NAMES)
        iqr = pd.Series(s["iqr"], dtype=float).reindex(FEATURE_NAMES)
        usable = pd.Series({f: f in set(s["usable"]) for f in FEATURE_NAMES})
        return self._z(X.reindex(columns=FEATURE_NAMES).astype(float), med, iqr, usable)

    # ---------------------------------------------------------------- 예측
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        vol = X["vol20"].astype(float)
        if not self.trained:
            out["p_up"] = out["p_gap"] = out["p_hit"] = 0.5
            out["exp_net"] = -settings.round_trip_cost
            out["exp_close"] = 0.0
            out["logit_up"] = 0.0
            out["contrib"] = [dict() for _ in range(len(X))]
            rs = {"resid_scale": 1.0, "up_q50": 0.5, "up_q80": 1.2, "dn_q50": -0.5, "dn_q20": -1.2}
        else:
            Z = self.transform(X)
            c = self.spec["coefs"]
            for t, col in (("up", "p_up"), ("gap", "p_gap"), ("hit", "p_hit")):
                w = pd.Series(c[t]["w"]).reindex(FEATURE_NAMES).fillna(0).values
                logit = c[t]["b0"] + Z.values @ w
                out[col] = np.clip(sigmoid(logit), 0.01, 0.99)
                if t == "up":
                    out["logit_up"] = logit
                    contrib = Z.values * w
            wr = pd.Series(c["ret"]["w"]).reindex(FEATURE_NAMES).fillna(0).values
            out["exp_net"] = c["ret"]["b0"] + Z.values @ wr
            out["exp_close"] = out["exp_net"] + settings.round_trip_cost
            out["contrib"] = [dict(zip(FEATURE_NAMES, row)) for row in contrib]
            rs = self.spec["range"]
        sig = vol.where(vol > 0)
        out["range_low"] = out["exp_close"] - 1.2816 * rs["resid_scale"] * sig
        out["range_high"] = out["exp_close"] + 1.2816 * rs["resid_scale"] * sig
        out["up_med"], out["up_p80"] = rs["up_q50"] * sig, rs["up_q80"] * sig
        out["down_med"], out["down_p20"] = rs["dn_q50"] * sig, rs["dn_q20"] * sig
        return out

    def explain(self, contrib: dict, raw: dict) -> dict:
        """그룹별 기여 점수(1점 = logit 0.05) + 주요 근거 항목."""
        groups = {g: {"label": GROUP_LABELS[g], "score": 0, "logit": 0.0, "items": []} for g in GROUPS}
        for f, v in contrib.items():
            g = FEATURE_GROUP[f]
            groups[g]["logit"] += v
            if abs(v) >= POINT / 2:
                val = raw.get(f)
                groups[g]["items"].append({"feature": f, "label": FEATURE_LABEL[f], "points": round(v / POINT, 1),
                                           "value": None if val is None or (isinstance(val, float) and math.isnan(val))
                                           else round(float(val), 4)})
        for g in groups.values():
            g["score"] = int(round(g["logit"] / POINT))
            g["logit"] = round(g["logit"], 4)
            g["items"].sort(key=lambda i: -abs(i["points"]))
            g["items"] = g["items"][:4]
        if not self.trained:
            groups["technical"]["items"].append({"label": "학습된 모델 없음 (표본 부족)", "points": 0, "value": None})
        groups["news"]["items"].append({"label": "뉴스 데이터 없음 (Phase 3)", "points": 0, "value": None})
        return groups

    def coef_table(self) -> list[dict]:
        if not self.trained:
            return []
        c = self.spec["coefs"]
        return sorted([{"feature": f, "label": FEATURE_LABEL[f], "group": FEATURE_GROUP[f],
                        **{t: round(c[t]["w"].get(f, 0.0), 5) for t in TARGETS}} for f in FEATURE_NAMES],
                      key=lambda r: -abs(r["up"]))
