"""환경변수 기반 설정. Secret 은 서버 환경변수/.env 에만 존재하며 프론트엔드로 절대 전달하지 않는다."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


def _f(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _i(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'predictor.db'}")

    # 외부 데이터 API 키 (없으면 해당 수집기는 DATA_UNAVAILABLE 로 기록)
    krx_id: str | None = os.getenv("KRX_ID")
    krx_pw: str | None = os.getenv("KRX_PW")
    dart_api_key: str | None = os.getenv("DART_API_KEY")
    ecos_api_key: str | None = os.getenv("ECOS_API_KEY")
    kis_app_key: str | None = os.getenv("KIS_APP_KEY")
    kis_app_secret: str | None = os.getenv("KIS_APP_SECRET")
    kis_base_url: str = os.getenv("KIS_BASE_URL", "https://openapi.koreainvestment.com:9443")
    naver_client_id: str | None = os.getenv("NAVER_CLIENT_ID")
    naver_client_secret: str | None = os.getenv("NAVER_CLIENT_SECRET")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")

    # 수집 대상: ALL 또는 시가총액 상위 N (전체 시장은 분석 시간이 길어짐)
    universe_markets: tuple[str, ...] = tuple(os.getenv("UNIVERSE_MARKETS", "KOSPI,KOSDAQ").split(","))
    universe_top_n: int = _i("UNIVERSE_TOP_N", 0)  # 0 = 전체

    # Overnight 다단계 분석 일정 (KST). (단계명, 분석시각, 장중 스냅샷 수집시각, 대상 종목 수: 0=전체)
    # 각 단계는 분석시각까지 공개된 데이터만 사용. 15:20 최종 단계가 공식 Overnight 후보.
    stages: tuple = (
        ("S1400", time(14, 0), time(13, 57), 0),    # 전체 종목 1차 분석 및 후보 풀 선정
        ("S1430", time(14, 30), time(14, 27), 0),   # 가격·거래량·수급 업데이트 후 재평가
        ("S1500", time(15, 0), time(14, 58), 60),   # 후보 재평가 및 압축
        ("S1510", time(15, 10), time(15, 8), 30),   # 최종 분석
        ("S1520", time(15, 20), time(15, 18), 20),  # Overnight 후보 확정 (종가 단일가 매수)
    )
    final_stage: str = "S1520"
    analysis_time: time = time(14, 0)
    grading_time: time = time(18, 30)
    max_final_picks: int = _i("MAX_FINAL_PICKS", 5)
    high_hit_pct: float = _f("HIGH_HIT_PCT", 0.02)             # 다음날 장중 고가 +2% 도달 여부
    min_edge_net: float = _f("MIN_EDGE_NET", 0.002)            # 비용 차감 후 기대수익 최소 +0.2%
    edge_min_days: int = _i("EDGE_MIN_DAYS", 40)               # 전략 우위 검증 최소 OOS 거래일
    edge_min_tstat: float = _f("EDGE_MIN_TSTAT", 1.5)          # OOS 일별 순수익 t-stat 기준

    # 데이터 공개 시각 규칙 (point-in-time)
    bar_available_time: time = time(16, 0)        # 일봉 확정
    flow_available_time: time = time(18, 0)       # 투자자별 매매 확정
    short_balance_lag_days: int = 3               # 공매도 잔고는 T+2 공시 → 보수적으로 T+3

    # 예측/후보 선정 파라미터
    min_history_days: int = _i("MIN_HISTORY_DAYS", 120)
    min_avg_value_krw: float = _f("MIN_AVG_VALUE_KRW", 1e9)     # 20일 평균 거래대금 10억 미만 → 유동성 부족
    max_atr_pct_candidate: float = _f("MAX_ATR_PCT", 0.08)      # ATR/가격 8% 초과 → 고변동성, 후보 제외
    candidate_min_p: float = _f("CANDIDATE_MIN_P", 0.55)
    min_oos_bss: float = _f("MIN_OOS_BSS", 0.002)              # 이 이하이면 Confidence 는 Low
    min_calib_n: int = _i("MIN_CALIB_N", 300)                   # 보정 표본 최소치
    calib_shrink_n0: int = _i("CALIB_SHRINK_N0", 500)
    neutral_band: float = _f("NEUTRAL_BAND", 0.001)             # |수익률|<0.1% → 보합(NEUTRAL)
    conflict_rel_tol: float = _f("CONFLICT_REL_TOL", 0.005)

    # 전략 개선
    holdout_days: int = _i("HOLDOUT_DAYS", 40)
    min_holdout_n: int = _i("MIN_HOLDOUT_N", 400)
    retrain_every: int = _i("RETRAIN_EVERY", 20)               # 워크포워드 재학습 주기(거래일)

    # 백테스트 기본값
    fee_bps: float = _f("FEE_BPS", 1.5)          # 편도 수수료
    tax_bps: float = _f("TAX_BPS", 18.0)         # 매도 시 거래세(2025~ 0.15%+농특세 등, 시장별 상이 → 설정)
    slippage_bps: float = _f("SLIPPAGE_BPS", 10.0)

    allow_synthetic: bool = os.getenv("ALLOW_SYNTHETIC", "0") == "1"


    @property
    def round_trip_cost(self) -> float:
        """매수 수수료+슬리피지, 매도 수수료+세금+슬리피지 (수익률 단위)."""
        return (2 * self.fee_bps + self.tax_bps + 2 * self.slippage_bps) / 1e4


settings = Settings()
