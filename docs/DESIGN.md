# KR Overnight Predictor — 시스템 설계서

> 목표: **당일 장 마감 전(15:20 종가 단일가) 매수 → 다음 거래일 매도**하는 Overnight 전략의 후보를 찾는다.
> 14:00 → 14:30 → 15:00 → 15:10 → 15:20 단계마다 **그 시각까지 공개된 데이터만** 사용해 평가하고, 모든 예측을 변경 불가능하게 기록한 뒤,
> 다음 거래일 장 마감 후 시가·장중 고가/저가·종가 기준으로 비용 차감 성과를 자동 채점한다.
> 판단 기준은 적중률이 아니라 **비용과 손실을 포함한 장기 Risk-adjusted Return**. 통계적 우위가 확인되지 않으면 "오늘은 매수 후보 없음".

## 0. Overnight 다단계 일정 (KST, 거래일)

| 단계 | 스냅샷 | 분석 | 대상 | 하는 일 |
|---|---|---|---|---|
| S1400 | 13:57 | 14:00 | 전체 | 1차 분석, 후보 풀 선정 |
| S1430 | 14:27 | 14:30 | 전체 | 가격·거래량·수급·공시 업데이트 후 재평가 |
| S1500 | 14:58 | 15:00 | 직전 단계 상위 60 | 재평가·압축 |
| S1510 | 15:08 | 15:10 | 상위 30 | 최종 분석 |
| **S1520** | 15:18 | **15:20** | 상위 20 | **후보 확정 (최대 5종목)** — 공식 Overnight 후보 |

- 각 단계의 `as_of` = 분석 시각. 스냅샷 `available_at`, 일봉(16:00 확정), 수급(18:00 확정), 공시(접수 시각) 모두 `available_at ≤ as_of` 만 보인다.
  **15:20 예측에는 15:20 이후 가격·거래량·뉴스·수급이 절대 들어가지 않는다** (테스트 `test_stage_features_ignore_post_snapshot_data`).
- 단계별 결과는 `prediction_runs(trade_date, stage)` 로 각각 불변 저장 (`parent_run_id` 로 연결). 같은 날 같은 단계 재실행은 기존 기록 유지.
- 실데이터는 해당 시각 ±15분 안에서만 예측 생성 가능 (사후 생성 금지; 과거는 백테스트로만).

## 1. 전체 아키텍처

```
                ┌──────────────────────── Scheduler (APScheduler, Asia/Seoul) ───────────────────────┐
                │ 06:30 global/macro 수집 │ 14:00~15:20 5단계 분석 │ 18:30 채점·오류분석 │ 토 09:00 전략 재평가 │
                └────────────┬─────────────────────┬────────────────────┬────────────────────┬────────┘
                             ▼                     ▼                    ▼                    ▼
┌──────────────┐   ┌──────────────────┐   ┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ Data         │──▶│ Point-in-Time    │──▶│ Feature         │──▶│ Prediction       │──▶│ Prediction       │
│ Collectors   │   │ Observation Store│   │ Engineering     │   │ Engine           │   │ History (불변)   │
│ KRX/DART/ECOS│   │ (source,         │   │ (as_of 필터 강제)│   │ factor score →   │   │ append-only +    │
│ yfinance/KIS │   │  retrieved_at,   │   │ 기술/수급/시장/  │   │ calibrated prob →│   │ hash chain       │
│ Naver News   │   │  data_ts,        │   │ 이벤트/리스크    │   │ range/confidence │   └────────┬─────────┘
└──────────────┘   │  available_at)   │   └─────────────────┘   │ → 순위/후보없음  │            │
                   │ + 출처간 충돌감지 │                          └────────┬─────────┘            ▼
                   └──────────────────┘                                   │            ┌──────────────────┐
                                                                          ▼            │ Evaluation Engine│
                                                                ┌──────────────────┐   │ 채점, Brier,     │
                                                                │ Claude Analysis  │   │ Calibration,     │
                                                                │ (설명/뉴스해석/   │   │ Regime별 성능,   │
                                                                │  오류원인 서술만) │   │ Error Log        │
                                                                └──────────────────┘   └────────┬─────────┘
                                                                                                ▼
┌──────────────┐   ┌──────────────────┐                                            ┌──────────────────────┐
│ Frontend     │◀──│ FastAPI Backend  │◀───────────────────────────────────────────│ Strategy Manager     │
│ (대시보드/    │   │ /api/...         │      Backtesting Engine (동일 피처코드 재사용)│ champion/challenger, │
│  상세/백테스트)│   └──────────────────┘                                            │ OOS 검증 후에만 채택 │
└──────────────┘                                                                   │ Version History      │
                                                                                   └──────────────────────┘
```

핵심 흐름: **DATA → CODE/STATISTICAL MODEL → PREDICTION → CLAUDE EXPLANATION**.
확률, 지표, 통계, 백테스트, 채점은 전부 Python 코드가 계산한다. Claude 는 계산 결과를 입력으로 받아 설명문, 뉴스/공시 해석, 실패 원인 서술만 생성하며, Claude 가 만든 텍스트는 어떤 숫자에도 다시 들어가지 않는다 (뉴스 sentiment 는 Phase 3에서 *별도 factor* 로 추가하되, 성능 검증 전까지 가중치 0으로 시작).

## 2. 기술 스택

| 영역 | 선택 | 이유 |
|---|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn | 금융 데이터 생태계(pandas/numpy/pykrx) |
| DB | SQLAlchemy 2 + SQLite(Phase 1) → PostgreSQL(운영) | `DATABASE_URL` 하나로 전환. 불변성 트리거 양쪽 제공 |
| 수치 계산 | pandas, numpy (모델/보정 직접 구현, 블랙박스 의존 최소화) | 재현성·감사 가능성 |
| 스케줄러 | APScheduler (Asia/Seoul) / 운영 시 cron·systemd timer 도 가능 | |
| Frontend | 정적 HTML + Vanilla JS + Chart.js (FastAPI가 서빙) | Phase 1은 데이터 파이프라인이 우선. Phase 4에서 Next.js 전환 가능 |
| LLM | Anthropic Claude API (서버 측에서만 호출, `ANTHROPIC_API_KEY`) | 설명 생성 전용 |
| Secret | 환경변수 / `.env` (서버에만 존재, 프론트 미포함) | |

## 3. 데이터 소스와 Point-in-Time 규칙

모든 외부 값은 `observations` 테이블에 `source`, `retrieved_at`, `data_timestamp`, `available_at` 과 함께 저장된다.
**피처 계산은 `available_at <= as_of` 인 값만 조회할 수 있다** (`PointInTimeStore` 가 강제, 테스트로 검증).

| 데이터 | 소스(우선순위) | available_at 규칙 | 상태 |
|---|---|---|---|
| 일봉 OHLCV, 거래대금, 등락률 | KRX 정보데이터시스템(pykrx, `KRX_ID/KRX_PW`) → FinanceDataReader(교차검증) | 해당일 16:00 KST | Phase 1 구현 |
| 14:00 현재가/장중 누적거래량 | 한국투자증권 KIS Open API (`KIS_APP_KEY/SECRET`) | 조회 시각 | Phase 1 수집기 구현(키 필요). 없으면 "데이터 없음" 처리 |
| 투자자별 순매수(외국인/기관/개인) | KRX(pykrx) | 해당일 18:00 (확정치) → 14:00 예측엔 **T-1까지** 사용 | Phase 1 |
| 공매도 거래량/잔고 | KRX(pykrx) | 거래량 T+0 18:00, 잔고 T+2 공시 → 보수적으로 T+3 영업일 | Phase 1 |
| 프로그램 매매 | KRX 정보데이터시스템 (종목별 무료 API 없음) | — | **데이터 없음** (Phase 2: KIS API 프로그램매매 동향) |
| KOSPI/KOSDAQ/업종 지수 | KRX(pykrx) | 해당일 16:00 | Phase 1 |
| 원/달러 환율 | 한국은행 ECOS (`ECOS_API_KEY`) → yfinance `KRW=X` 교차검증 | ECOS 공표일 / yfinance 데이터시각 | Phase 1 |
| 미국 지수(S&P500, NASDAQ, SOX, VIX, 10Y) | yfinance | 미국 장마감(다음날 06:00 KST) | Phase 1 |
| 금리·거시(기준금리, 국고채3Y) | 한국은행 ECOS | 공표일 | Phase 1 |
| 공시 | DART OpenAPI (`DART_API_KEY`) | 접수 시각(rcept_dt) | Phase 1 수집, Phase 2 이벤트 factor |
| 실적(재무제표) | DART 정기보고서 API | 공시 접수일 | Phase 2 |
| 컨센서스 변화 | 무료 공식 소스 없음 (FnGuide 등 유료) | — | **데이터 없음** |
| 기업/업종/정책 뉴스 | 네이버 검색 API (`NAVER_CLIENT_ID/SECRET`) | 기사 발행 시각 | Phase 3 (Claude 분류 → 검증 후 가중치) |
| 거래정지/관리종목 | KRX 상장종목 정보(FinanceDataReader `KRX-ADMIN` 등) | 조회 시각 | Phase 1 (실패 시 "확인 불가" 위험 플래그) |

- API 실패, 키 미설정, 값 없음 → 값을 만들지 않고 `DATA_UNAVAILABLE` 로 기록. 피처는 `None`, 해당 factor 는 0점 + 데이터 완전성 감소 → Confidence 하락.
- 같은 (종목, 필드, 기준일)에 다른 출처 값이 있으면 상대 오차가 임계값(가격 0.5%) 초과 시 `data_conflicts` 에 기록하고 해당 종목 Confidence 를 Low 로 강등.
- `synthetic` 소스는 **테스트 전용**. DB/화면에 항상 "SYNTHETIC — 실제 데이터 아님" 표시.

### 백테스트의 시점 처리
워크포워드: 각 거래일 T 에 `as_of = T 15:20` 뷰로 라이브와 같은 코드를 실행하고, 모델은 `target_date < T` 표본으로만 재학습, 게이트도 T 이전 OOS 성과로만 판단.
과거 장중 스냅샷은 무료로 구할 수 없으므로 저장된 스냅샷(라이브 누적 또는 `import-intraday` 로 넣은 분봉 CSV)이 없는 날은 장중 피처가 "데이터 없음",
진입가는 당일 종가로 대체한다(보수적). 청산은 다음날 시가 또는 종가를 선택.

## 4. 데이터베이스 스키마 (주요 테이블)

| 테이블 | 용도 | 핵심 컬럼 |
|---|---|---|
| `instruments` | 종목 마스터 | ticker, name, market, sector, is_halted, is_admin_issue, status_source, status_checked_at |
| `observations` | 모든 외부 원천값 (Point-in-time) | entity, field, data_date, value, source, retrieved_at, data_timestamp, available_at, status(OK/UNAVAILABLE) |
| `daily_bars` | 정규화된 일봉 (조회 편의용, observations 에서 파생) | ticker, date, OHLCV, value, source, available_at |
| `investor_flows` | 투자자별 순매수 | ticker, date, foreign_net, institution_net, individual_net, source, available_at |
| `index_bars` | 지수/환율/해외지수 | symbol, date, close, ..., source, available_at |
| `data_conflicts` | 출처 간 불일치 | entity, field, data_date, source_a, value_a, source_b, value_b, rel_diff |
| `strategy_versions` | 전략 버전 이력 | version, weights(json), params(json), parent_version, status(champion/candidate/rejected/retired), created_at, activated_at, evaluation(json), reason |
| `prediction_runs` | 단계별 실행 단위 | run_id, **stage**, parent_run_id, as_of, trade_date, target_date, strategy_version, regime, n_universe, n_scored, no_candidate, summary(json), data_sources(json) — unique(trade_date, stage) |
| `predictions` | **불변** 예측 기록 | run_id, stage, ticker, trade_date, target_date, price_at_prediction, price_source, p_up, p_down, expected_return, expected_net_return, p_gap_up, p_high_hit, up_med, up_p80, down_med, down_p20, range_low/high, confidence, is_candidate, factor_scores, risk_flags, features_snapshot(json: 모델 입력값·출처·데이터없음 항목), rank, created_at, prev_hash, row_hash |
| `prediction_outcomes` | **불변** 채점 결과 (예측과 분리) | prediction_id, entry_price, entry_basis, next_open/high/low/close, open/high/low_return, actual_return(종가), net_open_return, net_close_return, gap_up, high_hit, result, trade_result(WIN/LOSS), graded_at |
| `intraday_snapshots` | 장중 스냅샷 (KIS) | ticker, ts, price, open, high, low, cum_volume, cum_value, source, available_at |
| `training_samples` | 학습·검증 표본 (BACKTEST/LIVE 구분) | trade_date, stage, ticker, source, features(json), open/high/low/close_ret, is_synthetic |
| `error_logs` | 실패 원인 | prediction_id, tags(json), primary_cause, details(json), llm_explanation |
| `performance_snapshots` | 일별 누적 성과 | date, metrics(json) |
| `backtest_runs` | 백테스트 결과 | params, strategy_version, metrics(json), equity_curve(json) |

### 불변성 보장
1. `predictions`, `prediction_runs`, `prediction_outcomes` 에 `BEFORE UPDATE / BEFORE DELETE` 트리거 → 수정·삭제 시 DB 오류.
2. 각 예측 행은 `row_hash = sha256(prev_hash + canonical_json(내용))` 해시 체인. `/api/integrity` 가 전체 체인을 재계산해 변조 여부를 공개.
3. 채점 결과는 별도 테이블(`prediction_outcomes`)에만 기록되므로 원 예측을 건드리지 않는다.

## 5. 예측 엔진 (Overnight)

1. **피처 (44개, `prediction/model.py` FEATURES)** — 어느 것도 유효하다고 미리 가정하지 않음
   - 기술적: 1/5/20/60일 수익률, 20일선 이격, 정배열, RSI, MACD, %B, 전일 거래량 급증, 연속 상승/하락, 전일 윗꼬리
   - 장중(스냅샷): 당일 등락률, 시가 대비, 시가 갭, VWAP 대비, 당일 고가 대비, 당일 범위 내 위치, 전일 고가 돌파,
     거래대금 페이스, 14:00 이후 모멘텀, 장 후반 거래 강도 변화, 당일 시장 대비 강도
   - 수급: 외국인 1/5/20일, 기관 5일, 개인 5일(전일까지 확정치), 공매도 비중 / 프로그램 매매: 데이터 없음
   - 상대강도: 업종 5일, 업종 대비 종목 / 시장: 미국 S&P·나스닥·SOX 전일, VIX, 원/달러, KOSPI 5일·당일, Risk-on 지표 / 미국 선물 장중: 데이터 없음
   - 이벤트: 최근 공시, 실적 공시 / 리스크: 변동성, ATR, 거래대금
2. **모델**: robust z-score(중앙값/IQR, ±3 클리핑, 결측=0) → L2 로지스틱 3개(종가 상승, 갭 상승, 장중 고가 +2% 도달) + 릿지 회귀(비용 차감 종가 수익).
   상승/하락 범위 = 학습 잔차 분위수를 종목 변동성으로 스케일. 그룹 기여도(1점 = logit 0.05)로 상승 근거/하락 위험 표시.
3. **출력(종목별)**: 상승확률, 예상수익률(비용 전/후), 예상 상승범위(장중 최대 중앙값~80분위), 예상 하락범위(장중 최저 중앙값~20분위),
   장중 고가 도달 가능성, 갭 상승 가능성, Confidence, 상승 근거, 하락 위험, 위험 플래그.
4. **전략 우위 게이트**: champion 의 OOS 그림자 포트폴리오(매일 기대수익 상위 5 매수 → 다음날 종가 매도, 비용 차감)가
   최근 250일 중 ≥40일, 평균 > 0, t ≥ 1.5 일 때만 통과. 미통과면 모든 Confidence = Low → 후보 없음.
5. **후보 조건**: 게이트 통과 & 자격(정지/관리/유동성/고변동/데이터부족/출처불일치 아님) & 상승확률 ≥ 0.55 & 비용 차감 기대수익 ≥ 0.2% & Confidence ≥ Medium.
   15:20 단계는 이 중 기대수익 상위 최대 5종목. 하나도 없으면 **"오늘은 매수 후보 없음"**.

## 6. 채점·평가

- 진입가 = 15:20 스냅샷 가격(없으면 당일 종가, `entry_basis` 에 기록). 다음 거래일 시가/고가/저가/종가 수익률과
  비용 차감 순수익(왕복 0.41% = 수수료 1.5bp×2 + 거래세 18bp + 슬리피지 10bp×2)을 `prediction_outcomes` 에 별도 저장.
- 지표: Overnight 시가 수익률, 다음날 장중 최대 수익률·최대 손실, 종가 수익률, 승률, Profit Factor, Sharpe, MDD (종가 청산/시가 청산 각각),
  방향 적중률(전체·최근 20일·TOP5/10·후보·Confidence별·국면별), Brier/BSS, Calibration(상승·갭·고가도달), 단계별(14:00→15:20) 비교.
- 오류 분석 태그: GAP_DOWN, FADE_AFTER_GAP, INTRADAY_REVERSAL, LATE_MOMENTUM_FADE, COST_EATEN, MARKET_WIDE_MOVE, `*_OVERWEIGHTED` 등.
- 특징 검증(`research/feature_study.py`, 화면 "특징 검증"): 특징별 일별 횡단면 Spearman IC(종가·시가 수익), 5분위 스프레드, 국면별 IC를
  학습 구간과 OOS 로 나눠 계산. 학습 |t|≥2 이고 OOS 같은 방향 |t|≥1.5 일 때만 "OOS 유효".

## 7. 전략 개선 (과최적화 방지)

- 매 거래일 채점 표본(LIVE)이 `training_samples` 에 쌓이고, 주 1회 `reevaluate()`:
  1. 직전 champion 학습 이후 들어온 **새 홀드아웃 구간**(최근 40일)을 분리, 그 이전 표본으로 challenger 학습.
  2. 홀드아웃에서 champion vs challenger 그림자 상위 5 일별 순수익 차이를 블록 부트스트랩 → 95% 하한 > 0 이고 Brier 악화 ≤ 0.001 일 때만 채택.
  3. 새 학습 데이터가 없으면 보류. 채택·거절 모두 `strategy_versions` 에 근거와 함께 기록.

## 8. 시장 Regime

KOSPI 기준: 60일 수익률·MA60 기울기로 Bull/Bear/Sideways, 20일 실현변동성의 1년 분위수로 High/Low Volatility.
결과는 `TREND|VOL` (예: `BULL|LOW_VOL`) 로 모든 예측에 저장되어 Regime별 성능 집계에 사용.

## 9. 자동화

| 시각(KST, 거래일) | 작업 |
|---|---|
| 06:30 | 전일 일봉·수급·공매도 확정치, 미국 지수·환율·거시, 공시 수집 |
| 13:57 / 14:27 / 14:58 / 15:08 / 15:18 | KIS 장중 스냅샷(종목 + KOSPI/KOSDAQ 지수) |
| 14:00 / 14:30 / 15:00 / 15:10 / 15:20 | `run_stage` S1400 … S1520 (15:20 = 후보 확정) |
| 18:30 | 당일 확정치 수집 → 전 거래일 예측 Overnight 채점 → 오류 분석 → 학습 표본 저장 → 성과 스냅샷 |
| 토 09:00 | 전략 재평가 (champion/challenger) |

휴장일이면 모두 skip. 운영: `python -m app.scheduler` 또는 cron 에서 `python -m app.cli stage S1520 | grade | reevaluate`.

## 10. 개발 단계

| Phase | 내용 | 상태 |
|---|---|---|
| **1** | 불변 기록·해시체인, PIT 저장소, 수집기(KRX·KIS 장중·DART·ECOS·yfinance), 5단계 Overnight 분석, 학습 모델·우위 게이트, Overnight 채점·오류분석, 특징 OOS 검증, 워크포워드 백테스트(시가/종가 청산), champion/challenger, API·화면, 스케줄러 | **구현 완료 (합성 데이터로 검증, 실데이터는 키·네트워크 필요)** |
| 2 | 실데이터 적재·장중 스냅샷 누적 후 장중 피처 실검증, 프로그램매매(KIS), 미국 선물 장중, PostgreSQL, 인증 | |
| 3 | 뉴스 수집 + Claude 분류 sentiment 피처(검증 후 편입) | |
| 4 | 알림(텔레그램/이메일), 모니터링, 배포(Docker) | |
