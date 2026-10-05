# KR Overnight Predictor (Phase 1)

**당일 장 마감 전(15:20) 매수 → 다음 거래일 매도** 하는 Overnight 전략의 후보를 찾는 시스템.
14:00 → 14:30 → 15:00 → 15:10 → 15:20 단계마다 **그 시각까지 공개된 데이터만으로** 평가하고, 예측을 변경 불가능하게 저장한 뒤
다음 거래일 시가·장중 고가/저가·종가 기준 비용 차감 성과로 자동 채점한다. 통계적 우위가 확인되지 않으면 **"오늘은 매수 후보 없음"**.

- 설계서: [docs/DESIGN.md](docs/DESIGN.md) (단계 일정, 아키텍처, DB, 데이터 공개시각 규칙, 모델·게이트, 평가, 자동화)
- 흐름: `DATA → CODE/STATISTICAL MODEL → PREDICTION → CLAUDE EXPLANATION` (모든 수치는 코드 계산, Claude 는 설명만)

## 구조
```
backend/app/
  config.py, db.py            설정(.env) / 모델 + 불변 트리거 + 해시체인
  calendar_kr.py              KRX 휴장일
  collectors/                 krx(pykrx) · kis(장중 스냅샷, 분봉 CSV 가져오기) · secondary(FDR) · global_markets(yfinance) · ecos · dart · synthetic(테스트 전용)
  features/                   store(Point-in-Time 뷰) · technical · builder(일봉+장중 피처) · regime
  prediction/                 model(44개 피처, 로지스틱/릿지) · engine(범위·위험·Confidence·우위 게이트·후보) · calibration
  evaluation/                 outcomes(Overnight 수익) · metrics · error_analysis
  research/feature_study.py   특징별 IC 학습/OOS 검증
  backtest/engine.py          워크포워드 Overnight 백테스트 (시가/종가 청산, 비용 반영)
  strategy/manager.py         학습 표본, champion/challenger(새 홀드아웃 + 블록 부트스트랩)
  explain/claude.py           Claude 설명 (키 없으면 템플릿)
  pipeline.py                 수집 → 단계별 분석·저장 → 채점·오류분석 → 성과 스냅샷
  cli.py, scheduler.py        명령행 / 자동 스케줄
  api/main.py                 REST API + 프론트 서빙
frontend/                     대시보드 · 종목 상세 · 성과/검증 · 특징 검증 · 백테스트 · 데이터 상태
backend/tests/                단계별 look-ahead · 불변성 · 해시체인 · Overnight 비용 계산
```

## 실데이터로 운영
```bash
cd backend && pip install -r requirements.txt
# backend/.env 에 키 입력 (KRX_ID/PW, KIS_APP_KEY/SECRET 필수, DART·ECOS 권장, ANTHROPIC 선택) — 서버에만 둠
python -m app.cli init-db
python -m app.cli collect --start 2025-01-01 --end 2026-10-02     # 과거 일봉·수급·지수·공시 적재
python -m app.cli import-intraday --csv minute_bars.csv          # (선택) 과거 분봉이 있으면 단계별 스냅샷 생성
python -m app.cli bootstrap                                       # 워크포워드 백테스트 → v1 모델과 OOS 우위 기록
python -m app.cli launch                                          # 앱 실행 시 분석 (밀린 수집·채점 → 지금 시점 분석)
uvicorn app.api.main:app --port 8000                              # http://localhost:8000
```
맥에서는 `bash start_real.command` 한 번이면 위 과정을 모두 처리한다 (처음 1회만 과거 데이터 수집·초기 학습).

### 분석 시점: 앱을 켤 때마다
- 켤 때마다 밀린 확정 데이터 수집 → 결과가 나온 예측 채점 → (7일마다) 전략 재평가 → **지금 시점까지 공개된 데이터로** 오늘 후보 분석.
- 거래일 15:30(종가 단일가 주문 마감) 전에 켰을 때만 오늘 매수 후보를 만든다. 이후·휴장일에는 갱신·채점만 한다.
- 단계명은 실행 시각 `L{HHMM}`. 같은 날 여러 번 실행하면 모두 불변 기록되고, 성과 집계에는 그날 마지막 실행만 쓴다.
- 화면의 "지금 다시 분석" 버튼도 같은 동작. 고정 일정(14:00~15:20 5단계, `python -m app.scheduler`)도 그대로 쓸 수 있다.

수동 실행: `python -m app.cli launch | stage S1400|S1430|S1500|S1510|S1520 | grade | reevaluate | study | backtest --exit open|close | verify`

## 합성 데이터 데모 (네트워크 없이 파이프라인 검증)
```bash
DATABASE_URL=sqlite:///./data/synthetic_demo.db uvicorn app.api.main:app --port 8000   # 이미 생성된 데모 DB
# 새로 만들려면: DATABASE_URL=sqlite:///./data/demo2.db ALLOW_SYNTHETIC=1 python -m app.cli demo-synthetic --days 330 --replay 130
python -m pytest -q tests
```
합성 데이터는 `SYNTHETIC(TEST ONLY)` 로 표시되며 화면에 경고 배지가 붙는다. 실데이터 DB 와 절대 섞지 말 것.
합성 데이터에는 검증용으로 "14:00 이후 모멘텀 → 다음날 갭", "외국인 5일 순매수 → 다음날 수익" 두 신호만 심어 두었다.
특징 검증 화면이 이 둘을 찾아내고 나머지를 "효과 없음"으로 판정하는지가 파이프라인 검증 포인트다.

## 핵심 보장 장치
| 요구 | 구현 |
|---|---|
| 시점별 Look-ahead 금지 | 모든 행에 `available_at`; 단계 `as_of` 이전 값만 접근; 15:20 단계는 15:18 스냅샷까지만, 당일 일봉(16:00)·수급(18:00) 미사용; 테스트로 이후 데이터 유무와 무관하게 피처 동일함을 검증 |
| 데이터 없음 | 실패·미설정은 값 생성 없이 UNAVAILABLE 로그, 피처 결측, 완전성↓ → Confidence↓, 화면에 "데이터 없음" |
| 예측 사후 수정 불가 | 예측·실행·채점 테이블 UPDATE/DELETE 트리거 차단 + SHA-256 해시 체인(`/api/integrity`) + 같은 날·단계 재실행 시 기존 유지 + 실데이터 과거 시점 생성 금지 |
| 비용 반영 | 왕복 0.41% (수수료 1.5bp×2, 거래세 18bp, 슬리피지 10bp×2) — 기대수익·채점·백테스트·게이트 모두 차감 후 |
| 후보 없음 판단 | OOS 그림자 포트폴리오가 비용 차감 후 유의한 양의 수익(t≥1.5, ≥40일)이 아니면 모든 Confidence Low → "오늘은 매수 후보 없음" |
| 특징 검증 | 특징별 IC 를 학습/OOS 로 분리해 판정. 미리 유효하다고 가정하지 않음 |
| 과최적화 방지 | 새 홀드아웃에서만 비교, 블록 부트스트랩 95% 하한 > 0 일 때만 채택, 모든 후보 기록 |

## 알려진 한계
- 과거 장중 데이터가 없으면 백테스트·초기 모델은 장중 피처 없이 학습된다. 라이브 스냅샷이 쌓이면 재평가에서 장중 피처가 검증·반영된다.
- 프로그램 매매, 미국 선물 장중, 뉴스 sentiment, 컨센서스는 현재 "데이터 없음".
- 업종 매핑은 현재 구성종목 기준. 휴장일 목록은 매년 갱신 필요 (`KRX_HOLIDAYS` 로 추가 가능).
- KRX/KIS/DART/ECOS/yfinance 수집기는 공식 스펙 기준으로 작성했으나, 개발 환경 네트워크 정책으로 해당 호스트가 차단되어 실호출 검증은 아직 못 함.
