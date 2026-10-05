#!/bin/bash
# KR Overnight Predictor — 실데이터 실행 (backend/.env 의 키 사용)
# 처음 실행 시: DB 생성 → 과거 1년 데이터 수집(수십 분 걸릴 수 있음) → 초기 모델 학습 → 서버 + 자동 스케줄러 시작
# 사용법: 터미널에서  bash start_real.command
cd "$(dirname "$0")/backend" || exit 1
source ../scripts/_setup_python.sh || exit 1
unset DATABASE_URL   # .env 의 DATABASE_URL 사용
"$PY" -m app.cli init-db || exit 1
if [ ! -f data/.bootstrapped ]; then
  START=$(date -v-400d +%Y-%m-%d); END=$(date -v-1d +%Y-%m-%d)
  echo "과거 데이터 수집: $START ~ $END (처음 한 번만, 시간이 걸립니다)"
  "$PY" -m app.cli collect --start "$START" --end "$END" || { echo "수집 실패 — 위 메시지를 Claude 에게 보여주세요"; exit 1; }
  "$PY" -m app.cli bootstrap || { echo "초기 학습 실패 — 위 메시지를 Claude 에게 보여주세요"; exit 1; }
  touch data/.bootstrapped
fi
echo "자동 스케줄러(14:00~15:20 분석, 18:30 채점)를 백그라운드에서 시작합니다. 로그: backend/data/scheduler.log"
"$PY" -m app.scheduler > data/scheduler.log 2>&1 &
SCHED=$!
trap "kill $SCHED 2>/dev/null" EXIT
(sleep 3; open http://localhost:8000) &
echo "서버: http://localhost:8000  (끄려면 Control + C — 스케줄러도 함께 꺼집니다)"
"$PY" -m uvicorn app.api.main:app --port 8000
