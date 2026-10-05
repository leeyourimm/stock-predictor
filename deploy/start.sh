#!/bin/bash
# 서버 시작: DB 준비 → (처음 한 번) 과거 데이터 수집·초기 학습을 백그라운드로 → 자동 스케줄러 → 웹 API
set -u
cd /app/backend
DATA_DIR="${DATA_DIR:-/var/data}"
mkdir -p "$DATA_DIR"
export DATABASE_URL="${DATABASE_URL:-sqlite:///$DATA_DIR/predictor.db}"
python -m app.cli init-db

if [ ! -f "$DATA_DIR/.bootstrapped" ]; then
  (
    START=$(date -d '-400 days' +%Y-%m-%d); END=$(date -d '-1 day' +%Y-%m-%d)
    echo "[bootstrap] 과거 데이터 수집 $START ~ $END"
    python -m app.cli collect --start "$START" --end "$END" && \
    python -m app.cli bootstrap && touch "$DATA_DIR/.bootstrapped" && echo "[bootstrap] 완료"
  ) >> "$DATA_DIR/bootstrap.log" 2>&1 &
fi

python -m app.scheduler >> "$DATA_DIR/scheduler.log" 2>&1 &
exec python -m uvicorn app.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
