#!/bin/bash
# KR Overnight Predictor — 데모 실행 (합성 테스트 데이터, 인터넷/키 불필요)
# 사용법: 터미널에서  bash start_demo.command
cd "$(dirname "$0")/backend" || exit 1
source ../scripts/_setup_python.sh || exit 1
export DATABASE_URL="sqlite:///$(pwd)/data/demo.db"
if [ ! -f data/demo.db ]; then
  echo "데모용 합성 데이터를 처음 만듭니다 (10분 안팎 걸립니다)…"
  mkdir -p data
  ALLOW_SYNTHETIC=1 "$PY" -m app.cli demo-synthetic --days 240 --tickers 60 --replay 60 || { rm -f data/demo.db; exit 1; }
fi
echo ""
echo "데모 서버를 켭니다. 잠시 뒤 브라우저가 열립니다 → http://localhost:8000"
echo "끄려면 이 창에서 Control + C"
(sleep 3; open http://localhost:8000) &
"$PY" -m uvicorn app.api.main:app --port 8000
