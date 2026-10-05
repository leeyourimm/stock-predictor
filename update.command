#!/bin/bash
# 업데이트: GitHub 에서 Download ZIP 으로 새 버전을 받은 뒤 실행. 키 파일(env)·설치 패키지·데이터는 유지, 바탕화면/코드/클로드코드 폴더에 설치.
D="$HOME/Desktop/코드/클로드코드"; DL="$HOME/Downloads"; mkdir -p "$D" && cd "$D" && (
set -e
OLD=""; for p in "$D/stock-predictor-main" "$D/stock-predictor" "$DL/stock-predictor-main"; do if [ -d "$p" ]; then OLD="$p"; break; fi; done
NEW=""
if [ -d "$DL/stock-predictor-main 2" ]; then NEW="$DL/stock-predictor-main 2"
elif [ -f "$DL/stock-predictor-main.zip" ]; then NEW="$DL/stock-predictor-main.zip"
elif [ -d "$DL/stock-predictor-main" ] && [ "$OLD" != "$DL/stock-predictor-main" ]; then NEW="$DL/stock-predictor-main"; fi
if [ -z "$NEW" ]; then
  echo "새로 받은 파일이 없어 기존 버전을 이 폴더로 옮기기만 합니다."
  if [ "$OLD" != "$D/stock-predictor-main" ]; then mv "$OLD" "$D/stock-predictor-main"; fi
  exit 0
fi
if [ -n "$OLD" ]; then mv "$OLD" "$D/sp-old"; fi
case "$NEW" in
  *.zip) unzip -q -o "$NEW" -d "$D" && rm -f "$NEW" ;;
  *) mv "$NEW" "$D/stock-predictor-main" ;;
esac
if [ -d sp-old ]; then
  if [ -f sp-old/env ]; then mv sp-old/env stock-predictor-main/; fi
  if [ -d sp-old/backend/.venv ]; then mv sp-old/backend/.venv stock-predictor-main/backend/; fi
  cp -R sp-old/backend/data/. stock-predictor-main/backend/data/ 2>/dev/null || true
  rm -rf sp-old
fi
echo "업데이트 완료"
) && cd "$D/stock-predictor-main" && bash start_real.command
