# Python 3.10+ 찾기 → backend/.venv 가상환경 만들고 패키지 설치. 성공 시 $PY 설정.
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  for p in "$(command -v $c 2>/dev/null)" /Library/Frameworks/Python.framework/Versions/*/bin/$c /opt/homebrew/bin/$c /usr/local/bin/$c; do
    [ -x "$p" ] || continue
    if "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then BASEPY="$p"; break 2; fi
  done
done
if [ -z "$BASEPY" ]; then
  echo "Python 3.10 이상이 필요합니다. 열리는 페이지에서 macOS용 Python 을 설치한 뒤 다시 실행하세요."
  open https://www.python.org/downloads/macos/
  return 1
fi
if [ ! -x .venv/bin/python ]; then
  echo "처음 실행: 필요한 패키지를 설치합니다 (몇 분 걸릴 수 있음)…"
  "$BASEPY" -m venv .venv || return 1
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "패키지 설치 실패"; return 1; }
fi
PY="$(pwd)/.venv/bin/python"
