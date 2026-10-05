# Python 3.10+ 찾기 → backend/.venv 가상환경 만들고 패키지 설치. 성공 시 $PY 설정.
# 일부 맥 Python(Homebrew 등)은 venv 의 pip 설치(ensurepip)가 실패하므로 여러 Python 을 차례로 시도하고,
# 모두 실패하면 pip 없이 venv 를 만든 뒤 get-pip.py 로 pip 를 설치한다.
CANDS=()
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  for p in /Library/Frameworks/Python.framework/Versions/*/bin/$c "$(command -v $c 2>/dev/null)" /opt/homebrew/bin/$c /usr/local/bin/$c; do
    [ -x "$p" ] || continue
    "$p" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null || continue
    case " ${CANDS[*]} " in *" $p "*) ;; *) CANDS+=("$p") ;; esac
  done
done
if [ ${#CANDS[@]} -eq 0 ]; then
  echo "Python 3.10 이상이 필요합니다. 열리는 페이지에서 macOS용 Python 을 설치한 뒤 다시 실행하세요."
  open https://www.python.org/downloads/macos/ 2>/dev/null
  return 1
fi
if [ ! -f .venv/.installed ]; then
  echo "처음 실행: 가상환경을 만들고 필요한 패키지를 설치합니다 (몇 분 걸릴 수 있음)…"
  rm -rf .venv
  for p in "${CANDS[@]}"; do
    if "$p" -m venv .venv >/dev/null 2>&1 && .venv/bin/python -m pip --version >/dev/null 2>&1; then
      echo "사용하는 Python: $p"; break
    fi
    rm -rf .venv
  done
  if [ ! -x .venv/bin/python ]; then
    echo "기본 방식 실패 → pip 를 직접 설치합니다 (${CANDS[0]})"
    "${CANDS[0]}" -m venv --without-pip .venv || { echo "가상환경 생성 실패"; return 1; }
    curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && .venv/bin/python /tmp/get-pip.py -q \
      || { echo "pip 설치 실패 — 이 화면을 Claude 에게 보여주세요"; rm -rf .venv; return 1; }
  fi
  .venv/bin/python -m pip install -q --upgrade pip
  .venv/bin/python -m pip install -q -r requirements.txt || { echo "패키지 설치 실패 — 이 화면을 Claude 에게 보여주세요"; return 1; }
  touch .venv/.installed
fi
PY="$(pwd)/.venv/bin/python"
