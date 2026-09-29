#!/usr/bin/env bash
# Creates .venv (and a private Python in .python if none is found). Linux and macOS.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PYDIR="$ROOT/.python"
VENV="$ROOT/.venv"
VPY="$VENV/bin/python"
PYVER="3.12.7"
PBSTAG="20241016"

[ -x "$VPY" ] && exit 0

usable() {
  command -v "$1" >/dev/null 2>&1 || [ -x "$1" ] || return 1
  "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

PY=""
for cand in "$PYDIR/bin/python3" python3 python; do
  if usable "$cand"; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-aarch64|Linux-arm64) TRIPLE="aarch64-unknown-linux-gnu" ;;
    Linux-*)                   TRIPLE="x86_64-unknown-linux-gnu" ;;
    Darwin-arm64)              TRIPLE="aarch64-apple-darwin" ;;
    Darwin-*)                  TRIPLE="x86_64-apple-darwin" ;;
    *) echo "[setup] unsupported platform; install Python 3.10+ and re-run"; exit 1 ;;
  esac
  URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PBSTAG/cpython-$PYVER+$PBSTAG-$TRIPLE-install_only.tar.gz"
  echo "[setup] no Python 3.10+ found - downloading a private CPython into $PYDIR"
  TGZ="$(mktemp -t ws-cpython.XXXXXX).tar.gz"
  curl -fSL --retry 3 -o "$TGZ" "$URL"
  rm -rf "$PYDIR" "${PYDIR}_tmp" && mkdir -p "${PYDIR}_tmp"
  tar -xzf "$TGZ" -C "${PYDIR}_tmp" && mv "${PYDIR}_tmp/python" "$PYDIR"
  rm -rf "${PYDIR}_tmp" "$TGZ"
  PY="$PYDIR/bin/python3"
fi

echo "[setup] creating the virtual environment in $VENV (python: $PY)"
"$PY" -m venv --system-site-packages "$VENV"
"$VPY" -m pip install -q --disable-pip-version-check --upgrade pip wheel
echo "[setup] ready. the pipeline installs the heavier packages itself when first needed."
