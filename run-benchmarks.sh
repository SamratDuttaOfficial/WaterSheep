#!/usr/bin/env bash
# Benchmarks the newest export (or --model). Arguments pass through to benchmark.py.
set -e
cd "$(dirname "$0")"
if [ -x ./.venv/Scripts/python.exe ]; then
  exec ./.venv/Scripts/python.exe benchmark.py "$@"
fi
./setup.sh
exec ./.venv/bin/python benchmark.py "$@"
