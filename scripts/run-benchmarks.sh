#!/usr/bin/env bash
# Benchmarks the newest export (or --model). Arguments pass through to watersheep.benchmark.
set -e
cd "$(dirname "$0")/.."
if [ -x ./.venv/Scripts/python.exe ]; then
  exec ./.venv/Scripts/python.exe -m watersheep.benchmark "$@"
fi
scripts/setup.sh
exec ./.venv/bin/python -m watersheep.benchmark "$@"
