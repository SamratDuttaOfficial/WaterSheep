#!/usr/bin/env bash
# Asks the trained model. Arguments pass through to watersheep.cli.
set -e
cd "$(dirname "$0")/.."
if [ -x ./.venv/Scripts/python.exe ]; then
  exec ./.venv/Scripts/python.exe -m watersheep.cli "$@"
fi
scripts/setup.sh
exec ./.venv/bin/python -m watersheep.cli "$@"
