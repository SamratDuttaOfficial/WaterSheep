#!/usr/bin/env bash
# Asks the trained model. Arguments pass through to predict.py.
set -e
cd "$(dirname "$0")"
if [ -x ./.venv/Scripts/python.exe ]; then
  exec ./.venv/Scripts/python.exe predict.py "$@"
fi
./setup.sh
exec ./.venv/bin/python predict.py "$@"
