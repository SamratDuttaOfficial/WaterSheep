#!/usr/bin/env bash
# Runs or resumes the pipeline. Arguments pass through to run.py.
set -e
cd "$(dirname "$0")"
./setup.sh
exec ./.venv/bin/python run.py "$@"
