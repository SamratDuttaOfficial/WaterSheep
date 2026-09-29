#!/usr/bin/env bash
# Runs or resumes the pipeline. Arguments pass through to watersheep.run.
set -e
cd "$(dirname "$0")/.."
scripts/setup.sh
exec ./.venv/bin/python -m watersheep.run "$@"
