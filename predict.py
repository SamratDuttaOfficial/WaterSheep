#!/usr/bin/env python
"""Ask a trained WaterSheep model for decisions. Same options as the `watersheep` command.

  python predict.py --request request.json
  python predict.py --file requests.jsonl
  python predict.py --serve
  python predict.py --question "Which team?" --options billing,shipping --state "..."
"""
import sys

from watersheep.cli import main

if __name__ == "__main__":
    sys.exit(main(pin=True))
