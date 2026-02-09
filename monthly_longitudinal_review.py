#!/usr/bin/env python3
from __future__ import annotations

import sys

from trade_playbook_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(["monthly", *sys.argv[1:]]))
