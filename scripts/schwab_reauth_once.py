#!/usr/bin/env python3
"""Back-compat: auto-catch HTTPS callback and sync GCP. No paste file."""
from pathlib import Path
import runpy
import sys

runpy.run_path(str(Path(__file__).with_name("schwab_reauth.py")), run_name="__main__")
sys.exit(0)
