import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compoundcore.report import calc_markdown
from compoundcore.webcalc import calculator_html, write_calculator


ROOT = Path(__file__).resolve().parent.parent


class TestCliAndCalculator(unittest.TestCase):
    def test_markdown_contains_both_sleeves(self):
        md = calc_markdown(100000, weekly=250, monthly=1000, sleeve="both")
        self.assertIn("VOO", md)
        self.assertIn("Compound Core (default)", md)
        self.assertIn("Aggressive variant", md)
        self.assertIn("$48,000", md)
        self.assertIn("$45,000", md)
        self.assertIn("**Base**", md)

    def test_cli_amount_shortcut(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT)
        proc = subprocess.run(
            [sys.executable, "-m", "compoundcore", "100000", "--json"],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        data = json.loads(proc.stdout)
        self.assertEqual(data["amount"], 100000.0)
        voo = data["sleeves"]["default"]["allocation"]["rows"][0]
        self.assertEqual(voo["ticker"], "VOO")
        self.assertEqual(voo["dollars"], 48000.0)

    def test_html_embeds_live_weights(self):
        html = calculator_html()
        self.assertIn('"VOO": 0.48', html)
        self.assertIn('"SMH": 0.07', html)
        self.assertIn("Compound Core calculator", html)
        snap_start = html.index('type="application/json">') + len('type="application/json">')
        snap_end = html.index("</script>", snap_start)
        blob = html[snap_start:snap_end].strip()
        data = json.loads(blob)
        self.assertEqual(data["sleeves"]["default"]["weights"]["VXUS"], 0.20)
        self.assertEqual(data["sleeves"]["aggressive"]["weights"]["SMH"], 0.10)

    def test_write_calculator(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_calculator(Path(tmp) / "calculator.html")
            text = path.read_text(encoding="utf-8")
            self.assertIn('id="amount"', text)
            self.assertIn("Aggressive variant", calculator_html())
