import unittest
from pathlib import Path


class BoundaryTest(unittest.TestCase):
    def test_no_prior_pipeline_imports(self):
        root = Path(__file__).resolve().parents[1] / "corat"
        forbidden = ("import codexuw", "from codexuw", "import uwos", "from uwos", "import groki_eq", "from groki_eq", "import groko", "from groko")
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, "{} contains {}".format(path, token))

    def test_no_broker_order_endpoint(self):
        root = Path(__file__).resolve().parents[1] / "corat"
        forbidden = ("/orders", "submit_order(", "cancel_order(", "replace_order(")
        for path in root.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, text, "{} contains {}".format(path, token))

