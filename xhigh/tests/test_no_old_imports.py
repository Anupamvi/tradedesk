import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "xhigh"

BANNED = (
    "groat",
    "groat1",
    "wheelo",
    "groko",
    "groki",
    "codexuw",
    "uwos",
    "swingdesk",
)


class TestNoOldImports(unittest.TestCase):
    def test_package_ast_imports(self):
        py_files = sorted(PKG.rglob("*.py"))
        self.assertTrue(py_files)
        hits = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = (alias.name or "").split(".")[0]
                        if root in BANNED:
                            hits.append("%s import %s" % (path, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".")[0]
                    if root in BANNED:
                        hits.append("%s from %s" % (path, node.module))
        self.assertEqual(hits, [])

    def test_no_positions_or_harvest_or_occ(self):
        self.assertFalse((PKG / "harvest.py").exists())
        text = "\n".join(p.read_text(encoding="utf-8") for p in PKG.rglob("*.py"))
        self.assertNotIn("def positions_all", text)
        self.assertNotIn("/v1/orders", text)
        self.assertNotIn("def pick_cc", text)
        self.assertNotIn("def parse_occ", text)
        self.assertNotIn("trader/v1", text)


if __name__ == "__main__":
    unittest.main()
