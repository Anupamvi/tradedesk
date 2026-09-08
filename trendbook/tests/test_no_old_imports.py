import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "trendbook"

BANNED = (
    "groat",
    "groat1",
    "wheelo",
    "groko",
    "groki",
    "xhigh",
    "codexuw",
    "uwos",
    "swingdesk",
    "grok",
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

    def test_no_orders_or_uw(self):
        text = "\n".join(p.read_text(encoding="utf-8") for p in PKG.rglob("*.py"))
        self.assertNotIn("/v1/orders", text)
        self.assertNotIn("unusualwhales", text.lower())
        self.assertNotIn("trader/v1", text)

    def test_no_local_schwab_token_copy(self):
        self.assertFalse((ROOT / "tokens").exists())
        from trendbook.envload import schwab_credentials

        creds = schwab_credentials()
        if creds:
            path = Path(creds["token_path"]).resolve()
            self.assertFalse(str(path).startswith(str((ROOT / "tokens").resolve())))
            self.assertNotEqual(path.parent, ROOT)


if __name__ == "__main__":
    unittest.main()
