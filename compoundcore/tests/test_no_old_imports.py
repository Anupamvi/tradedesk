import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "compoundcore"

NEEDLES = (
    "codex" + "uw",
    "uw" + "os",
    "options" + "_agent",
    "groki_eq",
)


def _banned_alias(name: str) -> bool:
    lowered = (name or "").lower()
    if lowered in ("groat", "groko", "wheelo", "xhigh", "groki") or any(
        lowered.startswith(p) for p in ("groat.", "groko.", "wheelo.", "xhigh.", "groki.")
    ):
        return True
    return any(n in lowered for n in NEEDLES)


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
                        if _banned_alias(alias.name):
                            hits.append("%s import %s" % (path, alias.name))
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if _banned_alias(mod):
                        hits.append("%s from %s" % (path, mod))
        self.assertEqual(hits, [])
