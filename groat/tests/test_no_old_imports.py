import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "groat"

NEEDLES = (
    "swing" + "desk",
    "codex" + "uw",
    "uw" + "os",
    "options" + "_agent",
    "groki_eq",
)


def _banned_alias(name: str) -> bool:
    lowered = (name or "").lower()
    if lowered == "groki" or lowered.startswith("groki."):
        return True
    if lowered == "groko" or lowered.startswith("groko."):
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

    def test_docs_have_no_old_execute_paths(self):
        targets = [ROOT / "AGENTS.md", ROOT / "configs" / "universe.txt"]
        targets.extend((ROOT / "skills").rglob("*.md"))
        hits = []
        for path in targets:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            for needle in NEEDLES:
                if needle in text:
                    hits.append("%s contains %s" % (path.name, needle))
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
