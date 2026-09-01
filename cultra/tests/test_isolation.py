import tempfile
import textwrap
import unittest
from pathlib import Path

from cultra.pipeline import PROJECT_ROOT, scan_clean_room


class CleanRoomIsolationTests(unittest.TestCase):
    def test_actual_package_passes_clean_room_scan(self):
        violations = scan_clean_room(PROJECT_ROOT / "cultra", project_root=PROJECT_ROOT)
        self.assertEqual(
            violations,
            (),
            "\n".join("%s:%d %s" % (v.path, v.line, v.detail) for v in violations),
        )

    def test_rejects_external_import_cross_pipeline_path_and_order_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "cultra"
            package.mkdir()
            (package / "bad.py").write_text(
                textwrap.dedent(
                    """
                    import requests
                    SOURCE = "/Users/anuppamvi/tradedesk/sibling-pipeline/cache"
                    def mutate(client):
                        return client.submit_order({})
                    """
                )
            )
            violations = scan_clean_room(package, project_root=root)
            rules = {item.rule for item in violations}
            self.assertIn("stdlib-only", rules)
            self.assertIn("path-isolation", rules)
            self.assertIn("manual-only", rules)

    def test_network_client_import_is_allowed_only_in_gateway(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "cultra"
            package.mkdir()
            (package / "worker.py").write_text("from urllib.request import urlopen\n")
            (package / "gateway.py").write_text("from urllib.request import urlopen\n")
            violations = scan_clean_room(package, project_root=root)
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0].path, "worker.py")
            self.assertEqual(violations[0].rule, "network-boundary")


if __name__ == "__main__":
    unittest.main()

