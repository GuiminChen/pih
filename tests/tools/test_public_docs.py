from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "verify_public_docs", Path(__file__).resolve().parents[2] / "tools/verify_public_docs.py"
)
assert SPEC and SPEC.loader
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class PublicDocsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.docs = self.root / "docs"
        self.docs.mkdir()
        for name in ("index", "architecture", "development", "plugins", "deployment", "support"):
            (self.docs / f"{name}.md").write_text(
                f"English | [中文]({name}.zh.md)\n", encoding="utf-8"
            )
            (self.docs / f"{name}.zh.md").write_text(
                f"[English]({name}.md) | 中文\n", encoding="utf-8"
            )

    def check(self):
        with patch.object(CHECKER, "ROOT", self.root), patch.object(CHECKER, "DOCS", self.docs):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return CHECKER.main()

    def test_complete_guides(self):
        self.assertEqual(self.check(), 0)

    def test_missing_required_guide_fails(self):
        (self.docs / "support.md").unlink()
        (self.docs / "support.zh.md").unlink()
        self.assertEqual(self.check(), 1)

    def test_missing_language_pair_fails(self):
        (self.docs / "development.zh.md").unlink()
        self.assertEqual(self.check(), 1)

    def test_broken_relative_link_fails(self):
        (self.root / "README.md").write_text("[start](docs/missing.md)", encoding="utf-8")
        self.assertEqual(self.check(), 1)

    def test_example_links_are_checked(self):
        example = self.root / "examples" / "external"
        example.mkdir(parents=True)
        (example / "README.md").write_text("[bad](absent.md)", encoding="utf-8")
        self.assertEqual(self.check(), 1)

    def test_build_output_does_not_affect_public_docs(self):
        generated = self.root / "out"
        generated.mkdir()
        (generated / "README.md").write_text("[bad](absent.md)", encoding="utf-8")
        self.assertEqual(self.check(), 0)


if __name__ == "__main__":
    unittest.main()
