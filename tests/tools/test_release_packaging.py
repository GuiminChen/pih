"""Distribution boundary tests; synthetic files are never executed or released."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("package_release", ROOT / "tools/package_release.py")
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def test_source_policy_preserves_native_sources_excludes_reference_and_private_history():
    names = package.source_files()
    assert "src/worker/main.cpp" in names
    assert "plugins/model-qwen3/qwen_cubin_identity.h" in names
    assert "plugins/common/DEEPSEEK_LICENSE" in names
    assert not any(p.startswith(("out/", "python/pih/", "plugins/model-deepseek-v41-reference/",
                                 "docs/superpowers/", "docs/implementation/")) for p in names)


def test_existing_archive_is_preserved(tmp_path):
    archive = tmp_path / "existing.tar.gz"
    archive.write_bytes(b"owned content")
    with pytest.raises(ValueError, match="new"):
        package.package(tmp_path, [], archive, "source")
    assert archive.read_bytes() == b"owned content"


def test_extra_native_member_is_rejected(tmp_path, monkeypatch):
    (tmp_path / "NATIVE-INSTALL.json").write_text(json.dumps(
        {"schema": "pih.native-install.v1", "files": []}))
    (tmp_path / "lib").mkdir()
    (tmp_path / "lib/old-runtime.py").write_text("# nonrelease fixture")
    output = tmp_path.parent / (tmp_path.name + ".tar.gz")
    monkeypatch.setattr(sys, "argv", ["package", "native", "--root", str(tmp_path), "--output", str(output)])
    with pytest.raises(ValueError, match="extra files"):
        package.main()
    assert not output.exists()
