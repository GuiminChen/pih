"""Orchestration-only cases; no native executable or model is run."""

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("qwen_preparation_deploy", ROOT / "deploy/native.py")
assert SPEC and SPEC.loader
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    monkeypatch.setattr(DEPLOY.platform, "system", lambda: "Linux")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "model.safetensors"
    source.write_bytes(b"not executed")
    expectation = tmp_path / "expectation.json"
    expectation.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    events = []

    def select(*args):
        events.append("select")
        return Path("/fake/pih-qwen-int4-convert")

    def metadata(args):
        assert args.source_dir == source_dir
        assert args.model_dir == output_dir
        assert args.config_sha256 == "a" * 64
        events.append("metadata")

    monkeypatch.setattr(DEPLOY, "qwen_artifact_tool", select)
    monkeypatch.setattr(DEPLOY, "prepare_qwen_metadata", metadata)
    monkeypatch.setattr(DEPLOY, "run", lambda command: events.append("convert"))
    argv = ["convert-int4", "--source", str(source), "--expectation", str(expectation),
            "--output", str(output_dir / "model.xing-int4")]
    return argv, events


def test_combined_preparation_orders_metadata_before_conversion(conversion):
    argv, events = conversion
    assert DEPLOY.main(argv + ["--prepare-metadata", "--config-sha256", "a" * 64]) == 0
    assert events == ["select", "metadata", "convert"]


@pytest.mark.parametrize("options", [
    ["--prepare-metadata"], ["--config-sha256", "a" * 64],
    ["--prepare-metadata", "--config-sha256", "0" * 64],
])
def test_bad_metadata_options_fail_before_build(conversion, options):
    argv, events = conversion
    with pytest.raises(SystemExit) as failure:
        DEPLOY.main(argv + options)
    assert failure.value.code == 2
    assert events == []


def test_metadata_failure_prevents_conversion(conversion, monkeypatch):
    argv, events = conversion

    def reject(args):
        raise ValueError("metadata digest mismatch")

    monkeypatch.setattr(DEPLOY, "prepare_qwen_metadata", reject)
    with pytest.raises(SystemExit) as failure:
        DEPLOY.main(argv + ["--prepare-metadata", "--config-sha256", "a" * 64])
    assert failure.value.code == 2
    assert events == ["select"]


def test_weight_only_conversion_does_not_prepare_metadata(conversion):
    argv, events = conversion
    assert DEPLOY.main(argv) == 0
    assert events == ["select", "convert"]


def test_tokenizer_pin_matches_native_source():
    # Static drift check only, not tokenizer equivalence or provenance proof.
    pin = DEPLOY.QWEN_TOKENIZER_SHA256
    sources = (ROOT / "plugins/model-qwen3").glob("tokenizer.*")
    assert any(pin in path.read_text(encoding="utf-8") for path in sources)


def test_configure_resets_cached_options_before_explicit_overrides(monkeypatch):
    captured = []
    monkeypatch.setattr(DEPLOY.subprocess, "run", lambda command, **kwargs: captured.append(command))
    DEPLOY.run(["cmake", "-S", "source", "-B", "build", "-DPIH_BUILD_TOKENIZER_TOOLS=ON"])
    command = captured[0]
    assert "-DPIH_BUILD_NATIVE_CONTRACT_TESTS=OFF" in command
    assert "-DBUILD_TESTING=OFF" in command
    assert command.index("-DPIH_BUILD_TOKENIZER_TOOLS=OFF") < command.index("-DPIH_BUILD_TOKENIZER_TOOLS=ON")


def test_qwen_bundle_explicitly_enables_requested_artifact_targets(monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(DEPLOY.platform, "system", lambda: "Linux")
    monkeypatch.setattr(DEPLOY, "run", commands.append)
    DEPLOY.build(tmp_path / "build", tmp_path / "bundle", 2, "rtx4090d")
    assert "-DPIH_BUILD_QWEN_ARTIFACT_TOOLS=ON" in commands[0]
    assert "-DPIH_ENABLED_PLUGINS=pih.backend.nvidia-cuda;pih.execution.default;pih.model.qwen3;pih.platform.linux;pih.storage.verified-artifact;pih.surface.text-http" in commands[0]
    assert "pih_plugin_execution_default" in commands[1]
    assert "pih_plugin_backend_nvidia_cuda" in commands[1]
    assert "pih_plugin_platform_linux" in commands[1]


def test_qwen_lock_hash_binds_required_execution_provider(tmp_path, monkeypatch):
    bundle, model, build_dir = (tmp_path / name for name in ("bundle", "model", "build"))
    (bundle / "lib").mkdir(parents=True)
    model.mkdir()
    for name in ("config.json", "tokenizer.json", "model.safetensors"):
        (model / name).write_bytes(b"fixture")
    for name in ("pih_plugin_execution_default.so", "pih_plugin_model_qwen3.so",
                 "pih_plugin_backend_nvidia_cuda.so", "pih_plugin_platform_linux.so",
                 "pih_plugin_surface_text_http.so", "pih_kernels_qwen3_sm89.so",
                 "pih_plugin_storage_verified_artifact.so"):
        (bundle / "lib" / name).write_bytes(name.encode())
    # Synthetic orchestration fixture only; no checkpoint or model execution.
    fixture_digest = DEPLOY.digest(model / "model.safetensors")
    monkeypatch.setattr(DEPLOY, "QWEN_TOKENIZER_SHA256", fixture_digest)
    monkeypatch.setattr(DEPLOY.os, "fchmod", lambda *args: None, raising=False)
    lock_path = DEPLOY.seal(bundle, model, "bf16", "rtx4090d", 2048, build_dir, 600000,
                            fixture_digest, fixture_digest)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    provider = next(item for item in lock["plugins"] if item["plugin_id"] == "pih.execution.default")
    assert provider["entrypoint"] == "lib/pih_plugin_execution_default.so"
    assert provider["entrypoint_sha256_hex"] == DEPLOY.digest(bundle / provider["entrypoint"])
    assert next(item for item in lock["capabilities"] if item["capability_id"] == "execution.default.v1") == {
        "capability_id": "execution.default.v1", "provider_id": "pih.execution.default",
        "contract_id": "pih.execution.default.v1", "threading_model": 3, "scope": 2, "cardinality": 1,
    }
    assert next(item for item in lock["capabilities"] if item["capability_id"] == "execution.controller.v1") == {
        "capability_id": "execution.controller.v1", "provider_id": "pih.execution.default",
        "contract_id": "pih.execution.controller.v1", "threading_model": 3,
        "scope": 2, "cardinality": 1,
    }
    assert lock["plugins"][0]["plugin_id"] == "pih.backend.nvidia-cuda"
    for name in ("device.cuda-memory.v1", "device.cuda-runtime.v1",
                 "device.cuda-resources.v1", "device.cuda-async.v1"):
        capability = next(item for item in lock["capabilities"] if item["capability_id"] == name)
        assert capability["provider_id"] == "pih.backend.nvidia-cuda"
        assert capability["scope"] == 1
    assert any(item["plugin_id"] == "pih.platform.linux" for item in lock["plugins"])
