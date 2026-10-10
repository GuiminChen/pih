"""Runtime configuration bounds, independent of model weights/GPU."""
import importlib.util
from pathlib import Path
import pytest
import json

spec = importlib.util.spec_from_file_location("batch_deploy", Path(__file__).parents[2] / "deploy/native.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)

@pytest.mark.parametrize("sequences", [1, 2, 4, 32])
def test_batch_limits_allow_real_multisequence_capacity(sequences):
    assert deploy.batch_limits(sequences, 4096, 64) == {
        "maximum_batch_sequences": sequences,
        "maximum_batch_tokens": 4096,
        "maximum_pending_requests": 64,
    }

@pytest.mark.parametrize("values", [(0,4096,64),(33,4096,64),(4,3,64),
    (4,4097,64),(4,4096,0),(4,4096,4097),(True,4096,64)])
def test_batch_limits_reject_invalid_or_unreserved_capacity(values):
    with pytest.raises(ValueError):
        deploy.batch_limits(*values)


def test_sealed_configuration_and_capability_match_batch_contract(tmp_path, monkeypatch):
    source, model, bundle = [tmp_path / name for name in ("source", "model", "bundle")]
    for directory in (source, model, bundle):
        directory.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (model / name).write_bytes(b"fixture")
    monkeypatch.setattr(deploy, "ROOT", source)
    def fixture_digest(path):
        return {"model.safetensors": "a" * 64, "config.json": "b" * 64,
                "tokenizer.json": deploy.QWEN_TOKENIZER_SHA256}.get(path.name, "c" * 64)
    monkeypatch.setattr(deploy, "digest", fixture_digest)
    # Publication permissions are a separate Linux-only security contract.
    def fixture_publish(path, encoded):
        path.write_bytes(encoded)
        return path
    monkeypatch.setattr(deploy, "publish_lock", fixture_publish)
    lock = deploy.seal(bundle, model, "bf16", "rtx4090d", 4096, tmp_path / "build",
                       600000, "a" * 64, "b" * 64, 4, 256, 8)
    document = json.loads(lock.read_text())
    assert document["engine"]["configuration"]["maximum_batch_sequences"] == 4
    assert document["engine"]["configuration"]["maximum_batch_tokens"] == 256
    assert document["engine"]["configuration"]["maximum_pending_requests"] == 8
    model_capability = next(c for c in document["capabilities"] if c["provider_id"] == "pih.model.qwen3")
    assert model_capability["contract_id"] == "pih.inference.text.v3"
    assert model_capability["threading_model"] == 3
