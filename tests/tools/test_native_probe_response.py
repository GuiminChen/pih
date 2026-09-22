"""Response-envelope cases only; never contact a service or execute a model."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "native_probe_deploy", Path(__file__).resolve().parents[2] / "deploy/native.py")
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)


def response():
    return {"model": "Qwen/Qwen3-0.6B", "choices": [{"index": 0,
            "finish_reason": "stop", "message": {"role": "assistant", "content": "Hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}


def accepted(value):
    return DEPLOY.probe_response_is_complete(json.dumps(value), "Qwen/Qwen3-0.6B", 4)


def test_complete_response():
    assert accepted(response())


def test_v41_response_requires_exact_model_identity():
    document = response()
    document["model"] = "deepseek-ai/DeepSeek-V4.1-Flash"
    encoded = json.dumps(document)
    assert DEPLOY.probe_response_is_complete(
        encoded, "deepseek-ai/DeepSeek-V4.1-Flash", 4)
    assert not DEPLOY.probe_response_is_complete(encoded, "Qwen/Qwen3-0.6B", 4)


@pytest.mark.parametrize("field,value", [("index", True), ("index", 1),
    ("finish_reason", None), ("finish_reason", "error")])
def test_invalid_choice(field, value):
    document = response()
    document["choices"][0][field] = value
    assert not accepted(document)


@pytest.mark.parametrize("field,value", [("prompt_tokens", 0), ("completion_tokens", True),
    ("completion_tokens", 5), ("total_tokens", 99), ("total_tokens", 5.0)])
def test_invalid_usage(field, value):
    document = response()
    document["usage"][field] = value
    assert not accepted(document)


def test_error_or_multiple_choices():
    document = response()
    document["error"] = {"message": "failed"}
    assert not accepted(document)
    del document["error"]
    document["choices"] *= 2
    assert not accepted(document)


@pytest.mark.parametrize("suffix", ['"extra":NaN', '"extra":1e999', '"model":"Qwen/Qwen3-0.6B"'])
def test_duplicate_or_nonfinite_json(suffix):
    encoded = json.dumps(response())[:-1] + "," + suffix + "}"
    assert not DEPLOY.probe_response_is_complete(encoded, "Qwen/Qwen3-0.6B", 4)
