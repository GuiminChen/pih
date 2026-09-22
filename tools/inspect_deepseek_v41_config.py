"""Inspect V4.1 Flash metadata without importing native code or model code.

This is an onboarding diagnostic, not a weight verifier or executable profile.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from collections.abc import Sequence

MODEL_ID = "deepseek-ai/DeepSeek-V4.1-Flash"
REFERENCE_REVISION = "517ef625df97ec57aadc91b67506a57c20fdc5bb"
MAX_CONFIG_BYTES = 1 << 20


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate config key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON value: {value}")


def inspect_config(payload: bytes) -> dict[str, object]:
    if type(payload) is not bytes or not 0 < len(payload) <= MAX_CONFIG_BYTES:
        raise ValueError("config must be nonempty bytes no larger than 1 MiB")
    try:
        config = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique,
                            parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid V4.1 configuration JSON") from error
    if type(config) is not dict or config.get("model_type") != "deepseek_v41":
        raise ValueError("expected deepseek_v41; V4 Flash-0731 is a different model")
    if config.get("architectures") != ["DeepseekV41ForCausalLM"]:
        raise ValueError("unexpected V4.1 architecture")
    text = config.get("text_config")
    if type(text) is not dict or text.get("model_type") != "deepseek_v41_text":
        raise ValueError("V4.1 requires nested text_config")
    # Deliberately only an identity/shape sanity check. No tokenizer, weight,
    # attention semantics, quality, capacity or source authenticity is certified.
    expected = {
        "num_hidden_layers": 40, "hidden_size": 5120,
        "vocab_size": 129280, "n_routed_experts": 384,
        "num_experts_per_tok": 6, "max_position_embeddings": 1048576,
    }
    for key, value in expected.items():
        if type(text.get(key)) is not int or text[key] != value:
            raise ValueError(f"V4.1 Flash shape mismatch: text_config.{key}")
    if type(config.get("vision_config")) is not dict:
        raise ValueError("V4.1 Flash requires vision_config metadata")
    return {
        "schema": "pih.deepseek-v41-config-inspection.v1",
        "model_id": MODEL_ID,
        "reference_revision": REFERENCE_REVISION,
        "source_revision_verified": False,
        "config_sha256": sha256(payload).hexdigest(),
        "config_bytes": len(payload),
        "text_shapes": expected,
        "support_status": "unsupported",
        "native_runtime_status": "source_implemented_unqualified",
        "optional_reference_backend_status": "implemented_unqualified",
        "verification_scope": "config_identity_and_selected_shapes_only",
        "required_runtime_features": ["ced", "csa2", "engram", "vision_encoder"],
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(arguments)
    try:
        with args.config.open("rb") as stream:
            payload = stream.read(MAX_CONFIG_BYTES + 1)
        result = inspect_config(payload)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
