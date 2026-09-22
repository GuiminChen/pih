"""Offline comparison of the native Qwen plain-text, non-thinking template."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

QWEN3_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


MAXIMUM_CONFIG_BYTES = 64 * 1024
MAXIMUM_TOKENIZER_BYTES = 12 * 1024 * 1024
QWEN3_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"

CASES = (
    ({"messages": [{"role": "user", "content": "hello"}], "tools": []}),
    ({"messages": [
        {"role": "system", "content": "Be exact."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "again"},
    ], "tools": []}),
    ({"messages": [
        {"role": "user", "content": "你好 😀\nsecond line\t\"quoted\""},
    ], "tools": []}),
    ({"messages": [{"role": "user", "content": "\n\nleading newline"}], "tools": []}),
    ({"messages": [{"role": "user", "content": "   "}], "tools": []}),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-renderer", type=Path, required=True,
                        help="trusted locally built pih-qwen-format executable")
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--tokenizer", type=Path,
                        help="optional pinned tokenizer.json to compare service token IDs too")
    arguments = parser.parse_args()
    if not 1 <= arguments.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 1 and 600")
    executable = arguments.native_renderer.resolve(strict=True)
    if not executable.is_file():
        parser.error("--native-renderer must name a regular executable file")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    payload = sys.stdin.buffer.read(MAXIMUM_CONFIG_BYTES + 1)
    if len(payload) > MAXIMUM_CONFIG_BYTES:
        raise ValueError("tokenizer config exceeds verifier ceiling")
    config = json.loads(payload)
    template = config["chat_template"]
    template_sha256 = hashlib.sha256(template.encode()).hexdigest()
    if template_sha256 != QWEN3_CHAT_TEMPLATE_SHA256:
        raise ValueError("chat template differs from pinned Qwen3 template")
    environment = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, autoescape=False
    )
    reference = environment.from_string(template)
    tokenizer_payload = None
    token_reference = None
    if arguments.tokenizer is not None:
        from tokenizers import Tokenizer
        with arguments.tokenizer.open("rb") as artifact:
            tokenizer_payload = artifact.read(MAXIMUM_TOKENIZER_BYTES + 1)
        if (len(tokenizer_payload) > MAXIMUM_TOKENIZER_BYTES or
                hashlib.sha256(tokenizer_payload).hexdigest() != QWEN3_TOKENIZER_SHA256):
            raise ValueError("tokenizer differs from pinned Qwen3 artifact")
        token_reference = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    with executable.open("rb") as binary:
        executable_digest = hashlib.file_digest(binary, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(prefix="pih-qwen-format-") as directory:
        request = Path(directory) / "request.json"
        command = [str(executable), str(request)]
        if tokenizer_payload is not None:
            tokenizer_path = Path(directory) / "tokenizer.json"
            tokenizer_path.write_bytes(tokenizer_payload)
            command += ["--tokens", str(tokenizer_path)]
        for index, case in enumerate(CASES):
            expected = reference.render(
                messages=case["messages"], tools=case["tools"],
                add_generation_prompt=True, enable_thinking=False,
            )
            request.write_bytes(json.dumps({"messages": case["messages"]},
                                           ensure_ascii=False).encode("utf-8"))
            with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
                completed = subprocess.run(
                    command, stdin=subprocess.DEVNULL,
                    stdout=output, stderr=errors, timeout=arguments.timeout_seconds,
                    check=False, shell=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(f"native renderer exited with status {completed.returncode}")
                output.seek(0)
                payload = output.read(8 * 1024 * 1024 + 1)
                if not payload or len(payload) > 8 * 1024 * 1024:
                    raise ValueError("native renderer output exceeds byte bound")
                pairs = json.loads(payload.decode("utf-8"), object_pairs_hook=lambda pairs: pairs)
                required = {"text", "tokens"} if token_reference is not None else {"text"}
                if (type(pairs) is not list or len(pairs) != len(required) or
                        any(type(pair) is not tuple or len(pair) != 2 for pair in pairs) or
                        {pair[0] for pair in pairs} != required):
                    raise ValueError("native renderer returned unexpected fields")
                result = dict(pairs)
                observed = result["text"]
                if type(observed) is not str:
                    raise ValueError("native renderer text must be a string")
                if token_reference is not None:
                    tokens = result["tokens"]
                    if (type(tokens) is not list or len(tokens) > 65536 or
                            any(type(token) is not int or not 0 <= token < 151936 for token in tokens)):
                        raise ValueError("native renderer returned invalid token IDs")
                    if tokens != token_reference.encode(expected, add_special_tokens=False).ids:
                        raise AssertionError(f"native plain-chat token parity failed at case {index}")
            if observed != expected:
                raise AssertionError(f"native plain-chat parity failed at case {index}")
    with executable.open("rb") as binary:
        if hashlib.file_digest(binary, "sha256").hexdigest() != executable_digest:
            raise RuntimeError("native renderer executable changed during verification")
    print(json.dumps({
        "cases": len(CASES),
        "jinja2_version": __import__("jinja2").__version__,
        "status": "native_plain_chat_corpus_parity_pass",
        "candidate": "native-pih-qwen-format",
        "candidate_sha256": executable_digest,
        "token_ids_compared": token_reference is not None,
        "tokenizer_sha256": QWEN3_TOKENIZER_SHA256 if token_reference is not None else None,
        "exclusions": ["thinking", "reasoning histories", "tools", "special literals in content", "model execution"]
                      + ([] if token_reference is not None else ["token IDs"]),
        "template_sha256": template_sha256,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
