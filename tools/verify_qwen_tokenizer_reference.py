"""Compare the native plugin tokenizer with a pinned offline reference artifact.

This is an explicit qualification command, not a runtime fallback. The caller
must supply a trusted, locally built pih-tokenize executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


EXPECTED_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
MAXIMUM_ARTIFACT_BYTES = 12 * 1024 * 1024
MAXIMUM_OUTPUT_BYTES = 8 * 1024 * 1024
MAXIMUM_TOKENS = 65536

CORPUS = (
    "hello world",
    "I'm testing Qwen's tokenizer.",
    "e\u0301 é NFC",
    "你好，世界！",
    "١٢٣ 1234 １２３",
    "symbols!!!\r\nnext",
    "  leading   middle  trailing  ",
    "emoji 😀 family 👨‍👩‍👧‍👦",
)


def parse_native_tokens(payload: bytes) -> tuple[int, ...]:
    if not payload or len(payload) > MAXIMUM_OUTPUT_BYTES:
        raise ValueError("native tokenizer output exceeds its byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate native tokenizer output field")
            result[key] = value
        return result

    result = json.loads(payload.decode("utf-8"), object_pairs_hook=unique)
    if type(result) is not dict or set(result) != {"tokens"}:
        raise ValueError("native tokenizer output must contain only tokens")
    tokens = result["tokens"]
    if type(tokens) is not list or len(tokens) > MAXIMUM_TOKENS:
        raise ValueError("native tokenizer token count is invalid")
    if any(type(token) is not int or not 0 <= token < 151936 for token in tokens):
        raise ValueError("native tokenizer returned an invalid Qwen token ID")
    return tuple(tokens)


def native_encode(
    executable: Path, tokenizer: Path, request: Path, text: str, *, timeout: int,
) -> tuple[int, ...]:
    request.write_bytes(json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"))
    # Use files rather than retaining unbounded subprocess output in memory.
    # The explicitly selected local executable must still be trusted by the caller.
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        completed = subprocess.run(
            [str(executable), "qwen3", str(tokenizer), str(request)],
            stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
            timeout=timeout, check=False, shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"native tokenizer exited with status {completed.returncode}")
        output.seek(0)
        return parse_native_tokens(output.read(MAXIMUM_OUTPUT_BYTES + 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-tokenizer", type=Path, required=True,
                        help="trusted locally built pih-tokenize executable")
    parser.add_argument("--timeout-seconds", type=int, default=60,
                        help="per-corpus-row subprocess deadline (1..600)")
    arguments = parser.parse_args()
    if not 1 <= arguments.timeout_seconds <= 600:
        parser.error("--timeout-seconds must be between 1 and 600")
    executable = arguments.native_tokenizer.resolve(strict=True)
    if not executable.is_file():
        parser.error("--native-tokenizer must name a regular executable file")
    # Lazy import keeps --help and tooling inspection independent of this
    # optional qualification dependency, and never imports the old pih package.
    from tokenizers import Tokenizer

    payload = sys.stdin.buffer.read(MAXIMUM_ARTIFACT_BYTES + 1)
    if len(payload) > MAXIMUM_ARTIFACT_BYTES:
        raise ValueError("tokenizer artifact exceeds verifier ceiling")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("tokenizer artifact SHA-256 differs from pinned Qwen3 object")
    reference = Tokenizer.from_str(payload.decode("utf-8"))
    with executable.open("rb") as binary:
        executable_digest = hashlib.file_digest(binary, "sha256").hexdigest()
    with tempfile.TemporaryDirectory(prefix="pih-qwen-tokenizer-") as directory:
        tokenizer = Path(directory) / "tokenizer.json"
        request = Path(directory) / "request.json"
        tokenizer.write_bytes(payload)
        for index, text in enumerate(CORPUS):
            expected = tuple(reference.encode(text, add_special_tokens=False).ids)
            observed = native_encode(
                executable, tokenizer, request, text, timeout=arguments.timeout_seconds,
            )
            if observed != expected:
                raise AssertionError(
                    f"native tokenizer parity failed at corpus row {index}: "
                    f"expected {expected!r}, observed {observed!r}"
                )
    with executable.open("rb") as binary:
        if hashlib.file_digest(binary, "sha256").hexdigest() != executable_digest:
            raise RuntimeError("native tokenizer executable changed during verification")
    print(json.dumps({
        "artifact_sha256": digest,
        "candidate": "native-pih-tokenize",
        "candidate_sha256": executable_digest,
        "corpus_rows": len(CORPUS),
        "policy_exclusion": "registered special-token literals and decoding are outside this corpus",
        "status": "native_encode_corpus_parity_pass",
        "tokenizers_version": __import__("tokenizers").__version__,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
