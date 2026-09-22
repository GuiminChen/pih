#!/usr/bin/env python3
"""Deterministic subprocess fixture for the DeepSeek M5 target runner tests."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True)
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--baseline-version")
    parser.add_argument(
        "--behavior",
        choices=(
            "good",
            "nonzero",
            "timeout",
            "duplicate",
            "trailing",
            "leak-descendant",
        ),
        default="good",
    )
    arguments = parser.parse_args()
    if arguments.behavior == "nonzero":
        return 7
    if arguments.behavior == "timeout":
        time.sleep(10)
    invocation = os.environ["PIH_M5_INVOCATION_ID"]
    role = arguments.role
    candidate = role == "candidate"
    scale = 8 if candidate else 10
    observation = {
        "schema": (
            "pih.deepseek.m5.raw-run-observation.v2"
            if role in {"control", "candidate"}
            else "pih.deepseek.m5.raw-baseline-observation.v2"
        ),
        "invocation_id": invocation,
        "role": role,
        "repetition": arguments.repetition,
        "workload_instance_root": arguments.workload,
        "clock_abi": "monotonic_ns_v1",
        "requests": [
            {
                "ordinal": 0,
                "input_root": "8" * 64,
                "output_root": "9" * 64,
                "outcome": "completed",
                "prompt_tokens": 4,
                "output_tokens": 2,
                "prefill_ns": str(scale * 100),
                "decode_ns": str(scale * 50),
                "ttft_ns": str(scale * 10),
                "itl_ns": [str(scale * 5)],
            }
        ],
        "gpu_bytes_samples": ["1000", "2000"],
        "host_bytes_samples": ["3000", "4000"],
    }
    if role not in {"control", "candidate"}:
        observation["baseline_version"] = arguments.baseline_version
    encoded = json.dumps(
        observation,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    if arguments.behavior == "duplicate":
        payload = b'{"schema":"bad","schema":"duplicate"}\n'
    elif arguments.behavior == "trailing":
        payload = (encoded + "\ntrailing").encode("ascii")
    else:
        payload = (encoded + "\n").encode("ascii")
    if arguments.behavior == "leak-descendant":
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.DEVNULL,
        )
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
