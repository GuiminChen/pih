#!/usr/bin/env python3
"""Verify the deliberately non-authorizing Qwen quality corpus plan."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import io
import json
from pathlib import Path
from typing import Sequence


_HEADER = (
    "partition", "coverage_id", "measurement_unit", "minimum_count",
    "source_revision_state", "license_state", "redistribution_state",
    "cross_partition_overlap_state", "materialization_state", "gate_state",
)
_OPEN = ("unresolved",) * 5 + ("corpus_plan_open",)
_CATEGORIES = ("zh", "en", "code", "tool", "thinking", "non_thinking")


def _expected_rows() -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = [
        ("selection", "all_unique", "final_tokenizer_tokens", "1000000", *_OPEN)
    ]
    rows.extend(
        ("selection", category, "teacher_forced_target_positions", "100000", *_OPEN)
        for category in _CATEGORIES
    )
    rows.extend(
        ("locked_quality", category, "teacher_forced_target_positions", "100000", *_OPEN)
        for category in _CATEGORIES
    )
    rows.extend((
        ("locked_quality", "critical_code_task", "independent_scoring_units", "200", *_OPEN),
        ("locked_quality", "critical_tool_task", "independent_scoring_units", "200", *_OPEN),
    ))
    rows.extend(
        ("locked_trajectory", category, "fixtures", "200", *_OPEN)
        for category in _CATEGORIES
    )
    return tuple(rows)


_EXPECTED_ROWS = _expected_rows()


def verify_plan(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or path.is_symlink():
        raise ValueError("plan must be a non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("plan must be a regular file")
    raw = resolved.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise ValueError("plan must use canonical LF framing")
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("plan must be canonical ASCII") from error
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = tuple(next(reader))
        rows = tuple(tuple(row) for row in reader)
    except (StopIteration, csv.Error) as error:
        raise ValueError("plan CSV is malformed") from error
    if header != _HEADER or rows != _EXPECTED_ROWS:
        raise ValueError("plan header or row set differs from qwen_quality_corpus_plan_v2")
    return {
        "schema": "pih.qwen_quality_corpus_plan_observation.v1",
        "plan_abi": "qwen_quality_corpus_plan_v2",
        "row_count": len(rows),
        "gate_state": "corpus_plan_open",
        "plan_sha256": sha256(raw).hexdigest(),
    }


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parsed = parser.parse_args(arguments)
    print(json.dumps(verify_plan(parsed.plan), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
