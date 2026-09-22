#!/usr/bin/env python3
"""Compile target-collected Qwen relative-quality measurements fail closed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping


_REPOSITORY = Path(__file__).resolve().parents[1]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

from tools.evidence.qwen_teacher_forced_quality import (  # noqa: E402
    QwenTeacherForcedQuality, TaskScoreReceipt, TeacherForcedCategoryReceipt,
    compile_qwen_teacher_forced_quality,
    verify_qwen_teacher_forced_quality_manifest,
)


_MAX_INPUT_BYTES = 4 << 20
_FIELDS = {
    "schema", "locked_quality_partition_root", "bf16_run_root", "int4_run_root",
    "metric_definitions_root", "reduction_implementation_root", "categories",
    "tasks", "zero_baseline_absolute_delta_micros",
}
_CATEGORY_FIELDS = {
    "category", "evaluated_positions", "equal_argmax_positions", "bf16_nll_sum",
    "int4_nll_sum", "nonfinite_count", "paired_bootstrap_ci_root",
}
_TASK_FIELDS = {
    "task_id", "scoring_units", "official_full_set_units",
    "small_sample_exception", "bf16_score_micros", "int4_score_micros",
    "weight_units", "scorer_root", "data_root", "bootstrap_unit_root",
}


def _object(value: object, fields: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} fields are invalid")
    return value


def parse_quality_input(source: bytes) -> Mapping[str, Any]:
    if not isinstance(source, bytes) or not source or len(source) > _MAX_INPUT_BYTES:
        raise ValueError("quality input exceeds its control-byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(source.decode("ascii"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("quality input is not ASCII JSON") from error
    return _object(value, _FIELDS, "quality input")


def compile_quality(source: Mapping[str, Any]) -> QwenTeacherForcedQuality:
    root = _object(source, _FIELDS, "quality input")
    if root["schema"] != "pih.qwen_teacher_forced_quality_input.v1":
        raise ValueError("quality input schema differs")
    raw_categories = root["categories"]
    raw_tasks = root["tasks"]
    if not isinstance(raw_categories, list) or not isinstance(raw_tasks, list):
        raise ValueError("quality measurements must be arrays")
    categories = tuple(TeacherForcedCategoryReceipt.create(**_object(
        value, _CATEGORY_FIELDS, f"categories[{index}]"
    )) for index, value in enumerate(raw_categories))
    tasks = tuple(TaskScoreReceipt.create(**_object(
        value, _TASK_FIELDS, f"tasks[{index}]"
    )) for index, value in enumerate(raw_tasks))
    result = compile_qwen_teacher_forced_quality(
        locked_quality_partition_root=root["locked_quality_partition_root"],
        bf16_run_root=root["bf16_run_root"], int4_run_root=root["int4_run_root"],
        metric_definitions_root=root["metric_definitions_root"],
        reduction_implementation_root=root["reduction_implementation_root"],
        categories=categories, tasks=tasks,
        zero_baseline_absolute_delta_micros=(
            root["zero_baseline_absolute_delta_micros"]
        ),
    )
    if result.gate_state != "passed":
        raise ValueError("teacher-forced quality gate did not pass")
    return result


def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file_once(payload: bytes, target: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    published = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, target)
        published = True
        _sync_directory(target.parent)
    except BaseException:
        if published:
            target.unlink(missing_ok=True)
            try:
                _sync_directory(target.parent)
            except OSError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def compile_quality_file(
    source: Mapping[str, Any], destination: str | os.PathLike[str],
) -> QwenTeacherForcedQuality:
    result = compile_quality(source)
    verify_qwen_teacher_forced_quality_manifest(
        result.manifest_bytes, expected_root=result.quality_root
    )
    target = Path(destination).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"quality output already exists: {target}")
    _publish_file_once(result.manifest_bytes, target)
    if target.read_bytes() != result.manifest_bytes:
        target.unlink(missing_ok=True)
        try:
            _sync_directory(target.parent)
        except OSError:
            pass
        raise RuntimeError("published quality manifest drifted")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    source = arguments.input.read_bytes()
    result = compile_quality_file(parse_quality_input(source), arguments.output)
    print(result.quality_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
