"""Verify an installed native SDK header boundary without loading binaries."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HEADERS = (
    "plugin_sdk/abi.h", "plugin_sdk/capability.h", "plugin_sdk/handles.h",
    "plugin_sdk/kernel_pack.h", "plugin_sdk/lifecycle.h", "plugin_sdk/status.h",
    "contracts/engine_v1.h", "contracts/token_generation_v1.h",
    "contracts/text_inference_v2.h", "contracts/qwen_kernels_v1.h",
    "contracts/deepseek_kernels_v1.h", "contracts/deepseek_v41_sm103_kernels_v1.h",
    "contracts/execution_default_v1.h",
    "contracts/execution_controller_v1.h",
    "contracts/transport_collective_v1.h",
    "contracts/health_v1.h", "contracts/memory_host_spill_v1.h",
    "contracts/nvidia_cuda_v1.h", "contracts/nvidia_cuda_memory_v1.h",
    "contracts/nvidia_cuda_resources_v1.h", "contracts/nvidia_cuda_async_v1.h",
    "contracts/openai_http_v1.h", "contracts/platform_linux_v1.h",
    "contracts/verified_artifact_v1.h", "contracts/artifact_snapshot_v1.h",
)
CONFIGS = ("PIHPluginSDKConfig.cmake", "PIHPluginSDKConfigVersion.cmake",
           "PIHPluginSDKTargets.cmake")


def verify_source_boundary() -> None:
    expected = set(HEADERS)
    if len(expected) != len(HEADERS):
        raise ValueError("duplicate SDK header in verifier list")
    public_abi = {f"plugin_sdk/{path.name}" for path in
                  (ROOT / "include/pih/plugin_sdk").glob("*.h")
                  if path.name != "status_bridge.h"}
    expected_abi = {name for name in expected if name.startswith("plugin_sdk/")}
    if public_abi != expected_abi:
        raise ValueError(f"SDK ABI directory mismatch: missing={sorted(public_abi - expected_abi)}, "
                         f"extra={sorted(expected_abi - public_abi)}")
    cmake = (ROOT / "cmake/PIHPluginSDK.cmake").read_text(encoding="utf-8")
    contracts = set(re.findall(r'include/pih/(contracts/[^"\s]+\.h)', cmake))
    expected_contracts = {name for name in expected if name.startswith("contracts/")}
    if contracts != expected_contracts:
        raise ValueError(f"SDK install rule mismatch: missing={sorted(expected_contracts - contracts)}, "
                         f"extra={sorted(contracts - expected_contracts)}")
    probe = (ROOT / "examples/external-plugin/sdk_c_header_probe.c").read_text(encoding="utf-8")
    includes = re.findall(r'^#include "pih/([^"\n]+\.h)"', probe, re.MULTILINE)
    if len(includes) != len(expected) or set(includes) != expected:
        raise ValueError(f"external SDK C probe mismatch: missing={sorted(expected - set(includes))}, "
                         f"extra={sorted(set(includes) - expected)}")


def verify(prefix: Path) -> None:
    verify_source_boundary()
    directory = prefix / "include/pih"
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("missing or symlinked include/pih directory")
    observed = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"symlink in SDK headers: {path.relative_to(directory)}")
        if path.is_file():
            observed.add(path.relative_to(directory).as_posix())
    if observed != set(HEADERS):
        raise ValueError(f"header set mismatch: missing={sorted(set(HEADERS) - observed)}, "
                         f"extra={sorted(observed - set(HEADERS))}")
    for name in HEADERS:
        installed = (directory / name).read_bytes()
        source = (ROOT / "include/pih" / name).read_bytes()
        if hashlib.sha256(installed).digest() != hashlib.sha256(source).digest():
            raise ValueError(f"installed header differs from this source revision: {name}")
        for dependency in re.findall(rb'#\s*include\s*[<"](pih/[^>"\r\n]+)[>"]', installed):
            if dependency.decode("ascii").removeprefix("pih/") not in HEADERS:
                raise ValueError(f"SDK header references an unpublished dependency: {name}")
    for name in CONFIGS:
        path = prefix / "lib/cmake/PIHPluginSDK" / name
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= 1024 * 1024:
            raise ValueError(f"missing, symlinked or invalid SDK package file: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path, nargs="?")
    parser.add_argument("--source-only", action="store_true",
                        help="check CMake and C probe header declarations without an install")
    args = parser.parse_args()
    try:
        if args.source_only:
            verify_source_boundary()
        elif args.prefix is not None:
            verify(args.prefix.resolve(strict=True))
        else:
            parser.error("prefix is required unless --source-only is specified")
    except (ValueError, OSError, UnicodeError) as error:
        parser.exit(2, f"SDK install verification: {error}\n")
    if args.source_only:
        print(f"SDK source boundary verified: {len(HEADERS)} headers declared consistently")
        return 0
    print(f"SDK install boundary verified: {len(HEADERS)} exact source-matching headers and 3 package files; no binary execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
