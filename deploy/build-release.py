"""Configure, build and install a native profile without probing or activating GPUs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMON = ["pih.platform.linux", "pih.backend.nvidia-cuda", "pih.surface.text-http"]
SHARED = ["pih.execution.default", "pih.memory.host-spill", "pih.storage.verified-artifact"]
PROFILES = {
    "qwen-4090d": (89, COMMON + SHARED + ["pih.model.qwen3"]),
    "qwen-h100": (90, COMMON + SHARED + ["pih.model.qwen3"]),
    "deepseek-pp1": (89, COMMON + SHARED + ["pih.model.deepseek-v4-flash", "pih.kernels.deepseek-v4.sm89"]),
    "deepseek-b300": (103, COMMON + ["pih.transport.nccl", "pih.kernels.deepseek-v41.sm103", "pih.model.deepseek-v41"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=PROFILES)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--cmake-arg", action="append", default=[], help="Use --cmake-arg=-DNAME=value for a toolchain/library setting")
    args = parser.parse_args()
    build, bundle = args.build_dir.resolve(), args.bundle.resolve()
    if not 1 <= args.jobs <= 128:
        parser.error("jobs must be 1..128")
    if (args.bundle.is_symlink() or bundle.exists() or build == ROOT or ROOT.is_relative_to(build)
            or ROOT.is_relative_to(bundle) or build.is_relative_to(bundle) or bundle.is_relative_to(build)):
        parser.error("use a dedicated build directory and a new disjoint bundle")
    sm, plugins = PROFILES[args.profile]
    flags = dict(PIH_DEPLOYMENT_PROFILE="custom", PIH_ENABLED_PLUGINS=";".join(plugins),
                 CMAKE_BUILD_TYPE="Release", CMAKE_EXPORT_COMPILE_COMMANDS="ON",
                 CMAKE_CUDA_ARCHITECTURES=f"{sm}-real", PIH_QWEN_KERNEL_ARCHITECTURES=str(sm if sm != 103 else 89),
                 PIH_ENABLE_CUDA="ON", PIH_BUILD_WORKER="ON", PIH_BUILD_PLUGINS="ON",
                 PIH_BUILD_TOKENIZER_TOOLS="ON", PIH_BUILD_MONOLITH="OFF", PIH_BUILD_PYTHON="OFF",
                 PIH_ENABLE_NCCL="OFF", BUILD_TESTING="OFF", PIH_BUILD_TESTS="OFF",
                 PIH_BUILD_NATIVE_CONTRACT_TESTS="OFF", PIH_BUILD_QWEN_QUALIFICATION_TOOLS="OFF",
                 PIH_BUILD_NATIVE_ENGRAM_NCCL="ON" if sm == 103 else "OFF",
                 PIH_BUILD_ARTIFACT_TOOLS="ON" if args.profile == "deepseek-pp1" else "OFF",
                 PIH_BUILD_QWEN_ARTIFACT_TOOLS="ON" if args.profile.startswith("qwen") else "OFF")
    # Extra dependency settings cannot override the profile's closed selection.
    commands = [[args.cmake, "-S", str(ROOT), "-B", str(build), "-G", "Ninja", *args.cmake_arg,
                 *[f"-D{k}={v}" for k,v in flags.items()]],
                [args.cmake, "--build", str(build), "--target", "pih-release", "--parallel", str(args.jobs)],
                [args.cmake, "--install", str(build), "--prefix", str(bundle), "--component", "pih-release"]]
    for command in commands:
        print(json.dumps({"command": command}), flush=True)
        subprocess.run(command, check=True)
    print("Native build/install complete. Hardware execution unverified. No model Lock or weights were fabricated.")


if __name__ == "__main__":
    main()
