"""Extract pinned NVIDIA CUDA/NCCL headers only; no installation, binaries or model execution."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--nccl", action="store_true", help="prepare NCCL 2.31.2 instead of CUDA headers")
    args = parser.parse_args()
    destination = (args.output or ROOT / ("out/nccl-syntax-headers-2.31.2" if args.nccl
                                        else "out/cuda-syntax-headers-13.2.86")).resolve()
    allowed = (ROOT / "out").resolve()
    if allowed not in destination.parents or destination.exists():
        parser.error("output must be a new directory strictly below repository out")
    lock_bytes = Path(__file__).with_name("nccl-syntax-headers.lock.json" if args.nccl
                                        else "cuda-syntax-headers.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ("cuda-headers-" + uuid.uuid4().hex)
    staging.mkdir()
    include = staging / "include"
    include.mkdir()
    # Clang's CUDA header discovery requires bin/ as well as include/. Keep it
    # EMPTY: no compiler/tool binaries are installed. Always use -nocudalib.
    (staging / "bin").mkdir()
    ledger = {}
    total = 0
    for package in lock["packages"]:
        print(f"fetching {package['name']} {package.get('version', lock.get('cuda_version'))}", flush=True)
        request = urllib.request.Request(package["url"], headers={"User-Agent": "PIH-header-preparation/1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(package["bytes"] + 1)
        if len(payload) != package["bytes"] or hashlib.sha256(payload).hexdigest() != package["sha256"]:
            raise ValueError("CUDA package size/hash mismatch: " + package["name"])
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            package_members = set()
            for member in archive.infolist():
                path = PurePosixPath(member.filename)
                if path.is_absolute() or ".." in path.parts or "\\" in member.filename or ":" in member.filename:
                    raise ValueError("unsafe CUDA archive path")
                if member.filename in package_members:
                    raise ValueError("duplicate archive path")
                package_members.add(member.filename)
                if member.is_dir():
                    continue
                mode = member.external_attr >> 16
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    raise ValueError("CUDA archive contains a special file")
                if "include" in path.parts and path.parts[0] == "nvidia":
                    relative = PurePosixPath(*path.parts[path.parts.index("include") + 1:])
                    target = include / relative
                elif "license" in path.name.lower() or "copyright" in path.name.lower():
                    target = staging / "licenses" / package["name"] / path
                else:
                    continue
                total += member.file_size
                if total > 128 * 1024 * 1024 or member.file_size > 16 * 1024 * 1024:
                    raise ValueError("CUDA header extraction size limit exceeded")
                data = archive.read(member)
                if target.exists():
                    if target.read_bytes() != data:
                        raise ValueError("conflicting CUDA header: " + str(target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                ledger[target.relative_to(staging).as_posix()] = hashlib.sha256(data).hexdigest()
    required_headers = ("nccl.h",) if args.nccl else (
        "cuda.h", "cuda_runtime_api.h", "cuda_bf16.h", "cuda_fp8.h", "crt/host_config.h")
    for required in required_headers:
        if not (include / required).is_file():
            raise ValueError("required CUDA header missing: " + required)
    (staging / "receipt.json").write_text(json.dumps({
        "schema": lock["schema"], "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "cuda_version": lock.get("cuda_version"), "nccl_version": lock.get("nccl_version"), "members": ledger,
        "purpose": "syntax-only; no executable, runtime library, driver or model installation"
    }, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    print(f"NVIDIA syntax headers ready: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
