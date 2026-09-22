"""Prepare pinned Linux headers only. No package installation or maintainer scripts.

Requires bsdtar with Debian ar and zstd support (Windows system tar provides it).
The generated sysroot is for -fsyntax-only, never a deployable runtime/toolchain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import uuid
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out/linux-syntax-sysroot")
    parser.add_argument("--tar", default="tar")
    args = parser.parse_args()
    destination = args.output.resolve()
    allowed = (ROOT / "out").resolve()
    if allowed not in destination.parents or destination.exists():
        parser.error("output must be a new directory strictly below this repository's out directory")
    lock_path = Path(__file__).with_name("linux-syntax-sysroot.lock.json")
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep failed attempts for inspection; never recursively delete user paths.
    # Inherit workspace ACLs. Windows mkdtemp's private ACL can make an elevated
    # download result unreadable to a later unelevated compiler process.
    staging = destination.parent / ("syntax-headers-" + uuid.uuid4().hex)
    staging.mkdir()
    include_root = staging / "sysroot"
    include_root.mkdir()
    for package in lock["packages"]:
        source = "https://archive.ubuntu.com/ubuntu/" + package["path"]
        print(f"fetching {package['name']} {package['version']}", flush=True)
        request = urllib.request.Request(source, headers={"User-Agent": "PIH-header-preparation/1"})
        with urllib.request.urlopen(request, timeout=45) as response:
            payload = response.read(package["bytes"] + 1)
        if len(payload) != package["bytes"] or hashlib.sha256(payload).hexdigest() != package["sha256"]:
            raise ValueError(f"package content mismatch: {package['name']}")
        archive = staging / PurePosixPath(package["path"]).name
        archive.write_bytes(payload)
        members = subprocess.check_output([args.tar, "-tf", str(archive)], text=True).splitlines()
        data_members = [name for name in members if name.startswith("data.tar.")]
        if len(data_members) != 1:
            raise ValueError("Debian package data member is ambiguous")
        data = subprocess.check_output([args.tar, "-xOf", str(archive), data_members[0]])
        paths = subprocess.check_output([args.tar, "-tf", "-"], input=data).decode("utf-8").splitlines()
        for name in paths:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("unsafe package path")
        listing = subprocess.check_output([args.tar, "-tvf", "-"], input=data).splitlines()
        for line in listing:
            if b"./usr/include/" in line and not line.startswith((b"-", b"d")):
                raise ValueError("header archive contains links or special files")
        subprocess.run([args.tar, "-xf", "-", "-C", str(include_root), "./usr/include"], input=data, check=True)
        # Include package copyright notices without unpacking runtime binaries.
        copyright_name = f"./usr/share/doc/{package['name']}/copyright"
        if copyright_name in paths:
            notice = subprocess.check_output([args.tar, "-xOf", "-", copyright_name], input=data)
            notices = include_root / "licenses"
            notices.mkdir(exist_ok=True)
            (notices / f"{package['name']}.copyright").write_bytes(notice)
    (include_root / "receipt.json").write_text(json.dumps({
        "schema": lock["schema"], "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "packages": lock["packages"], "purpose": "syntax-only; no runtime or linker libraries"
    }, indent=2) + "\n", encoding="utf-8")
    include_root.rename(destination)
    print(f"headers ready: {destination}")
    print(f"download audit directory retained: {staging}")


if __name__ == "__main__":
    main()
