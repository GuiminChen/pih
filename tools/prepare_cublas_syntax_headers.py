"""Extract pinned NVIDIA cuBLAS headers only; no libraries, installation or tests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import urllib.request
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ("cublas.h", "cublas_api.h", "cublas_v2.h", "cublasLt.h", "cublasXt.h", "nvblas.h")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out/cublas-syntax-headers-13.4.1.3")
    parser.add_argument("--tar", default="tar", help="bsdtar with zstd support")
    args = parser.parse_args()
    destination = args.output.resolve()
    if (ROOT / "out").resolve() not in destination.parents or destination.exists():
        parser.error("output must be a new directory strictly below repository out")
    lock_bytes = Path(__file__).with_name("cublas-syntax-headers.lock.json").read_bytes()
    lock = json.loads(lock_bytes)
    request = urllib.request.Request(lock["url"], headers={"User-Agent": "PIH-header-preparation/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        package = response.read(lock["bytes"] + 1)
    if len(package) != lock["bytes"] or hashlib.sha256(package).hexdigest() != lock["sha256"]:
        raise ValueError("cuBLAS package size/hash mismatch")
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        matches = [item for item in archive.infolist() if item.filename == lock["payload"]]
        if len(matches) != 1 or matches[0].file_size > 1024 * 1024:
            raise ValueError("cuBLAS payload missing, ambiguous or oversized")
        payload = archive.read(matches[0])
    # Never extract an archive into the filesystem: read only allowlisted files
    # to memory. Ignore library stubs, symlinks, install metadata and test scripts.
    names = subprocess.check_output([args.tar, "-tf", "-"], input=payload).splitlines()
    details = subprocess.check_output([args.tar, "-tvf", "-"], input=payload).splitlines()
    selected = {f"targets/x86_64-linux/include/{name}": f"include/{name}" for name in HEADERS}
    selected["info/licenses/LICENSE"] = "licenses/NVIDIA_LICENSE"
    files = {}
    for member, relative in selected.items():
        encoded = member.encode("ascii")
        listing = [line for line in details if line.endswith(b" " + encoded)]
        if names.count(encoded) != 1 or len(listing) != 1 or not listing[0].startswith(b"-"):
            raise ValueError("cuBLAS selected member must be one regular file: " + member)
        content = subprocess.check_output([args.tar, "-xOf", "-", member], input=payload)
        if not content or len(content) > 1024 * 1024:
            raise ValueError("cuBLAS selected member size invalid")
        files[relative] = content
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ("cublas-headers-" + uuid.uuid4().hex)
    staging.mkdir()
    for relative, content in files.items():
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (staging / "receipt.json").write_text(json.dumps({
        "schema": lock["schema"], "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
        "package": lock, "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "purpose": "syntax-only; no runtime or linker libraries",
    }, indent=2) + "\n", encoding="utf-8")
    staging.rename(destination)
    print(f"cuBLAS headers ready: {destination}")


if __name__ == "__main__":
    main()
