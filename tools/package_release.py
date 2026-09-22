"""Build an explicitly inventoried source or installed-native tarball; never run it."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def source_files(root=ROOT):
    policy = json.loads((root / "release/source-policy.json").read_text(encoding="utf-8"))
    files = set()
    for pattern in policy["include"]:
        for path in root.glob(pattern):
            relative = path.relative_to(root).as_posix()
            if (path.is_file() and not any(fnmatch.fnmatchcase(relative, p) for p in policy["exclude"])
                    and path.suffix not in policy["forbidden_suffixes"]):
                if path.is_symlink():
                    raise ValueError(f"release symlink rejected: {relative}")
                files.add(relative)
    return sorted(files)


def audit_text(name, data):
    if b"\0" in data:
        return
    text = data.decode("utf-8", errors="replace")
    patterns = {
        "private key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "GitHub credential": r"\b(?:ghp_|github_pat_)[A-Za-z0-9_]{30,}",
        "AWS credential": r"\bAKIA[0-9A-Z]{16}\b",
        "local user path": r"[A-Za-z]:[\\/](?:Users|github|ProgramData)[\\/]",
    }
    for label, pattern in patterns.items():
        if re.search(pattern, text):
            raise ValueError(f"{label} in selected release file: {name}")


def package(root, files, output, kind, expected=None):
    if output.exists() or output.is_symlink():
        raise ValueError("output must be new")
    entries = []
    payloads = []
    for name in files:
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"invalid release member: {name}")
        data = path.read_bytes()
        if expected is not None:
            entry = expected[name]
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"native member changed before archive write: {name}")
        audit_text(name, data)
        entries.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        payloads.append((name, data))
    manifest = dict(schema="pih.release-inventory.v1", kind=kind,
                    hardware_validation="not performed", files=entries)
    payloads.append(("RELEASE-MANIFEST.json", (json.dumps(manifest, indent=2)+"\n").encode()))
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidentally replacing any existing archive.
    with output.open("xb") as stream, tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, data in payloads:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.startswith("bin/") or name.endswith(".sh") else 0o644
            archive.addfile(info, io.BytesIO(data))
    with tarfile.open(output) as archive:
        assert archive.getnames() == [name for name, _ in payloads]
        for entry in entries:
            assert hashlib.sha256(archive.extractfile(entry["path"]).read()).hexdigest() == entry["sha256"]
    print(json.dumps(dict(archive=str(output), files=len(entries),
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("source", "native"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    expected = None
    if args.kind == "source":
        files = source_files(root)
    else:
        # Package exactly the members written by the explicit CMake component.
        files = sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())
        receipt_bytes = (root / "NATIVE-INSTALL.json").read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt.get("schema") != "pih.native-install.v1":
            raise ValueError("invalid native install receipt")
        expected = {entry["path"]: entry for entry in receipt["files"]}
        if len(expected) != len(receipt["files"]) or set(files) != set(expected) | {"NATIVE-INSTALL.json"}:
            raise ValueError("native installation has missing, duplicate or extra files")
        for name, entry in expected.items():
            path = root / name
            if not path.resolve().is_relative_to(root) or path.is_symlink():
                raise ValueError("invalid native member path")
            data = path.read_bytes()
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError(f"native installed member changed: {name}")
        if not (root / "bin/pih-worker").is_file() or not (root / "share/pih/licenses/LICENSE").is_file():
            raise ValueError("not a complete native release installation")
        allowed = ("bin/", "lib/", "share/pih/", "deploy/", "cmake/")
        for name in files:
            if name == "NATIVE-INSTALL.json":
                continue
            if not name.startswith(allowed) or any(x in name for x in ("reference", "__pycache__", "minimal-plugin")):
                raise ValueError(f"nonrelease native member: {name}")
        expected["NATIVE-INSTALL.json"] = dict(bytes=len(receipt_bytes), sha256=hashlib.sha256(receipt_bytes).hexdigest())
    package(root, files, args.output.resolve(), args.kind, expected)


if __name__ == "__main__":
    main()
