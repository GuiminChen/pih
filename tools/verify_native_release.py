"""Inspect installed ELF/device images and real-library closure without loading them."""
from __future__ import annotations
import argparse
import functools
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--library-root", action="append", type=Path, default=[])
    p.add_argument("--readelf", default="readelf")
    p.add_argument("--cuobjdump", default="cuobjdump")
    p.add_argument("--report", required=True, type=Path)
    args = p.parse_args()
    root = args.root.resolve(strict=True)
    commands = []

    def run(command):
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        commands.append(dict(command=command, exit_code=result.returncode))
        if result.returncode:
            raise ValueError(f"inspection failed: {command}: {result.stderr}")
        return result.stdout

    def elf(path):
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"

    @functools.cache
    def dynamic(path):
        text = run([args.readelf, "-d", "-W", str(path)])
        return dict(needed=re.findall(r"\(NEEDED\).*?\[(.*?)\]", text),
                    soname=next(iter(re.findall(r"\(SONAME\).*?\[(.*?)\]", text)), None),
                    runpaths=re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", text))

    @functools.cache
    def symbols(path):
        defined, undefined = set(), set()
        for line in run([args.readelf, "--dyn-syms", "-W", str(path)]).splitlines():
            f = line.split()
            if len(f) < 8 or not f[0].rstrip(":").isdigit():
                continue
            if f[6] == "UND" and f[4] == "GLOBAL":
                undefined.add(f[7])
            elif f[6] != "UND" and f[4] in ("GLOBAL", "WEAK") and f[5] in ("DEFAULT", "PROTECTED"):
                defined.add(f[7])
        return defined, undefined

    index = {}
    for directory in [root / "lib", *args.library_root]:
        for path in directory.rglob("*.so*"):
            if not path.is_file() or "stubs" in path.parts or not elf(path):
                continue
            details = dynamic(path)
            if details["soname"]:
                index.setdefault(details["soname"], path)
    products = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not elf(path):
            continue
        data = path.read_bytes()
        item = dict(path=path.relative_to(root).as_posix(), bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(), executed=False)
        if path.suffix == ".cubin":
            sm = int(path.parent.name.removeprefix("sm_"))
            details = run([args.cuobjdump, "--dump-elf", str(path)])
            if not re.search(rf"\bsm\s*=\s*{sm}\b", details):
                raise ValueError("cubin architecture mismatch")
            exports = run([args.cuobjdump, "--dump-elf-symbols", str(path)])
            entries = re.findall(r"STO_ENTRY\s+(\S+)", exports)
            if len(entries) != 18 or sm not in (89, 90):
                raise ValueError("Qwen device entrypoint set mismatch")
            manifest = json.loads(Path(str(path)+".json").read_text())
            if manifest["cubin_sha256"] != item["sha256"] or manifest["cubin_bytes"] != len(data):
                raise ValueError("cubin manifest mismatch")
            model = root / "lib/pih_plugin_model_qwen3.so"
            if item["sha256"].encode() not in model.read_bytes():
                raise ValueError("Qwen model does not bind the installed cubin digest")
            item.update(sm=sm, device_entrypoints=entries)
        else:
            item.update(dynamic(path))
            if item["runpaths"] or any("python" in x.lower() for x in item["needed"]):
                raise ValueError(f"unexpected loader path/Python dependency: {path}")
            defined, undefined = symbols(path)
            if path.suffix == ".so":
                expected = ({"pih_kernel_pack_identity_v1_get", "pih_kernel_pack_api_v1_get"}
                            if path.name.startswith("pih_kernels_") else {"pih_plugin_entry_v1"})
                if defined != expected:
                    raise ValueError(f"unexpected exported ABI: {path}: {defined}")
            item["exports"] = sorted(defined)
            queue, dependencies, provided = item["needed"][:], {}, set()
            while queue:
                name = queue.pop()
                if name in dependencies:
                    continue
                if name not in index:
                    raise ValueError(f"missing real dependency {name} for {path}")
                library = index[name]
                dependencies[name] = str(library)
                queue.extend(dynamic(library)["needed"])
                exports, _ = symbols(library)
                provided.update(x.replace("@@", "@") for x in exports)
                provided.update(x.split("@")[0] for x in exports)
            missing = sorted(undefined - provided)
            if missing:
                raise ValueError(f"unresolved strong imports: {path}: {missing}")
            item.update(dependencies=dependencies, unresolved_strong_imports=missing)
            if path.name in ("pih_kernels_deepseek_v4_sm89.so", "pih_kernels_deepseek_v41_sm103.so"):
                sm, count = (89, 17) if "v4_sm89" in path.name else (103, 20)
                images = run([args.cuobjdump, "--list-elf", str(path)])
                architectures = re.findall(r"\.sm_(\d+)\.cubin", images)
                if architectures != [str(sm)] * count:
                    raise ValueError("embedded image count/architecture mismatch")
                ptx = run([args.cuobjdump, "--list-ptx", str(path)])
                if re.search(r"^PTX file\s+\d+:", ptx, re.MULTILINE):
                    raise ValueError("unexpected PTX fallback")
                item.update(sm=sm, device_image_count=count)
        products.append(item)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(dict(products=products, commands=commands,
        hardware_validation="not performed", remote_ci="not performed"), indent=2), encoding="utf-8")
    print(f"Verified {len(products)} ELF/device products; no program or GPU executed")


if __name__ == "__main__":
    main()
