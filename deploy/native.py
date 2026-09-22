"""Build, seal and run native plugins. No Python model execution or fallback."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"rtx4090d": ("NVIDIA GeForce RTX 4090 D", "8.9", 89),
           "h100-pcie": ("NVIDIA H100 PCIe", "9.0", 90)}
QWEN_TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"


def prepare_qwen_metadata(args):
    """Copy authenticated small assets only; never parse/execute model weights."""
    if platform.system() != "Linux":
        raise ValueError("Qwen metadata preparation requires Linux")
    if not re.fullmatch(r"[0-9a-f]{64}", args.config_sha256) or not args.config_sha256.strip("0"):
        raise ValueError("config-sha256 must be an independently trusted nonzero SHA-256")
    source = args.source_dir.resolve(strict=True)
    output = args.model_dir.resolve(strict=True)
    if source == output or output == ROOT or ROOT in output.parents:
        raise ValueError("use a distinct output model directory outside the repository")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    source_fd = os.open(source, flags)
    try:
        output_fd = os.open(output, flags)
        try:
            identity = os.fstat(output_fd)
            if identity.st_uid != os.geteuid() or identity.st_mode & 0o022:
                raise ValueError("output directory must be owned by this user and not group/other writable")
            assets = []
            for name, expected, limit in (
                ("config.json", args.config_sha256, 1024 * 1024),
                ("tokenizer.json", QWEN_TOKENIZER_SHA256, 16 * 1024 * 1024),
            ):
                # Authenticate both snapshots before creating either destination.
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
                             dir_fd=source_fd)
                with os.fdopen(fd, "rb") as stream:
                    before = os.fstat(stream.fileno())
                    if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
                        raise ValueError(f"invalid or oversized source metadata: {name}")
                    payload = stream.read(limit + 1)
                    after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                            after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                        raise ValueError(f"source metadata changed while reading: {name}")
                if len(payload) != before.st_size or hashlib.sha256(payload).hexdigest() != expected:
                    raise ValueError(f"metadata digest mismatch: {name}")
                try:
                    os.stat(name, dir_fd=output_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ValueError(f"destination already exists: {name}")
                assets.append((name, payload, expected))
            for name, payload, expected in assets:
                fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                             0o600, dir_fd=output_fd)
                with os.fdopen(fd, "w+b") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fchmod(stream.fileno(), 0o400)
                    os.fsync(stream.fileno())
                    stream.seek(0)
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(f"metadata readback mismatch: {name}")
                    held = os.fstat(stream.fileno())
                    named = os.stat(name, dir_fd=output_fd, follow_symlinks=False)
                    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino) or held.st_nlink != 1:
                        raise ValueError(f"metadata destination identity changed: {name}")
            named_dir = os.stat(output, follow_symlinks=False)
            if (identity.st_dev, identity.st_ino) != (named_dir.st_dev, named_dir.st_ino):
                raise ValueError("output directory identity changed")
            os.fsync(output_fd)
            print(json.dumps({"schema": "pih.qwen3.metadata-copy-receipt.v1",
                              "config_sha256": args.config_sha256,
                              "tokenizer_sha256": QWEN_TOKENIZER_SHA256,
                              "model_directory": str(output)}, sort_keys=True))
            return 0
        finally:
            os.close(output_fd)
    finally:
        os.close(source_fd)


def run(command):
    if str(command[0]) == "cmake" and "-S" in command:
        # Reset optional cached selections before this command's explicit
        # profile overrides. Otherwise a reused build tree can drag in tools,
        # tests or transports from an unrelated deployment.
        command = [command[0],
                   "-DPIH_BUILD_TESTS=OFF", "-DPIH_BUILD_NATIVE_CONTRACT_TESTS=OFF",
                   "-DBUILD_TESTING=OFF", "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF",
                   "-DPIH_BUILD_NATIVE_ENGRAM_NCCL=OFF", "-DPIH_ENABLE_NCCL=OFF",
                   "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=OFF",
                   "-DPIH_BUILD_QWEN_ARTIFACT_TOOLS=OFF",
                   "-DPIH_BUILD_QWEN_QUALIFICATION_TOOLS=OFF", *command[1:]]
    print(json.dumps({"command": [str(value) for value in command]}), flush=True)
    subprocess.run([str(value) for value in command], check=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_lock(lock, encoded):
    """Create a read-only Lock, or reuse only an identical owned regular file."""
    try:
        with lock.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
    except FileExistsError:
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
        with os.fdopen(os.open(lock, flags), "rb") as stream:
            identity = os.fstat(stream.fileno())
            if (not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1 or
                    identity.st_uid != os.geteuid() or identity.st_mode & 0o222 or
                    identity.st_size != len(encoded)):
                raise ValueError("existing Lock must be identical, owned, read-only and single-link")
            if stream.read(len(encoded) + 1) != encoded:
                raise ValueError("existing Lock differs; never overwriting")
    return lock


def parse_gpu_inventory(payload):
    """Parse complete physical-device rows; do not silently drop malformed GPUs."""
    if not payload or len(payload) > 64 * 1024:
        raise ValueError("GPU inventory is empty or exceeds its bound")
    devices = {}
    addresses = set()
    for fields in csv.reader(payload.splitlines()):
        if len(fields) != 4 or len(devices) >= 64:
            raise ValueError("invalid GPU inventory row count/fields")
        index, name, capability, address = (value.strip() for value in fields)
        if not index.isascii() or not index.isdecimal():
            raise ValueError("invalid GPU index")
        ordinal = int(index)
        match = re.fullmatch(r"([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])", address)
        if ordinal >= 64 or ordinal in devices or not match or not name:
            raise ValueError("invalid or duplicate GPU inventory identity")
        pci = tuple(int(part, 16) for part in match.groups())
        if pci[2] > 31 or pci in addresses:
            raise ValueError("invalid or duplicate GPU PCI address")
        addresses.add(pci)
        devices[ordinal] = (name, capability, pci)
    if not devices or set(devices) != set(range(len(devices))):
        raise ValueError("GPU inventory indexes must be complete and contiguous")
    return devices


def inspect_host(target):
    if platform.system() != "Linux":
        raise ValueError("native inference requires Linux; list and probe can run on Windows/macOS")
    if "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("NVIDIA_VISIBLE_DEVICES", "all") != "all":
        raise ValueError("unset CUDA_VISIBLE_DEVICES and expose all physical GPUs; this profile owns GPU 0")
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap,pci.bus_id", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True, timeout=15)
    rows = parse_gpu_inventory(result.stdout)
    expected = TARGETS[target][:2]
    if rows[0][:2] != expected:
        raise ValueError(f"GPU 0 must be {expected}, found {rows.get(0)}")
    if min(rows, key=lambda index: rows[index][2]) != 0:
        raise ValueError("nvidia-smi GPU 0 is not first in PCI order; fixed-device profile cannot safely select it")
    # Set before exec/any CUDA initialization. Never depend on FASTEST_FIRST,
    # and never silently mask/reorder physical devices via CUDA_VISIBLE_DEVICES.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    print(json.dumps({"gpu_selection": "physical-index-0-in-pci-order",
                      "pci_address": rows[0][2], "cuda_device_order": "PCI_BUS_ID"}), flush=True)


def validate_deployment_paths(build_dir, bundle, model=None):
    """Reject overlapping writable outputs before invoking a build/install."""
    source = ROOT.resolve()
    build_dir, bundle = Path(build_dir).resolve(), Path(bundle).resolve()

    def overlaps(left, right):
        return left == right or left in right.parents or right in left.parents

    for path in (build_dir, bundle):
        if path == Path(path.anchor) or path == source or path in source.parents:
            raise ValueError("build and bundle must be dedicated directories, not source/root ancestors")
        if path.exists() and not path.is_dir():
            raise ValueError("build and bundle paths must be directories")
    if overlaps(build_dir, bundle):
        raise ValueError("build and bundle directories must be disjoint")
    if model is not None:
        model = Path(model).resolve(strict=True)
        if not model.is_dir() or any(overlaps(model, path) for path in (source, build_dir, bundle)):
            raise ValueError("model directory must be disjoint from source, build and bundle trees")
    return build_dir, bundle


def build(build_dir, bundle, jobs, target):
    build_dir, bundle = validate_deployment_paths(build_dir, bundle)
    if platform.system() != "Linux":
        raise ValueError("build the native CUDA bundle on Linux")
    if bundle.exists():
        raise ValueError("build requires a new bundle directory; existing files are never overwritten")
    sm = TARGETS[target][2]
    for other in (89, 90):
        if other != sm and ((bundle / f"lib/pih_kernels_qwen3_sm{other}.so").exists() or
                            (bundle / f"lib/qwen3-sm{other}").exists()):
            raise ValueError("bundle contains another Qwen architecture; choose a fresh --bundle directory")
    run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_INSTALL_PREFIX={bundle}",
        "-DPIH_DEPLOYMENT_PROFILE=custom", "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF",
        "-DPIH_BUILD_TESTS=OFF", "-DPIH_ENABLE_NCCL=OFF", "-DPIH_ENABLE_CUDA=ON",
        "-DPIH_BUILD_WORKER=ON", "-DPIH_BUILD_PLUGINS=ON",
        "-DPIH_BUILD_QWEN_ARTIFACT_TOOLS=ON",
        f"-DPIH_QWEN_KERNEL_ARCHITECTURES={sm}", f"-DCMAKE_CUDA_ARCHITECTURES={sm}-real",
        "-DPIH_ENABLED_PLUGINS=pih.backend.nvidia-cuda;pih.execution.default;pih.model.qwen3;pih.platform.linux;pih.storage.verified-artifact;pih.surface.text-http"])
    run(["cmake", "--build", build_dir, "--parallel", jobs, "--target", "pih-worker", "pih-cli",
         "pih_plugin_backend_nvidia_cuda", "pih_plugin_platform_linux", "pih_plugin_execution_default",
         "pih_plugin_model_qwen3", "pih_plugin_storage_verified_artifact",
         "pih_plugin_surface_text_http", f"pih_kernels_qwen3_sm{sm}",
         "pih-qwen-int4-convert", "pih-qwen-int4-verify", "pih-qwen-source-verify"])
    run(["cmake", "--build", build_dir, "--parallel", jobs, "--target", "pih-release"])
    run(["cmake", "--install", build_dir, "--component", "pih-release"])


def qwen_artifact_tool(args, name, artifact_paths):
    """Select/build only native CPU tools; never import weight-processing code."""
    if platform.system() != "Linux":
        raise ValueError("use the Linux native Qwen artifact tools")
    if name not in ("pih-qwen-int4-convert", "pih-qwen-int4-verify", "pih-qwen-source-verify"):
        raise ValueError("unknown Qwen artifact tool")
    if not args.build:
        return args.bundle.resolve() / "bin" / name
    build_dir = args.build_dir.resolve()
    if not 1 <= args.jobs <= 128 or build_dir == ROOT or build_dir == Path(build_dir.anchor):
        raise ValueError("invalid Qwen artifact build directory or jobs")
    if any(path.is_relative_to(build_dir) for path in artifact_paths):
        raise ValueError("model artifacts must be outside the Qwen artifact build tree")
    run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
         "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
         "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF",
         "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF",
         "-DPIH_BUILD_TESTS=OFF", "-DPIH_ENABLE_CUDA=OFF",
         "-DPIH_ENABLE_NCCL=OFF", "-DPIH_BUILD_NATIVE_ENGRAM_NCCL=OFF",
         "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=OFF",
         "-DPIH_BUILD_QWEN_ARTIFACT_TOOLS=ON"])
    run(["cmake", "--build", build_dir, "--target", "pih-qwen-int4-convert", "pih-qwen-int4-verify", "pih-qwen-source-verify",
         "--parallel", args.jobs])
    return build_dir / "plugins/offline-qwen" / name


def verify_qwen_int4(args):
    artifact = args.artifact.resolve(strict=True)
    if not artifact.is_file():
        raise ValueError("artifact must be a regular file")
    roots = [args.file_sha256, args.source_root, args.binding_root, args.disposition_root]
    if any(len(value) != 64 or set(value) - set("0123456789abcdef") or not value.strip("0")
           for value in roots):
        raise ValueError("expected digests must be nonzero lowercase SHA-256")
    executable = qwen_artifact_tool(args, "pih-qwen-int4-verify", [artifact])
    run([executable, artifact, *roots])
    return 0


def seal(bundle, model, precision, target, context, build_dir, generation_timeout_ms,
         artifact_sha256, config_sha256):
    if type(generation_timeout_ms) is not int or not 1 <= generation_timeout_ms <= 86400000:
        raise ValueError("generation-timeout-ms must be 1..86400000")
    build_dir, bundle = validate_deployment_paths(build_dir, bundle, model)
    bundle = bundle.resolve(strict=True)
    model = model.resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", artifact_sha256) or not artifact_sha256.strip("0"):
        raise ValueError("artifact-sha256 must be an independently trusted nonzero SHA-256")
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256) or not config_sha256.strip("0"):
        raise ValueError("config-sha256 must be an independently trusted nonzero SHA-256")
    for name in ("config.json", "tokenizer.json", "model.safetensors" if precision == "bf16" else "model.xing-int4"):
        if not (model / name).is_file():
            raise ValueError(f"missing model artifact: {name}")
    artifact = model / ("model.safetensors" if precision == "bf16" else "model.xing-int4")
    for path, limit in ((artifact, 2 * 1024 * 1024 * 1024),
                        (model / "config.json", 1024 * 1024),
                        (model / "tokenizer.json", 16 * 1024 * 1024)):
        if path.is_symlink():
            raise ValueError(f"model artifact must not be a symlink: {path.name}")
        if not 0 < path.stat().st_size <= limit:
            raise ValueError(f"model artifact exceeds the native snapshot bound: {path.name}")
    if artifact.is_symlink() or digest(artifact) != artifact_sha256:
        raise ValueError("model artifact differs from the trusted SHA-256 or is a symlink")
    config = model / "config.json"
    if config.is_symlink() or digest(config) != config_sha256:
        raise ValueError("model config differs from the trusted SHA-256 or is a symlink")
    tokenizer = model / "tokenizer.json"
    if tokenizer.is_symlink() or digest(tokenizer) != QWEN_TOKENIZER_SHA256:
        raise ValueError("model tokenizer differs from the pinned SHA-256 or is a symlink")
    sm = TARGETS[target][2]
    pack_id = f"pih.kernels.qwen3.sm{sm}"
    pack_file = f"lib/pih_kernels_qwen3_sm{sm}.so"
    plugins = []
    for identity, file in (("pih.backend.nvidia-cuda", "lib/pih_plugin_backend_nvidia_cuda.so"),
                           ("pih.execution.default", "lib/pih_plugin_execution_default.so"),
                           ("pih.model.qwen3", "lib/pih_plugin_model_qwen3.so"),
                           ("pih.platform.linux", "lib/pih_plugin_platform_linux.so"),
                           ("pih.storage.verified-artifact", "lib/pih_plugin_storage_verified_artifact.so"),
                           ("pih.surface.text-http", "lib/pih_plugin_surface_text_http.so")):
        plugins.append({"plugin_id": identity, "plugin_version": "1.0.0", "entrypoint": file,
                        "entrypoint_sha256_hex": digest(bundle / file)})
    capabilities = [{"capability_id": capability, "provider_id": provider, "contract_id": contract,
                     "threading_model": threading, "scope": scope, "cardinality": 1}
                    for capability, provider, contract, threading, scope in (
                        ("artifact.authenticated-snapshot.v1", "pih.storage.verified-artifact", "pih.artifact.authenticated-snapshot.v1", 2, 2),
                        ("device.cuda-async.v1", "pih.backend.nvidia-cuda", "pih.device.cuda-async.v1", 2, 1),
                        ("device.cuda-memory.v1", "pih.backend.nvidia-cuda", "pih.device.cuda-memory.v1", 2, 1),
                        ("device.cuda-resources.v1", "pih.backend.nvidia-cuda", "pih.device.cuda-resources.v1", 2, 1),
                        ("device.cuda-runtime.v1", "pih.backend.nvidia-cuda", "pih.device.cuda-runtime.v1", 2, 1),
                        ("execution.default.v1", "pih.execution.default", "pih.execution.default.v1", 3, 2),
                        ("execution.controller.v1", "pih.execution.default", "pih.execution.controller.v1", 3, 2),
                        ("inference.text.v2", "pih.model.qwen3", "pih.inference.text.v2", 2, 2),
                        (pack_id, pack_id, "pih.qwen-kernels.v1", 3, 2),
                        ("platform.linux.v1", "pih.platform.linux", "pih.platform.linux.v1", 2, 1),
                        ("surface.text-http.v2", "pih.surface.text-http", "pih.surface.text-http.v2", 2, 2))]
    document = {"schema": "pih.development-lock.v1", "support_status": "unsupported",
        "kernel_packs": [{"pack_id": pack_id, "pack_version": "1.0.0", "pack_abi": "pih.qwen-kernels.v1",
            "architecture": f"sm_{sm}", "binary": pack_file, "binary_sha256_hex": digest(bundle / pack_file)}],
        "plugins": plugins, "capabilities": sorted(capabilities, key=lambda item: item["capability_id"]),
        "engine": {"capability_id": "inference.text.v2", "contract_id": "pih.inference.text.v2",
            "activation_epoch": 1, "configuration": {"model_directory": str(model), "precision": precision,
            "maximum_context_tokens": context, "generation_timeout_ms": generation_timeout_ms,
            "artifact_sha256": artifact_sha256, "config_sha256": config_sha256}, "smoke_http_body": {}}}
    # Existing native Lock parsing is intentionally ordered, not sort_keys JSON.
    encoded = (json.dumps(document, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    lock = bundle / f"qwen3-{target}-{precision}-{hashlib.sha256(encoded).hexdigest()[:16]}.lock"
    return publish_lock(lock, encoded)


def probe_response_is_complete(response, model, max_tokens):
    """Validate the observation envelope, not numerical correctness or quality."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate response field")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite response number")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite response number")
        return result

    try:
        value = json.loads(response, object_pairs_hook=unique_object,
                           parse_constant=reject_constant, parse_float=finite_float)
        if not isinstance(value, dict) or "error" in value or value.get("model") != model:
            return False
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return False
        choice = choices[0]
        if (not isinstance(choice, dict) or type(choice.get("index")) is not int or
                choice["index"] != 0 or choice.get("finish_reason") not in ("stop", "length")):
            return False
        message, usage = choice.get("message"), value.get("usage")
        if (not isinstance(message, dict) or message.get("role") != "assistant" or
                not isinstance(message.get("content"), str) or not message["content"].strip() or
                not isinstance(usage, dict)):
            return False
        if any(type(usage.get(key)) is not int for key in
               ("prompt_tokens", "completion_tokens", "total_tokens")):
            return False
        return (usage["prompt_tokens"] > 0 and
                0 < usage["completion_tokens"] <= max_tokens and
                usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"])
    except (ValueError, TypeError, RecursionError):
        return False


def probe(args):
    if args.output.exists():
        raise ValueError("probe output already exists; choose a new file")
    if not 1 <= args.max_tokens <= 4096 or not 0 < args.timeout <= 3600:
        raise ValueError("invalid probe token/timeout limit")
    body = {"model": args.model, "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens, "temperature": 0, "top_p": 1, "stream": False}
    request = urllib.request.Request(args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.monotonic()
    status, response = 0, ""
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as stream:
            status = stream.status
            payload = stream.read(8 * 1024 * 1024 + 1)
            if len(payload) > 8 * 1024 * 1024:
                raise ValueError("probe response exceeds 8 MiB")
            response = payload.decode("utf-8")
    except urllib.error.HTTPError as error:
        status, response = error.code, error.read(1024 * 1024).decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        response = str(error)
    success = status == 200 and probe_response_is_complete(response, body["model"], args.max_tokens)
    receipt = {"schema": "pih.native-service-observation.v1", "url": args.url, "request": body,
        "http_status": status, "elapsed_seconds": time.monotonic() - start,
        "response": response, "nonempty_response": success, "qualification": "not_established"}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(response)
    print(f"observation saved: {args.output}")
    return 0 if success else 1


def generate_deepseek_tokens(args):
    """Invoke the native token contract; never import a model or tokenizer."""
    inspect_host("rtx4090d")
    if not 1 <= args.max_tokens <= 65536 or not 1 <= args.jobs <= 128:
        raise ValueError("max-tokens must be 1..65536 and jobs 1..128")
    encoded = args.prompt_tokens
    if len(encoded) > 512 * 1024 or not encoded or any(
        not part.isascii() or not part.isdecimal() or int(part) >= 129280
        for part in encoded.split(",")
    ) or len(encoded.split(",")) > 65536:
        raise ValueError("prompt-tokens must be comma-separated IDs in 0..129279")
    build_dir = args.build_dir.resolve()
    if args.build:
        if not args.artifact_dir or not args.artifact_root:
            raise ValueError("--build requires --artifact-dir and --artifact-root")
        artifact = args.artifact_dir.resolve(strict=True)
        if not artifact.is_dir() or artifact == ROOT or ROOT in artifact.parents:
            raise ValueError("verified generation must be outside the repository")
        if len(args.artifact_root) != 64 or set(args.artifact_root) - set("0123456789abcdef") or not args.artifact_root.strip("0"):
            raise ValueError("artifact-root must be a nonzero canonical SHA-256")
        run(["cmake", "--preset", "deepseek-4090d-pp1", "-S", ROOT, "-B", build_dir,
             f"-DPIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256={args.artifact_root}",
             f"-DPIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY={artifact}"])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs,
             "--target", "pih_deepseek_4090d_pp1_bundle"])
        bundle = build_dir / "bundle/deepseek-4090d-pp1"
        run(["cmake", f"-DPIH_BUNDLE_ROOT={bundle}", f"-DPIH_ARTIFACT_DIRECTORY={artifact}",
             f"-DPIH_ARTIFACT_ROOT_SHA256={args.artifact_root}", "-P", ROOT / "cmake/PIHStageDeepSeekPp1Artifact.cmake"])
    else:
        bundle = build_dir / "bundle/deepseek-4090d-pp1"
    lock = bundle / "deepseek-v4-flash-rtx4090d-pp1.lock"
    # Do not rewrite hashes or repair old locks. Missing/new capabilities must
    # be resolved by rebuilding the entire bundle with its current manifest.
    document = json.loads(lock.read_text(encoding="utf-8"))
    if not any(item.get("capability_id") == "inference.tokens.v1" and
               item.get("provider_id") == "pih.model.deepseek-v4-flash" and
               item.get("contract_id") == "pih.inference.tokens.v1"
               for item in document.get("capabilities", [])):
        raise ValueError("bundle lacks the native generation capability; rebuild it")
    config = document["engine"]["configuration"]
    if len(encoded.split(",")) + args.max_tokens > config["attention_reserved_tokens_per_sequence"]:
        raise ValueError("prompt plus completion exceeds the sealed attention capacity")
    os.execv(str(bundle / "bin/pih-worker"), ["pih-worker", "--lock", str(lock),
              "--generate-tokens", encoded, "--max-tokens", str(args.max_tokens)])


def resolve_deepseek_service_artifact(args, build_dir, bundle):
    """Pin startup selection through native verification, not Python weight I/O."""
    def root(value):
        return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef") and bool(value.strip("0"))

    if args.store_dir is None:
        if args.pointer_root is not None or args.catalog_root is not None:
            raise ValueError("pointer-root/catalog-root require store-dir")
        if args.artifact_dir is None or not root(args.artifact_root):
            raise ValueError("direct startup requires artifact-dir and nonzero canonical artifact-root")
        return args.artifact_dir.resolve(strict=True), args.artifact_root
    if args.artifact_dir is not None or args.artifact_root is not None:
        raise ValueError("store startup cannot also specify an explicit artifact directory/root")
    if not root(args.pointer_root) or not root(args.catalog_root):
        raise ValueError("store startup requires independently expected pointer-root and catalog-root")
    if not args.store_dir.is_absolute() or ".." in args.store_dir.parts or not args.store_dir.is_dir():
        raise ValueError("store-dir must be an existing absolute directory without parent traversal")
    store = args.store_dir.resolve(strict=True)
    tool_build = args.artifact_tools_build_dir.resolve()
    for other in (ROOT.resolve(), build_dir, bundle, tool_build):
        if store == other or store in other.parents or other in store.parents:
            raise ValueError("store must be disjoint from repository, build, bundle and artifact-tool build trees")
    for other in (build_dir, bundle):
        if tool_build == other or tool_build in other.parents or other in tool_build.parents:
            raise ValueError("artifact-tool build must be disjoint from service build and bundle")
    if args.build:
        run(["cmake", "-S", ROOT, "-B", tool_build, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF", "-DPIH_BUILD_MONOLITH=OFF",
             "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF", "-DPIH_ENABLE_CUDA=OFF",
             "-DPIH_ENABLE_NCCL=OFF", "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=ON"])
        run(["cmake", "--build", tool_build, "--parallel", args.jobs, "--target", "pih-deepseek-generation-store"])
    command = [tool_build / "plugins/offline-deepseek/pih-deepseek-generation-store", "resolve",
               args.store_dir, args.pointer_root, args.catalog_root]
    print(json.dumps({"command": [str(value) for value in command]}), flush=True)
    result = subprocess.run([str(value) for value in command], stdout=subprocess.PIPE,
                            text=True, encoding="utf-8", check=True)
    if len(result.stdout) > 16384:
        raise ValueError("native active-generation observation exceeds its output bound")
    observed = json.loads(result.stdout)
    fields = {"schema", "pointer_root", "catalog_root", "artifact_root", "receipt_root", "generation_name",
              "activation_ordinal", "receipt_binding_verified", "source_payload_equivalence", "immutable_admission", "model_execution"}
    if (not isinstance(observed, dict) or set(observed) != fields
            or observed["schema"] != "pih.deepseek_active_generation_observation.v1"
            or observed["pointer_root"] != args.pointer_root or observed["catalog_root"] != args.catalog_root
            or not root(observed["artifact_root"]) or not root(observed["receipt_root"])
            or observed["generation_name"] != "sha256-" + observed["artifact_root"]
            or type(observed["activation_ordinal"]) is not int or not 1 <= observed["activation_ordinal"] <= (1 << 63) - 1
            or observed["receipt_binding_verified"] is not True
            or any(observed[name] is not False for name in ("source_payload_equivalence", "immutable_admission", "model_execution"))):
        raise ValueError("native active-generation observation does not match startup authority")
    print(json.dumps({"startup_selection": observed, "worker_storage_admission_required": True}), flush=True)
    return args.store_dir / "generations" / observed["generation_name"], observed["artifact_root"]


def serve_deepseek(args):
    """Compose only native PP1 providers and the text surface; no diagnostic bridge."""
    if not 1 <= args.generation_timeout_ms <= 86400000:
        raise ValueError("generation-timeout-ms must be 1..86400000")
    inspect_host("rtx4090d")
    if not 1 <= args.jobs <= 128 or not 1 <= args.port <= 65535:
        raise ValueError("jobs must be 1..128 and port 1..65535")
    if not 2 <= args.max_context <= 4096:
        raise ValueError("this development profile admits max-context 2..4096; capacity remains unqualified")
    build_dir, bundle = validate_deployment_paths(args.build_dir, args.bundle)
    if args.build and bundle.exists():
        raise ValueError("build requires a new bundle directory; existing files are never overwritten")
    semantic = Path(os.path.abspath(args.semantic_snapshot))
    semantic_resolved = semantic.resolve(strict=True)
    if not semantic.is_dir():
        raise ValueError("semantic-snapshot must be an existing directory")
    protected_trees = [ROOT.resolve(), build_dir, bundle]
    if args.store_dir is not None:
        protected_trees.append(args.artifact_tools_build_dir.resolve())
    for other in protected_trees:
        if semantic_resolved == other or semantic_resolved in other.parents or other in semantic_resolved.parents:
            raise ValueError("semantic snapshot must be disjoint from repository, build and bundle trees")
    artifact, artifact_root = resolve_deepseek_service_artifact(args, build_dir, bundle)
    artifact_resolved = artifact.resolve(strict=True)
    if (artifact_resolved == semantic_resolved or
            artifact_resolved in semantic_resolved.parents or
            semantic_resolved in artifact_resolved.parents):
        raise ValueError("weight generation and semantic snapshot must be disjoint")
    for directory in (artifact, semantic):
        if not directory.is_dir():
            raise ValueError("weight generation and semantic snapshot must exist as directories")
        resolved = directory.resolve(strict=True)
        for other in protected_trees:
            if resolved == other or resolved in other.parents or other in resolved.parents:
                raise ValueError("weight generation and semantic snapshot must be disjoint from source/build/bundle trees")
    base = json.loads((ROOT / "deploy/development/deepseek-v4-flash-rtx4090d-pp1.lock").read_text(encoding="utf-8"))
    excluded = {"pih.surface.openai-http", "pih.surface.health"}
    base["plugins"] = [item for item in base["plugins"] if item["plugin_id"] not in excluded]
    base["plugins"].append({"plugin_id": "pih.surface.text-http", "plugin_version": "1.0.0",
                            "entrypoint": "lib/pih_plugin_surface_text_http.so"})
    if args.build:
        identities = [item["plugin_id"] for item in base["plugins"]] + ["pih.kernels.deepseek-v4.sm89"]
        run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
             f"-DCMAKE_INSTALL_PREFIX={bundle}", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=ON", "-DPIH_BUILD_PLUGINS=ON", "-DPIH_BUILD_MONOLITH=OFF",
             "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF", "-DPIH_ENABLE_CUDA=ON",
             "-DPIH_ENABLE_NCCL=OFF", "-DPIH_BUILD_TOKENIZER_TOOLS=OFF",
             "-DPIH_ENABLED_PLUGINS=" + ";".join(identities)])
        targets = [Path(item["entrypoint"]).stem for item in base["plugins"]]
        run(["cmake", "--build", build_dir, "--parallel", args.jobs, "--target",
             "pih-worker", "pih-cli", "pih_kernels_deepseek_v4_sm89", *targets])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs, "--target", "pih-release"])
        run(["cmake", "--install", build_dir, "--component", "pih-release"])
    bundle = bundle.resolve(strict=True)
    run(["cmake", f"-DPIH_BUNDLE_ROOT={bundle}", f"-DPIH_ARTIFACT_DIRECTORY={artifact}",
         f"-DPIH_ARTIFACT_ROOT_SHA256={artifact_root}", "-P", ROOT / "cmake/PIHStageDeepSeekPp1Artifact.cmake"])
    for plugin in base["plugins"]:
        plugin["entrypoint_sha256_hex"] = digest(bundle / plugin["entrypoint"])
    for pack in base["kernel_packs"]:
        pack["binary_sha256_hex"] = digest(bundle / pack["binary"])
    base["capabilities"] = [item for item in base["capabilities"] if item["provider_id"] not in excluded]
    if not any(item["capability_id"] == "inference.text.v2" for item in base["capabilities"]):
        raise ValueError("source template is missing the DeepSeek native text contract")
    base["capabilities"].append({"capability_id": "surface.text-http.v2", "provider_id": "pih.surface.text-http",
        "contract_id": "pih.surface.text-http.v2", "threading_model": 2, "scope": 2, "cardinality": 1})
    base["capabilities"].sort(key=lambda item: item["capability_id"])
    native = base["engine"]["configuration"]
    native["artifact_root_sha256_hex"] = artifact_root
    native["attention_reserved_tokens_per_sequence"] = args.max_context
    base["engine"] = {"capability_id": "inference.text.v2", "contract_id": "pih.inference.text.v2",
        "activation_epoch": 1, "configuration": {"schema": "pih.deepseek-v4-flash.text.v1",
            "deployment_root": str(bundle), "semantic_snapshot": str(semantic),
            "generation_timeout_ms": args.generation_timeout_ms,
            "engine_configuration": json.dumps(native, ensure_ascii=True, separators=(",", ":"))},
        "smoke_http_body": {}}
    encoded = (json.dumps(base, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
    lock = bundle / f"deepseek-text-{hashlib.sha256(encoded).hexdigest()[:16]}.lock"
    publish_lock(lock, encoded)
    print(f"native DeepSeek text Lock: {lock}", flush=True)
    os.execv(str(bundle / "bin/pih-worker"), ["pih-worker", "--lock", str(lock), "--serve", str(args.port)])


def verify_deepseek_semantics(args):
    """Build/run native admission only; no Python tokenization or inference."""
    if platform.system() != "Linux":
        raise ValueError("native semantic admission requires Linux; no symlink-following fallback")
    if not 1 <= args.jobs <= 128:
        raise ValueError("jobs must be 1..128")
    # Preserve path components for native O_NOFOLLOW traversal. resolve() here
    # would erase evidence that the supplied snapshot path contained symlinks.
    snapshot = Path(os.path.abspath(args.snapshot_dir))
    if not snapshot.is_dir():
        raise ValueError("snapshot-dir must be an existing directory")
    build_dir = args.build_dir.resolve()
    if build_dir == snapshot or snapshot in build_dir.parents:
        raise ValueError("build directory must not be inside the semantic snapshot")
    if args.build:
        run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF",
             "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF",
             "-DPIH_ENABLE_CUDA=OFF", "-DPIH_ENABLE_NCCL=OFF", "-DPIH_BUILD_TOKENIZER_TOOLS=ON"])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs, "--target",
             "pih-deepseek-semantic-verify", "pih-tokenize", "pih-deepseek-format"])
    run([build_dir / "plugins/common/pih-deepseek-semantic-verify", snapshot])
    return 0


def verify_deepseek_artifact(args):
    """Offline full-file observation, not immutable storage admission or inference."""
    if platform.system() != "Linux":
        raise ValueError("native artifact verification requires Linux")
    if not 1 <= args.jobs <= 128:
        raise ValueError("jobs must be 1..128")
    if len(args.artifact_root) != 64 or any(c not in "0123456789abcdef" for c in args.artifact_root) or not int(args.artifact_root, 16):
        raise ValueError("artifact-root must be a nonzero lowercase SHA-256 from a trusted publication")
    if ".." in args.artifact_dir.parts:
        raise ValueError("artifact-dir must not contain parent traversal")
    artifact = Path(os.path.abspath(args.artifact_dir))
    if not artifact.is_dir():
        raise ValueError("artifact-dir must exist")
    build_dir = args.build_dir.resolve()
    resolved_artifact = artifact.resolve(strict=True)
    if build_dir == resolved_artifact or resolved_artifact in build_dir.parents:
        raise ValueError("build directory must not be inside the artifact generation")
    roots = [args.model_root, args.semantic_root, args.inventory_root,
             args.payload_root, args.converter_identity_root]
    native_args = [artifact, args.artifact_root]
    if args.source_dir is None:
        if args.verification_projection:
            raise ValueError("--verification-projection requires --source-dir and all authority roots")
        if any(root is not None for root in roots):
            raise ValueError("source authority roots require --source-dir")
    else:
        if any(root is None or len(root) != 64 or any(c not in "0123456789abcdef" for c in root)
               or not int(root, 16) for root in roots):
            raise ValueError("source-bound verification requires all five nonzero lowercase authority roots")
        if not args.source_dir.is_absolute() or ".." in args.source_dir.parts or not args.source_dir.is_dir():
            raise ValueError("source-dir must be an existing absolute directory without parent traversal")
        source = args.source_dir.resolve(strict=True)
        if source == resolved_artifact or source in resolved_artifact.parents or resolved_artifact in source.parents:
            raise ValueError("source and target artifact directories must be disjoint")
        for directory in (source, resolved_artifact):
            if build_dir == directory or directory in build_dir.parents or build_dir in directory.parents:
                raise ValueError("build directory must be disjoint from source and target")
        # Preserve the lexical source path for native no-follow traversal.
        mode = "--source-bound-projection" if args.verification_projection else "--source-bound"
        native_args = [mode, args.source_dir, artifact, *roots, args.artifact_root]
    if args.build:
        run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF",
             "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF",
             "-DPIH_ENABLE_CUDA=OFF", "-DPIH_ENABLE_NCCL=OFF",
             "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=ON"])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs, "--target",
             "pih-deepseek-artifact-verify"])
    run([build_dir / "plugins/offline-deepseek/pih-deepseek-artifact-verify", *native_args])
    return 0


def deepseek_generation_store(args):
    """Native store structure only; no Python publication or inference."""
    if platform.system() != "Linux":
        raise ValueError("native generation stores require Linux")
    if not 1 <= args.jobs <= 128:
        raise ValueError("jobs must be 1..128")
    path = args.store_dir
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("store-dir must be absolute without parent traversal")
    if args.operation == "init" and (path.exists() or path.is_symlink()):
        raise ValueError("store-dir already exists; partial stores are never overwritten")
    if args.operation in ("verify", "commit", "activate", "resolve") and not path.is_dir():
        raise ValueError("store-dir must already exist")
    resolved = path.resolve()
    repository = ROOT.resolve()
    if resolved == repository or repository in resolved.parents or resolved in repository.parents:
        raise ValueError("store must be outside and not contain the repository")
    build_dir = args.build_dir.resolve()
    if build_dir == resolved or resolved in build_dir.parents or build_dir in resolved.parents:
        raise ValueError("store and build directory must be disjoint")
    roots = [args.model_root, args.semantic_root, args.inventory_root, args.payload_root,
             args.converter_identity_root, args.artifact_root]
    native_args = [args.operation, path]
    activation_args = [args.receipt_root, args.catalog_root, args.activation_ordinal, args.previous_pointer_root]
    if args.operation != "resolve" and args.pointer_root is not None:
        raise ValueError("pointer-root applies only to resolve")
    if args.operation == "resolve":
        if args.source_dir is not None or args.staging_name is not None or any(value is not None for value in roots + [args.receipt_root, args.activation_ordinal, args.previous_pointer_root]):
            raise ValueError("resolve accepts only store, pointer-root and catalog-root authority")
        for value in (args.pointer_root, args.catalog_root):
            if value is None or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or not int(value, 16):
                raise ValueError("resolve requires independently expected pointer-root and catalog-root")
        native_args = ["resolve", path, args.pointer_root, args.catalog_root]
    elif args.operation in ("commit", "activate"):
        if args.source_dir is None or not args.source_dir.is_absolute() or ".." in args.source_dir.parts or not args.source_dir.is_dir():
            raise ValueError("commit/activate requires an existing absolute source-dir without parent traversal")
        if args.operation == "commit" and (not args.staging_name or args.staging_name in (".", "..") or any(c in args.staging_name for c in "/\\\0")):
            raise ValueError("commit requires a single staging-name beneath store/.staging")
        if any(root is None or len(root) != 64 or any(c not in "0123456789abcdef" for c in root)
               or not int(root, 16) for root in roots):
            raise ValueError("commit/activate requires all six nonzero lowercase authority/artifact roots")
        source = args.source_dir.resolve(strict=True)
        for other in (resolved, build_dir):
            if source == other or source in other.parents or other in source.parents:
                raise ValueError("source must be disjoint from store and build directory")
        if args.operation == "commit":
            if any(value is not None for value in activation_args):
                raise ValueError("activation authority arguments apply only to activate")
            native_args = ["commit", args.source_dir, path, args.staging_name, *roots]
        else:
            if args.staging_name is not None:
                raise ValueError("activate does not accept staging-name")
            for value in (args.receipt_root, args.catalog_root):
                if value is None or len(value) != 64 or any(c not in "0123456789abcdef" for c in value) or not int(value, 16):
                    raise ValueError("activate requires nonzero receipt-root and independently admitted catalog-root")
            if args.activation_ordinal is None or not 1 <= args.activation_ordinal <= (1 << 63) - 1:
                raise ValueError("activation-ordinal must be 1..INT64_MAX")
            previous = args.previous_pointer_root
            if previous != "none" and (previous is None or len(previous) != 64 or any(c not in "0123456789abcdef" for c in previous) or not int(previous, 16)):
                raise ValueError("previous-pointer-root must be explicit none or a nonzero digest")
            if (previous == "none") != (args.activation_ordinal == 1):
                raise ValueError("previous-pointer-root is none exactly for the first activation")
            native_args = ["activate", args.source_dir, path, *roots, *activation_args]
    elif args.source_dir is not None or args.staging_name is not None or any(root is not None for root in roots + activation_args):
        raise ValueError("source/staging/authority arguments apply only to commit or activate")
    if args.build:
        run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF",
             "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF",
             "-DPIH_ENABLE_CUDA=OFF", "-DPIH_ENABLE_NCCL=OFF",
             "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=ON"])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs,
             "--target", "pih-deepseek-generation-store"])
    run([build_dir / "plugins/offline-deepseek/pih-deepseek-generation-store", *native_args])
    return 0


def prepare_deepseek_artifact(args):
    """Build/run the native converter; Python performs no weight processing."""
    if platform.system() != "Linux":
        raise ValueError("native artifact preparation requires Linux")
    if not 1 <= args.jobs <= 128:
        raise ValueError("jobs must be 1..128")
    roots = [args.model_root, args.semantic_root, args.inventory_root,
             args.payload_root, args.converter_identity_root]
    if any(len(root) != 64 or any(c not in "0123456789abcdef" for c in root)
           or not int(root, 16) for root in roots):
        raise ValueError("all authority roots must be nonzero lowercase SHA-256 values")
    paths = [args.source_dir, args.staging_dir]
    if any(not path.is_absolute() or ".." in path.parts for path in paths):
        raise ValueError("source/staging must be absolute paths without parent traversal")
    resolved = [path.resolve(strict=True) for path in paths]
    source, staging = resolved
    if any(not path.is_dir() for path in paths):
        raise ValueError("source and staging directories must already exist")
    if source == staging or source in staging.parents or staging in source.parents:
        raise ValueError("source and staging must be disjoint")
    if staging == ROOT.resolve() or ROOT.resolve() in staging.parents or staging in ROOT.resolve().parents:
        raise ValueError("staging must be outside and not contain the repository")
    if any(args.staging_dir.iterdir()):
        raise ValueError("staging must be empty; failed conversion files are not automatically removed")
    build_dir = args.build_dir.resolve()
    for directory in resolved:
        if build_dir == directory or directory in build_dir.parents or build_dir in directory.parents:
            raise ValueError("build directory must be disjoint from source and staging")
    if args.build:
        run(["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja",
             "-DCMAKE_BUILD_TYPE=Release", "-DPIH_DEPLOYMENT_PROFILE=custom",
             "-DPIH_BUILD_WORKER=OFF", "-DPIH_BUILD_PLUGINS=OFF",
             "-DPIH_BUILD_MONOLITH=OFF", "-DPIH_BUILD_PYTHON=OFF", "-DPIH_BUILD_TESTS=OFF",
             "-DPIH_ENABLE_CUDA=OFF", "-DPIH_ENABLE_NCCL=OFF",
             "-DPIH_BUILD_TOKENIZER_TOOLS=OFF", "-DPIH_BUILD_ARTIFACT_TOOLS=ON"])
        run(["cmake", "--build", build_dir, "--parallel", args.jobs,
             "--target", "pih-deepseek-artifact-prepare"])
    # Keep lexical paths: the native tool must reject symlink components itself.
    run([build_dir / "plugins/offline-deepseek/pih-deepseek-artifact-prepare",
         args.source_dir, args.staging_dir, *roots])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    deepseek = commands.add_parser("serve-deepseek")
    deepseek.add_argument("--build", action="store_true")
    deepseek.add_argument("--build-dir", type=Path, default=ROOT / "out/build/deepseek-text")
    deepseek.add_argument("--bundle", type=Path, default=ROOT / "out/install/deepseek-text")
    artifact_source = deepseek.add_mutually_exclusive_group(required=True)
    artifact_source.add_argument("--artifact-dir", type=Path)
    artifact_source.add_argument("--store-dir", type=Path)
    deepseek.add_argument("--artifact-root")
    deepseek.add_argument("--pointer-root")
    deepseek.add_argument("--catalog-root")
    deepseek.add_argument("--artifact-tools-build-dir", type=Path, default=ROOT / "out/build/deepseek-artifact")
    deepseek.add_argument("--semantic-snapshot", type=Path, required=True)
    deepseek.add_argument("--max-context", type=int, default=4096)
    deepseek.add_argument("--generation-timeout-ms", type=int, default=600000,
                          help="request preparation/generation deadline, 1..86400000 ms")
    deepseek.add_argument("--port", type=int, default=8000)
    deepseek.add_argument("--jobs", type=int, default=4)
    semantic = commands.add_parser("verify-deepseek-semantics")
    semantic.add_argument("--snapshot-dir", type=Path, required=True)
    semantic.add_argument("--build", action="store_true")
    semantic.add_argument("--build-dir", type=Path, default=ROOT / "out/build/tokenizer")
    semantic.add_argument("--jobs", type=int, default=4)
    artifact = commands.add_parser("verify-deepseek-artifact")
    artifact.add_argument("--artifact-dir", type=Path, required=True)
    artifact.add_argument("--artifact-root", required=True)
    artifact.add_argument("--source-dir", type=Path, help="also verify source-to-target payload equivalence")
    artifact.add_argument("--verification-projection", action="store_true",
                          help="emit existing-format canonical projection; requires source-bound verification")
    for name in ("model-root", "semantic-root", "inventory-root", "payload-root", "converter-identity-root"):
        artifact.add_argument("--" + name, help="required with --source-dir; independently trusted authority")
    artifact.add_argument("--build", action="store_true")
    artifact.add_argument("--build-dir", type=Path, default=ROOT / "out/build/deepseek-artifact")
    artifact.add_argument("--jobs", type=int, default=4)
    prepare = commands.add_parser("prepare-deepseek-artifact")
    store = commands.add_parser("deepseek-generation-store")
    store.add_argument("operation", choices=("init", "verify", "commit", "activate", "resolve"))
    store.add_argument("--store-dir", type=Path, required=True)
    store.add_argument("--build", action="store_true")
    store.add_argument("--build-dir", type=Path, default=ROOT / "out/build/deepseek-artifact")
    store.add_argument("--jobs", type=int, default=4)
    store.add_argument("--source-dir", type=Path)
    store.add_argument("--staging-name")
    store.add_argument("--receipt-root")
    store.add_argument("--catalog-root")
    store.add_argument("--pointer-root")
    store.add_argument("--activation-ordinal", type=int)
    store.add_argument("--previous-pointer-root", help="none for first activation; otherwise expected current pointer root")
    for name in ("model-root", "semantic-root", "inventory-root", "payload-root", "converter-identity-root", "artifact-root"):
        store.add_argument("--" + name)
    prepare.add_argument("--source-dir", type=Path, required=True)
    prepare.add_argument("--staging-dir", type=Path, required=True)
    for name in ("model-root", "semantic-root", "inventory-root", "payload-root", "converter-identity-root"):
        prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--build", action="store_true")
    prepare.add_argument("--build-dir", type=Path, default=ROOT / "out/build/deepseek-artifact")
    prepare.add_argument("--jobs", type=int, default=4)
    generation = commands.add_parser("generate-deepseek-tokens")
    generation.add_argument("--build", action="store_true")
    generation.add_argument("--build-dir", type=Path, default=ROOT / "out/build/deepseek-4090d-pp1")
    generation.add_argument("--artifact-dir", type=Path)
    generation.add_argument("--artifact-root")
    generation.add_argument("--jobs", type=int, default=4)
    generation.add_argument("--prompt-tokens", required=True)
    generation.add_argument("--max-tokens", type=int, default=128)
    metadata = commands.add_parser("prepare-qwen-metadata", help="copy authenticated config/tokenizer; no inference")
    metadata.add_argument("--source-dir", type=Path, required=True)
    metadata.add_argument("--model-dir", type=Path, required=True)
    metadata.add_argument("--config-sha256", required=True)
    conversion = commands.add_parser("convert-int4")
    conversion.add_argument("--bundle", type=Path, default=ROOT / "out/install/qwen3-native")
    conversion.add_argument("--source", type=Path, required=True)
    conversion.add_argument("--expectation", type=Path, required=True)
    conversion.add_argument("--output", type=Path, required=True)
    conversion.add_argument("--prepare-metadata", action="store_true",
                            help="copy pinned tokenizer and authenticated config from the source parent before conversion")
    conversion.add_argument("--config-sha256",
                            help="independently trusted config digest; required with --prepare-metadata")
    conversion.add_argument("--build", action="store_true", help="build the CPU-only converter first")
    conversion.add_argument("--build-dir", type=Path, default=ROOT / "out/build/qwen3-artifact")
    conversion.add_argument("--jobs", type=int, default=4)
    verification = commands.add_parser("verify-int4", help="read-only native Qwen INT4 artifact verification")
    verification.add_argument("--artifact", type=Path, required=True)
    for name in ("file-sha256", "source-root", "binding-root", "disposition-root"):
        verification.add_argument("--" + name, required=True)
    verification.add_argument("--bundle", type=Path, default=ROOT / "out/install/qwen3-native")
    verification.add_argument("--build", action="store_true", help="build CPU artifact tools first")
    verification.add_argument("--build-dir", type=Path, default=ROOT / "out/build/qwen3-artifact")
    verification.add_argument("--jobs", type=int, default=4)
    source_check = commands.add_parser("verify-qwen-source", help="read-only native BF16 source verification")
    source_check.add_argument("--source", type=Path, required=True)
    source_check.add_argument("--expectation", type=Path, required=True)
    source_check.add_argument("--bundle", type=Path, default=ROOT / "out/install/qwen3-native")
    source_check.add_argument("--build", action="store_true")
    source_check.add_argument("--build-dir", type=Path, default=ROOT / "out/build/qwen3-artifact")
    source_check.add_argument("--jobs", type=int, default=4)
    for command in ("build", "serve", "seal"):
        sub = commands.add_parser(command)
        sub.add_argument("--build-dir", type=Path, default=ROOT / "out/build/qwen3-native")
        sub.add_argument("--bundle", type=Path, default=ROOT / "out/install/qwen3-native")
        if command in ("build", "serve"):
            sub.add_argument("--jobs", type=int, default=4)
        if command == "build":
            sub.add_argument("--target", choices=TARGETS, required=True)
        if command != "build":
            sub.add_argument("--model-dir", type=Path, required=True)
            sub.add_argument("--artifact-sha256", required=True,
                             help="independently trusted SHA-256 of model.safetensors or model.xing-int4")
            sub.add_argument("--config-sha256", required=True,
                             help="independently trusted SHA-256 of model config.json")
            sub.add_argument("--precision", choices=("bf16", "int4"), default="bf16")
            sub.add_argument("--target", choices=TARGETS, required=True)
            sub.add_argument("--max-context", type=int, default=4096)
            sub.add_argument("--generation-timeout-ms", type=int, default=600000,
                             help="Qwen request preparation/generation deadline, 1..86400000 ms")
        if command == "serve":
            sub.add_argument("--build", action="store_true")
            sub.add_argument("--port", type=int, default=8000)
    p = commands.add_parser("probe")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", choices=("Qwen/Qwen3-0.6B", "deepseek-ai/DeepSeek-V4.1-Flash"),
                   default="Qwen/Qwen3-0.6B")
    p.add_argument("--prompt", default="请用一句话解释什么是张量并行。")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            print("qwen3-0.6b: Linux RTX 4090 D / H100 PCIe; native C ABI model + kernel pack + HTTP; BF16/INT4; unqualified")
            print("deepseek-v4-0731: Linux RTX 4090 D PP1; native token/text APIs and SSE wired; requires verified weights and pinned semantic snapshot; unqualified")
            print("deepseek-v41-b300: native text plugin + SM103 rank pack; use deploy/run-v41-native.sh build-service/seal-rank/seal-service/serve; unqualified")
            return 0
        if args.command == "probe":
            return probe(args)
        if args.command == "serve-deepseek":
            return serve_deepseek(args)
        if args.command == "verify-deepseek-semantics":
            return verify_deepseek_semantics(args)
        if args.command == "verify-deepseek-artifact":
            return verify_deepseek_artifact(args)
        if args.command == "prepare-deepseek-artifact":
            return prepare_deepseek_artifact(args)
        if args.command == "deepseek-generation-store":
            return deepseek_generation_store(args)
        if args.command == "generate-deepseek-tokens":
            generate_deepseek_tokens(args)
            return 0
        if args.command == "prepare-qwen-metadata":
            return prepare_qwen_metadata(args)
        if args.command == "convert-int4":
            if platform.system() != "Linux":
                raise ValueError("use the Linux native conversion tool")
            if args.prepare_metadata != (args.config_sha256 is not None):
                raise ValueError("--prepare-metadata and --config-sha256 must be supplied together")
            if args.prepare_metadata and (
                    not re.fullmatch(r"[0-9a-f]{64}", args.config_sha256) or
                    not args.config_sha256.strip("0")):
                raise ValueError("config-sha256 must be an independently trusted nonzero SHA-256")
            if args.output.is_symlink():
                raise ValueError("output must not be a symbolic link")
            output = args.output.resolve()
            if output == ROOT or ROOT in output.parents:
                raise ValueError("converted weights must be outside the repository")
            source = args.source.resolve(strict=True)
            expectation = args.expectation.resolve(strict=True)
            if not source.is_file() or not expectation.is_file() or not output.parent.is_dir():
                raise ValueError("source/expectation must be files and output parent must exist")
            if output.exists() or os.path.lexists(str(output) + ".staging"):
                raise ValueError("output and staging must not exist")
            if args.prepare_metadata and output.name != "model.xing-int4":
                raise ValueError("metadata preparation requires the runtime filename model.xing-int4")
            executable = qwen_artifact_tool(args, "pih-qwen-int4-convert", [source, expectation, output])
            if args.prepare_metadata:
                prepare_qwen_metadata(argparse.Namespace(
                    source_dir=source.parent, model_dir=output.parent,
                    config_sha256=args.config_sha256))
            run([executable, source, expectation, output])
            return 0
        if args.command == "verify-int4":
            return verify_qwen_int4(args)
        if args.command == "verify-qwen-source":
            source = args.source.resolve(strict=True)
            expectation = args.expectation.resolve(strict=True)
            if not source.is_file() or not expectation.is_file():
                raise ValueError("source and expectation must be regular files")
            executable = qwen_artifact_tool(args, "pih-qwen-source-verify", [source, expectation])
            run([executable, source, expectation])
            return 0
        args.build_dir, args.bundle = validate_deployment_paths(
            args.build_dir, args.bundle,
            args.model_dir if args.command in ("serve", "seal") else None)
        if args.command in ("build", "serve") and not 1 <= args.jobs <= 128:
            raise ValueError("jobs must be 1..128")
        if args.command == "build":
            build(args.build_dir, args.bundle, args.jobs, args.target)
            return 0
        if not 2 <= args.max_context <= 40960:
            raise ValueError("max-context must be 2..40960; capacity still requires target measurement")
        if not 1 <= args.generation_timeout_ms <= 86400000:
            raise ValueError("generation-timeout-ms must be 1..86400000")
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be 1..65535")
            inspect_host(args.target)
            if args.build:
                build(args.build_dir, args.bundle, args.jobs, args.target)
        lock = seal(args.bundle, args.model_dir, args.precision, args.target, args.max_context,
                    args.build_dir, args.generation_timeout_ms,
                    args.artifact_sha256, args.config_sha256)
        print(f"native Lock: {lock}", flush=True)
        if args.command == "serve":
            os.execv(str(args.bundle / "bin/pih-worker"), ["pih-worker", "--lock", str(lock), "--serve", str(args.port)])
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.exit(2, f"native deployment: {error}\n")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
