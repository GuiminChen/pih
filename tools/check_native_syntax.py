"""Linux-target C++20 syntax checks, without linking, CUDA compilation or tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "plugins/model-deepseek-v41/supervisor_request.cpp",
    "plugins/model-deepseek-v41/supervisor_execution.cpp",
    "plugins/model-deepseek-v41/entrypoint.cpp",
    "plugins/model-deepseek-v41/tokenizer.cpp",
    "plugins/common/bytelevel_tokenizer.cpp", "plugins/common/text_json.cpp",
    "plugins/model-qwen3/format_main.cpp",
    "plugins/model-qwen3/tokenizer.cpp",
    "plugins/common/deepseek_encoding.cpp", "plugins/common/deepseek_completion_decoder.cpp",
    "plugins/common/deepseek_semantic_artifact.cpp", "plugins/common/tokenize_main.cpp",
    "plugins/common/deepseek_format_main.cpp", "plugins/common/deepseek_semantic_main.cpp",
    "src/worker/main.cpp", "src/worker/worker_bootstrap.cpp",
    "plugins/surface-openai-http/http_request_framing.cpp",
    "plugins/surface-text-http/entrypoint.cpp", "plugins/model-deepseek-v4-flash/generation.cpp",
    "plugins/execution-default/controller_runtime.cpp", "src/core/bounded_json.cpp",
    "src/core/status.cpp", "src/core/sha256.cpp",
    "plugins/model-deepseek-v4-flash/entrypoint.cpp", "plugins/model-qwen3/entrypoint.cpp",
    "plugins/offline-qwen/convert_main.cpp",
    "plugins/offline-qwen/verify_main.cpp",
    "plugins/offline-qwen/verify_source_main.cpp",
    "plugins/model-deepseek-v4-flash/text_completion.cpp",
    "plugins/model-deepseek-v4-flash/cuda/nvidia_deepseek_shared_expert_operations.cpp",
    "plugins/model-deepseek-v4-flash/cuda/nvidia_deepseek_rank_runtime.cpp",
    "plugins/model-deepseek-v4-flash/cuda/nvidia_deepseek_engine_bootstrap.cpp",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--sysroot", type=Path, default=ROOT / "out/linux-syntax-sysroot")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--artifact-tools", action="store_true",
                           help="check the explicit DeepSeek offline, supervisor and artifact-tool source set, including the conversion command variant")
    selection.add_argument("--deepseek-model-build", type=Path,
                        help="also check the 161-unit native PP1 model target from this CMake compile_commands.json")
    selection.add_argument("--deepseek-model-source", action="store_true",
                           help="check the pinned 161-unit PP1 model source closure without a compilation database")
    selection.add_argument("--qwen-model", action="store_true",
                           help="check the explicit Qwen model and shared packed-plan value sources")
    selection.add_argument("--qwen-qualification", action="store_true",
                           help="check the separate Qwen CUDA fixture command without execution")
    selection.add_argument("--v41-model", action="store_true",
                           help="check every native V4.1 model and SM103 pack C++ translation unit")
    parser.add_argument("--deepseek-backend", action="store_true",
                        help="also check the exact 24 native PP1 CUDA-backend C++ adapters (no CUDA SDK or kernels)")
    parser.add_argument("--plugin-support", action="store_true",
                        help="also check execution, storage, host-spill and Linux platform plugin C++ sources")
    parser.add_argument("--cuda-headers", type=Path, default=ROOT / "out/cuda-syntax-headers-13.2.86")
    parser.add_argument("--cublas-headers", type=Path, default=ROOT / "out/cublas-syntax-headers-13.4.1.3")
    parser.add_argument("--nccl-headers", type=Path, default=ROOT / "out/nccl-syntax-headers-2.31.2")
    parser.add_argument("--generated-dir", type=Path,
                        help="CMake-generated include directory required for Qwen's version.cpp")
    args = parser.parse_args()
    if args.deepseek_backend and args.artifact_tools:
        parser.error("backend and artifact-tool source selections cannot be combined")
    sources = list(SOURCES)
    if args.v41_model:
        sources = []
        for owner in ("plugins/model-deepseek-v41", "plugins/kernels/deepseek-v41-sm103",
                      "plugins/transport-nccl"):
            directory = ROOT / owner
            for source in sorted(directory.glob("*.cpp")):
                if source.is_symlink() or not source.is_file():
                    parser.error(f"V4.1 source must be a regular file: {source}")
                sources.append(source.relative_to(ROOT).as_posix())
        required = {
            "plugins/model-deepseek-v41/entrypoint.cpp",
            "plugins/model-deepseek-v41/supervisor_execution.cpp",
            "plugins/model-deepseek-v41/worker_main.cpp",
            "plugins/model-deepseek-v41/rank_worker_loop.cpp",
            "plugins/model-deepseek-v41/inference_operation.cpp",
            "plugins/model-deepseek-v41/kernel_binding.cpp",
            "plugins/kernels/deepseek-v41-sm103/kernel_pack.cpp",
            "plugins/transport-nccl/entrypoint.cpp",
        }
        if len(sources) != 134 or not required.issubset(sources):
            parser.error("V4.1 native source closure is incomplete")
        for source in (ROOT / "plugins/model-deepseek-v41").glob("*.cpp"):
            content = source.read_text(encoding="utf-8")
            if re.search(r'#include\s*<nccl\.h>|\bnccl(?:Comm|AllReduce|AllGather|GetUniqueId)', content):
                parser.error(f"V4.1 model still calls NCCL directly: {source.name}")
        rank_wiring = (ROOT / "cmake/PIHSealV41.cmake").read_text(encoding="utf-8")
        deployment = (ROOT / "deploy/run-v41-native.sh").read_text(encoding="utf-8")
        if ("pih_plugin_transport_nccl.so" not in rank_wiring or
                "transport.collective.v1" not in rank_wiring or
                "pih_plugin_transport_nccl" not in deployment or
                "pih.transport.nccl" not in deployment):
            parser.error("V4.1 rank Lock or build script omits the transport plugin")
    if args.qwen_model:
        sources = ["plugins/model-qwen3/entrypoint.cpp", "plugins/model-qwen3/tokenizer.cpp",
                   "plugins/common/bytelevel_tokenizer.cpp"]
        qwen_build = (ROOT / "plugins/model-qwen3/CMakeLists.txt").read_text(encoding="utf-8")
        if "qwen_execution_sources" in qwen_build or "pih_execution_default_impl SOURCES" in qwen_build:
            parser.error("Qwen model still compiles the execution provider implementation")
        content = (ROOT / "plugins/model-qwen3/sources.cmake").read_text(encoding="utf-8")
        if "qwen_int4_cuda_fixture.cpp" in content:
            parser.error("Qwen production model includes a hardware-test-only fixture")
        names = re.findall(r'"\$\{CMAKE_CURRENT_LIST_DIR\}/([^"\s]+\.cpp)"', content)
        retired = {"qwen3_bf16_supervised_arena_assembly.cpp",
                   "qwen3_bf16_supervised_owner_manifest.cpp",
                   "qwen3_int4_supervised_arena_assembly.cpp",
                   "qwen3_int4_supervised_owner_manifest.cpp"}
        if len(names) != 183 or "controller_client.cpp" not in names or retired.intersection(names):
            parser.error("Qwen model source closure is incomplete or includes retired supervised code")
        for name in names:
            source = ROOT / "plugins/model-qwen3" / name
            if source.is_symlink() or not source.is_file():
                parser.error(f"missing or symlinked Qwen source: {name}")
            relative = source.relative_to(ROOT).as_posix()
            if relative not in sources:
                sources.append(relative)
        for name in ("packed_token_plan.cpp", "packed_token_metadata_arena.cpp"):
            source = ROOT / "plugins/execution-default" / name
            if source.is_symlink() or not source.is_file():
                parser.error(f"missing shared packed-plan value source: {name}")
            relative = source.relative_to(ROOT).as_posix()
            if relative not in sources:
                sources.append(relative)
    if args.qwen_qualification:
        sources = ["plugins/model-qwen3/qualification_main.cpp",
                   "src/backend/cuda/qwen_int4_cuda_fixture.cpp",
                   "plugins/model-qwen3/qwen3_teacher_forced_metric_oracle.cpp",
                   "src/backend/cuda/cuda_driver_status.cpp",
                   "src/backend/cuda/cuda_status.cpp"]
        cmake = (ROOT / "plugins/model-qwen3/CMakeLists.txt").read_text(encoding="utf-8")
        qualification = cmake.split("if(PIH_BUILD_QWEN_QUALIFICATION_TOOLS)", 1)[-1]
        if ("PIH_BUILD_QWEN_QUALIFICATION_TOOLS" not in cmake or
                "EXCLUDE_FROM_ALL" not in cmake or
                "qwen_int4_cuda_fixture.cpp" not in cmake or
                "pih_qwen3_native_impl" in qualification):
            parser.error("Qwen qualification command is missing its opt-in build boundary")
    if args.artifact_tools:
        sources = ["plugins/model-deepseek-v41/supervisor_request.cpp",
                   "plugins/model-deepseek-v41/supervisor_execution.cpp",
                   "plugins/model-deepseek-v41/entrypoint.cpp",
                   "plugins/model-deepseek-v41/supervisor_operation.cpp",
                   "plugins/model-deepseek-v41/supervisor_config.cpp",
                   "plugins/model-deepseek-v41/supervisor_main.cpp",
                   "plugins/model-deepseek-v41/supervisor_deployment.cpp",
                   "plugins/model-deepseek-v41/supervisor_request_policy.cpp",
                   "plugins/model-deepseek-v41/supervisor_output.cpp",
                   "plugins/model-deepseek-v41/tokenizer.cpp",
                   "plugins/common/bytelevel_tokenizer.cpp",
                   "plugins/common/tokenize_main.cpp",
                   "plugins/offline-deepseek/verify_main.cpp",
                   "plugins/offline-deepseek/prepare_main.cpp",
                   "plugins/offline-deepseek/store_main.cpp",
                   "plugins/offline-deepseek/identity_copy.cpp", "plugins/offline-deepseek/shard_layout.cpp",
                   "plugins/offline-deepseek/generation_layout.cpp",
                   "plugins/offline-deepseek/generation_metadata.cpp",
                   "plugins/offline-deepseek/prepare_generation.cpp",
                   "plugins/offline-deepseek/source_payload.cpp",
                   "plugins/offline-deepseek/source_artifact.cpp",
                   "plugins/offline-deepseek/source_semantics.cpp",
                   "plugins/offline-deepseek/source_inventory.cpp",
                   "plugins/offline-deepseek/disposition_records.cpp",
                   "plugins/offline-deepseek/generation_copy.cpp",
                   "plugins/offline-deepseek/generation_verify.cpp",
                   "plugins/offline-deepseek/generation_receipt.cpp",
                   "plugins/offline-deepseek/generation_store.cpp",
                   "plugins/offline-deepseek/generation_commit.cpp",
                   "plugins/offline-deepseek/generation_pointer.cpp",
                   "plugins/offline-deepseek/generation_activate.cpp",
                   "plugins/offline-deepseek/active_generation.cpp",
                   "plugins/model-deepseek-v41/config.cpp",
                   "plugins/model-deepseek-v41/attention_step.cpp",
                   "plugins/model-deepseek-v41/engram_hash.cpp",
                   "plugins/model-deepseek-v41/engram_weights.cpp",
                   "plugins/model-deepseek-v41/weight_inventory.cpp",
                    "plugins/model-deepseek-v41/weight_partition_copy.cpp",
                    "plugins/model-deepseek-v41/weight_dequantize.cpp",
                    "plugins/model-deepseek-v41/weight_expert_convert.cpp",
                    "plugins/model-deepseek-v41/token_map_builder.cpp",
                    "plugins/model-deepseek-v41/token_map_main.cpp",
                    "plugins/model-deepseek-v41/weight_scalar_convert.cpp",
                    "plugins/model-deepseek-v41/weight_source_wo_a.cpp",
                    "plugins/model-deepseek-v41/weight_source_file.cpp",
                    "plugins/model-deepseek-v41/weight_source_inventory.cpp",
                    "plugins/model-deepseek-v41/weight_source_checkpoint.cpp",
                    "plugins/model-deepseek-v41/weight_output_layout.cpp",
                    "plugins/model-deepseek-v41/weight_materialize.cpp",
                    "plugins/model-deepseek-v41/reshard_main.cpp",
                   "plugins/model-deepseek-v41/weight_catalog.cpp",
                   "plugins/model-deepseek-v41/weight_files.cpp",
                   "plugins/model-deepseek-v41/weight_manifest.cpp",
                   "plugins/model-deepseek-v41/uploaded_bindings.cpp",
                   "plugins/model-deepseek-v41/sequence_cache.cpp",
                   "plugins/model-deepseek-v41/boundary_plan.cpp",
                   "plugins/model-deepseek-v41/ffn_plan.cpp",
                   "plugins/model-deepseek-v41/attention_prepare_plan.cpp",
                   "plugins/model-deepseek-v41/attention_output_plan.cpp",
                   "plugins/model-deepseek-v41/engram_plan.cpp",
                   "plugins/model-deepseek-v41/compressor_plan.cpp",
                   "plugins/model-deepseek-v41/indexer_plan.cpp",
                   "plugins/model-deepseek-v41/engram_launch.cpp",
                   "plugins/model-deepseek-v41/engram_pipeline.cpp",
                   "plugins/model-deepseek-v41/mhc_launch.cpp",
                   "plugins/model-deepseek-v41/norm_launch.cpp",
                   "plugins/model-deepseek-v41/model_head.cpp",
                   "plugins/model-deepseek-v41/token_embedding.cpp",
                   "plugins/model-deepseek-v41/sampling.cpp",
                   "plugins/model-deepseek-v41/token_stop.cpp",
                   "plugins/model-deepseek-v41/token_ledger.cpp",
                   "plugins/model-deepseek-v41/rank_commit.cpp",
                   "plugins/model-deepseek-v41/rank_channel.cpp",
                   "plugins/model-deepseek-v41/rank_collector.cpp",
                   "plugins/model-deepseek-v41/inference_request.cpp",
                   "plugins/model-deepseek-v41/request_channel.cpp",
                   "plugins/model-deepseek-v41/lifecycle_channel.cpp",
                   "plugins/model-deepseek-v41/rank_socket.cpp",
                   "plugins/model-deepseek-v41/rank_channels.cpp",
                   "plugins/model-deepseek-v41/worker_spawn.cpp",
                   "plugins/model-deepseek-v41/worker_executable.cpp",
                   "plugins/model-deepseek-v41/worker_environment.cpp",
                   "plugins/model-deepseek-v41/nccl_bootstrap_channel.cpp",
                   "plugins/model-deepseek-v41/nccl_bootstrap_broker.cpp",
                   "plugins/model-deepseek-v41/worker_bootstrap.cpp",
                   "plugins/model-deepseek-v41/worker_group.cpp",
                   "plugins/model-deepseek-v41/worker_cgroup.cpp",
                   "plugins/model-deepseek-v41/worker_artifacts.cpp",
                   "plugins/model-deepseek-v41/generation_loop.cpp",
                   "plugins/model-deepseek-v41/rank_process_watch.cpp",
                   "plugins/model-deepseek-v41/generation_session.cpp",
                   "plugins/model-deepseek-v41/worker_handles.cpp",
                   "plugins/model-deepseek-v41/token_output.cpp",
                   "plugins/execution-default/output_burst_credit_pool.cpp",
                   "plugins/model-deepseek-v41/rope_launch.cpp",
                   "plugins/model-deepseek-v41/window_kv.cpp",
                   "plugins/model-deepseek-v41/sparse_attention.cpp",
                   "plugins/model-deepseek-v41/attention_output.cpp",
                   "plugins/model-deepseek-v41/attention_input.cpp",
                   "plugins/model-deepseek-v41/compressor_pool.cpp",
                   "plugins/model-deepseek-v41/compressed_kv.cpp",
                   "plugins/model-deepseek-v41/indexer_input.cpp",
                   "plugins/model-deepseek-v41/indexer_score.cpp",
                   "plugins/model-deepseek-v41/indexer_select.cpp",
                   "plugins/model-deepseek-v41/indexer_reduce.cpp",
                   "plugins/model-deepseek-v41/indexer_pipeline.cpp",
                   "plugins/model-deepseek-v41/indexer_chain.cpp",
                   "plugins/model-deepseek-v41/attention_assemble.cpp",
                   "plugins/model-deepseek-v41/attention_residual.cpp",
                   "plugins/model-deepseek-v41/attention_reduce.cpp",
                   "plugins/model-deepseek-v41/attention_pipeline.cpp",
                   "plugins/model-deepseek-v41/expert_fp8.cpp",
                   "plugins/model-deepseek-v41/linear_fp4.cpp",
                   "plugins/model-deepseek-v41/router.cpp",
                   "plugins/model-deepseek-v41/expert_dispatch.cpp",
                   "plugins/model-deepseek-v41/expert_chain.cpp",
                   "plugins/model-deepseek-v41/expert_workspace.cpp",
                   "plugins/model-deepseek-v41/expert_workspace_owner.cpp",
                   "plugins/model-deepseek-v41/attention_prepare.cpp",
                   "plugins/model-deepseek-v41/attention_sources.cpp",
                   "plugins/model-deepseek-v41/step_phases.cpp",
                   "plugins/model-deepseek-v41/compressed_prepare.cpp",
                   "plugins/model-deepseek-v41/expert_residual.cpp",
                   "plugins/model-deepseek-v41/ffn_route.cpp",
                   "plugins/model-deepseek-v41/linear_fp8.cpp",
                   "plugins/offline-deepseek/generation_promote.cpp",
                   "plugins/offline-deepseek/tensor_inventory.cpp",
                   "plugins/model-deepseek-v4-flash/deepseek_runtime_artifact_manifest.cpp",
                   "plugins/model-deepseek-v4-flash/deepseek_runtime_records_manifest.cpp",
                   "plugins/model-deepseek-v4-flash/deepseek_tensor_ownership_plan.cpp", "plugins/model-deepseek-v4-flash/deepseek_v4_config.cpp",
                   "src/core/allocator.cpp", "src/core/canonical_json.cpp", "src/core/buffer.cpp",
                   "src/core/float16.cpp", "src/core/memory_copier.cpp", "src/core/canonical_hash.cpp",
                   "src/core/tensor_view.cpp", "src/model/safetensors_header.cpp",
                   "src/model/safetensors_shard_index.cpp", "src/core/bounded_json.cpp",
                   "src/core/sha256.cpp", "src/core/status.cpp"]
    if args.deepseek_model_source:
        closure = sorted(path.relative_to(ROOT).as_posix() for path in
                         (ROOT / "plugins/model-deepseek-v4-flash").glob("*deepseek*.cpp"))
        expected_root = "8c47647f1a5f72ef1d35df8cbfe03733dd19b93a641ae778c44243b75a117a27"
        cmake = (ROOT / "plugins/model-deepseek-v4-flash/host.cmake").read_text(encoding="utf-8")
        if (len(closure) != 161 or
                hashlib.sha256("\n".join(closure).encode()).hexdigest() != expected_root or
                expected_root not in cmake):
            parser.error("PP1 model source closure differs from pinned native CMake selection")
        sources.extend(path for path in closure if path not in sources)
    if args.deepseek_model_build:
        commands = json.loads((args.deepseek_model_build / "compile_commands.json").read_text(encoding="utf-8"))
        closure = []
        for entry in commands:
            if "pih_model_deepseek_v4_flash_impl.dir" not in entry.get("output", entry.get("command", "")):
                continue
            command = entry.get("command", " ".join(entry.get("arguments", [])))
            if "PIH_DEEPSEEK_DSPARK_DISABLED=1" not in command or "PIH_DEEPSEEK_NCCL_DISABLED" in command:
                parser.error("CMake model target does not select the native PP1 compile definition")
            closure.append(Path(entry["file"]).resolve(strict=True).relative_to(ROOT).as_posix())
        closure.sort()
        if len(closure) != 161 or hashlib.sha256("\n".join(closure).encode()).hexdigest() != "8c47647f1a5f72ef1d35df8cbfe03733dd19b93a641ae778c44243b75a117a27":
            parser.error("CMake model source closure does not match native PP1")
        sources.extend(path for path in closure if path not in sources)
    if args.deepseek_backend:
        cmake = (ROOT / "plugins/model-deepseek-v4-flash/cuda.cmake").read_text(encoding="utf-8")
        backend = sorted(path.relative_to(ROOT).as_posix()
                         for path in (ROOT / "plugins/model-deepseek-v4-flash/cuda").glob("*.cpp"))
        expected = "1c712c76584b56ca89f9b6c31d76ed26a05b49bf556d17e814c700784959b1c8"
        if (len(backend) != 24
                or hashlib.sha256("\n".join(backend).encode()).hexdigest() != expected
                or expected not in cmake
                or "PIH_DEEPSEEK_PP1_CUDA_SOURCE_COUNT EQUAL 24" not in cmake):
            parser.error("native backend source set differs from pinned CMake closure")
        sources.extend(path for path in backend if path not in sources)
    if args.plugin_support:
        for owner in ("execution-default", "storage-verified-artifact", "memory-host-spill", "platform-linux"):
            directory = ROOT / "plugins" / owner
            implementation = sorted(directory.glob("*.cpp"))
            if not implementation:
                parser.error(f"plugin support owner has no source files: {owner}")
            for path in implementation:
                name = path.relative_to(ROOT).as_posix()
                if path.is_symlink() or not path.is_file():
                    parser.error(f"plugin source must be a regular non-symlink file: {name}")
                if name not in sources:
                    sources.append(name)
    sysroot = args.sysroot.resolve(strict=True)
    lock_bytes = Path(__file__).with_name("linux-syntax-sysroot.lock.json").read_bytes()
    receipt = json.loads((sysroot / "receipt.json").read_text(encoding="utf-8"))
    if receipt.get("lock_sha256") != hashlib.sha256(lock_bytes).hexdigest():
        parser.error("header preparation receipt does not match the pinned package lock")
    flags = ["--target=x86_64-linux-gnu", f"--sysroot={sysroot}", "-nostdinc++"]
    for path in ("usr/include/c++/13", "usr/include/x86_64-linux-gnu/c++/13", "usr/include/x86_64-linux-gnu"):
        flags += ["-isystem", str(sysroot / path)]
    flags += ["-std=c++20", "-fsyntax-only", "-I", str(ROOT / "include"), "-I", str(ROOT / "src"),
              "-DPIH_DEEPSEEK_DSPARK_DISABLED=1",
              # Host syntax selection only; no SM103 kernel generation or device qualification.
              "-DPIH_V41_COMPILED_SM=103"]
    if args.v41_model:
        flags += ["-I", str(ROOT / "plugins/model-deepseek-v41"),
                  "-isystem", str(args.nccl_headers.resolve(strict=True) / "include")]
        directory = args.nccl_headers.resolve(strict=True)
        lock = hashlib.sha256(Path(__file__).with_name("nccl-syntax-headers.lock.json").read_bytes()).hexdigest()
        prepared = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        if prepared.get("lock_sha256") != lock or not isinstance(prepared.get("members"), dict):
            parser.error("NCCL syntax header receipt differs from pinned lock")
        for relative, expected in prepared["members"].items():
            member_path = PurePosixPath(relative)
            if (member_path.is_absolute() or ".." in member_path.parts or
                    "\\" in relative or ":" in relative):
                parser.error("unsafe NCCL header receipt path")
            member = (directory / member_path).resolve(strict=True)
            if directory not in member.parents or hashlib.sha256(member.read_bytes()).hexdigest() != expected:
                parser.error(f"NCCL syntax header content differs: {relative}")
    if args.qwen_model or args.qwen_qualification or args.v41_model:
        if args.qwen_model:
            if args.generated_dir is None or not (args.generated_dir / "pih/version.h").is_file():
                parser.error("--qwen-model requires --generated-dir containing pih/version.h")
            project = re.search(r'project\(PIH VERSION ([0-9]+\.[0-9]+\.[0-9]+) LANGUAGES CXX\)',
                                (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
            generated_version = (args.generated_dir / "pih/version.h").read_text(encoding="utf-8")
            if project is None or f'#define PIH_VERSION_STRING "{project.group(1)}"' not in generated_version:
                parser.error("--generated-dir has a stale or unrelated PIH version header")
            flags += ["-I", str(args.generated_dir.resolve(strict=True))]
        for name, directory, lock_name in (
                ("CUDA", args.cuda_headers.resolve(strict=True), "cuda-syntax-headers.lock.json"),
                ("cuBLAS", args.cublas_headers.resolve(strict=True), "cublas-syntax-headers.lock.json")):
            lock = hashlib.sha256(Path(__file__).with_name(lock_name).read_bytes()).hexdigest()
            prepared = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
            members = prepared.get("members" if name == "CUDA" else "files")
            if prepared.get("lock_sha256") != lock or not isinstance(members, dict) or not members:
                parser.error(f"{name} syntax header receipt differs from pinned lock")
            for relative, expected in members.items():
                member_path = PurePosixPath(relative)
                if (member_path.is_absolute() or ".." in member_path.parts or
                        "\\" in relative or ":" in relative):
                    parser.error(f"unsafe {name} header receipt path")
                member = (directory / member_path).resolve(strict=True)
                if directory not in member.parents or hashlib.sha256(member.read_bytes()).hexdigest() != expected:
                    parser.error(f"{name} syntax header content differs: {relative}")
            flags += ["-isystem", str(directory / "include")]
    version = subprocess.check_output([args.compiler, "--version"], text=True)
    print(version, end="", flush=True)
    exit_code = 0
    failed_batches = []
    diagnostics = []
    for start in range(0, len(sources), 16):
        batch = sources[start:start + 16]
        print(f"syntax checking {start + 1}..{start + len(batch)} / {len(sources)}", flush=True)
        result = subprocess.run([args.compiler, *flags, *batch], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        diagnostics.append(result.stdout + result.stderr)
        if result.returncode:
            exit_code = result.returncode
            failed_batches.append(batch)
            print("\n".join(result.stderr.splitlines()[:24]), flush=True)
    variants = []
    if args.artifact_tools:
        variant_source = "plugins/model-deepseek-v41/reshard_main.cpp"
        variant_flags = ["-DPIH_V41_CONVERT_COMMAND=1"]
        print("syntax checking V4.1 conversion command compile definition", flush=True)
        result = subprocess.run([args.compiler, *flags, *variant_flags, variant_source], cwd=ROOT,
                                capture_output=True, text=True, encoding="utf-8", errors="replace")
        diagnostics.append(result.stdout + result.stderr)
        variants.append({"source": variant_source, "extra_flags": variant_flags,
                         "exit_code": result.returncode})
        if result.returncode:
            exit_code = result.returncode
            failed_batches.append([variant_source, *variant_flags])
            print("\n".join(result.stderr.splitlines()[:24]), flush=True)
    report = {
        "schema": "pih.native_syntax_observation.v1", "compiler": version,
        "flags": flags, "header_lock_sha256": receipt["lock_sha256"],
        "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
        "exit_code": exit_code, "failed_batches": failed_batches,
        "compile_variants": variants,
        "scope": f"{len(sources)} explicit translation units; syntax only, not linking, execution, model tests or a whole-source/header closure receipt",
    }
    output = ROOT / ("out/native-syntax-closure-report.json" if
                     (args.deepseek_model_build or args.deepseek_model_source) else
                     "out/native-syntax-report.json")
    if args.artifact_tools:
        output = ROOT / "out/native-artifact-syntax-report.json"
    elif args.qwen_model:
        output = ROOT / "out/qwen-model-syntax-report.json"
    elif args.qwen_qualification:
        output = ROOT / "out/qwen-qualification-syntax-report.json"
    elif args.v41_model:
        output = ROOT / "out/v41-model-syntax-report.json"
    elif args.deepseek_backend and not args.deepseek_model_build:
        output = ROOT / "out/native-backend-syntax-report.json"
    if args.plugin_support:
        output = output.with_name(output.stem + "-plugin-support.json")
    output.parent.mkdir(exist_ok=True)
    log = output.with_suffix(".log")
    log.write_text("\n".join(diagnostics), encoding="utf-8")
    report["diagnostics"] = str(log)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"syntax observation: {output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
