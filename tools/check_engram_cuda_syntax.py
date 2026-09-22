"""Clang compatibility syntax checks with official CUDA headers; not NVCC qualification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="clang++")
    parser.add_argument("--sysroot", type=Path, default=ROOT / "out/linux-syntax-sysroot")
    parser.add_argument("--cuda-headers", type=Path, default=ROOT / "out/cuda-syntax-headers-13.2.86")
    parser.add_argument("--nccl-headers", type=Path, help="also check the NCCL adapter and TP scheduler")
    args = parser.parse_args()
    sysroot, cuda = args.sysroot.resolve(), args.cuda_headers.resolve()
    locks = {}
    header_trees = [
        ("linux", sysroot, "linux-syntax-sysroot.lock.json"),
        ("cuda", cuda, "cuda-syntax-headers.lock.json"),
    ]
    if args.nccl_headers:
        header_trees.append(("nccl", args.nccl_headers.resolve(), "nccl-syntax-headers.lock.json"))
    for name, directory, lock_name in header_trees:
        digest = hashlib.sha256(Path(__file__).with_name(lock_name).read_bytes()).hexdigest()
        receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
        if receipt.get("lock_sha256") != digest:
            parser.error(name + " header receipt differs from pinned lock")
        locks[name] = digest
        if name != "linux":
            for relative, expected in receipt["members"].items():
                path = PurePosixPath(relative)
                if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
                    parser.error("unsafe NVIDIA receipt path")
                member = (directory / path).resolve()
                if directory not in member.parents or hashlib.sha256(member.read_bytes()).hexdigest() != expected:
                    parser.error("NVIDIA header content differs: " + relative)
    flags = ["--target=x86_64-linux-gnu", f"--sysroot={sysroot}", "-nostdinc++"]
    for path in ("usr/include/c++/13", "usr/include/x86_64-linux-gnu/c++/13", "usr/include/x86_64-linux-gnu"):
        flags += ["-isystem", str(sysroot / path)]
    flags += ["-I", str(ROOT / "include"), "-std=c++20", "-fsyntax-only", "-ffp-contract=off"]
    source = "plugins/model-deepseek-v41/engram_cuda.cu"
    cuda_flags = ["-nocudalib", "--cuda-gpu-arch=sm_90", f"--cuda-path={cuda}"]
    # LLVM 22 maps newer SDKs to its latest partially supported 12.9 feature
    # level on the host. Make the same compatibility selection explicit for
    # device-only parsing. Do NOT suppress the unknown-version warning.
    checks = [
        ("completion-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/engram_completion.cpp"),
        ("expert-counts-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/expert_counts.cpp"),
        ("token-input-upload-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/token_input_upload.cpp"),
        ("weight-upload-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/weight_upload.cpp"),
        ("weight-memory-owner-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/weight_memory_owner.cpp"),
        ("expert-batch-host", ["-isystem", str(cuda / "include")], "plugins/model-deepseek-v41/expert_batch.cpp"),
        ("cuda-host", [*cuda_flags, "--cuda-host-only"], source),
        ("cuda-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"], source),
    ]
    checks += [
        ("mhc-launch-host", [], "plugins/model-deepseek-v41/mhc_launch.cpp"),
        ("mhc-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/mhc_cuda.cu"),
        ("mhc-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/mhc_cuda.cu"),
    ]
    if args.nccl_headers:
        nccl_flags = ["-isystem", str(cuda / "include"),
                      "-isystem", str(args.nccl_headers.resolve() / "include"),
                      "-I", str(ROOT / "plugins/model-deepseek-v41")]
        checks += [
            ("nccl-reduction-host", nccl_flags, "plugins/model-deepseek-v41/engram_reduce.cpp"),
            ("nccl-pipeline-host", nccl_flags, "plugins/model-deepseek-v41/engram_pipeline.cpp"),
            ("nccl-indexer-host", nccl_flags, "plugins/model-deepseek-v41/indexer_reduce.cpp"),
            ("nccl-indexer-pipeline-host", nccl_flags, "plugins/model-deepseek-v41/indexer_pipeline.cpp"),
            ("nccl-attention-host", nccl_flags, "plugins/model-deepseek-v41/attention_reduce.cpp"),
            ("nccl-attention-pipeline-host", nccl_flags, "plugins/model-deepseek-v41/attention_pipeline.cpp"),
            ("nccl-expert-pipeline-host", nccl_flags, "plugins/model-deepseek-v41/expert_pipeline.cpp"),
            ("nccl-ffn-continuation-host", nccl_flags, "plugins/model-deepseek-v41/ffn_continuation.cpp"),
            ("nccl-ffn-operation-host", nccl_flags, "plugins/model-deepseek-v41/ffn_operation.cpp"),
            ("nccl-prepared-block-host", nccl_flags, "plugins/model-deepseek-v41/prepared_block.cpp"),
            ("nccl-indexed-sources-host", nccl_flags, "plugins/model-deepseek-v41/indexed_sources.cpp"),
            ("nccl-block-bindings-host", nccl_flags, "plugins/model-deepseek-v41/block_bindings.cpp"),
            ("nccl-block-liveness-host", nccl_flags, "plugins/model-deepseek-v41/block_liveness.cpp"),
            ("nccl-block-operation-host", nccl_flags, "plugins/model-deepseek-v41/block_operation.cpp"),
            ("nccl-backbone-operation-host", nccl_flags, "plugins/model-deepseek-v41/backbone_operation.cpp"),
            ("nccl-inference-operation-host", nccl_flags, "plugins/model-deepseek-v41/inference_operation.cpp"),
            ("nccl-inference-memory-host", nccl_flags, "plugins/model-deepseek-v41/inference_memory.cpp"),
            ("nccl-inference-memory-owner-host", nccl_flags, "plugins/model-deepseek-v41/inference_memory_owner.cpp"),
            ("nccl-rank-worker-loop-host", nccl_flags, "plugins/model-deepseek-v41/rank_worker_loop.cpp"),
            ("nccl-worker-communicator-host", nccl_flags, "plugins/model-deepseek-v41/worker_communicator.cpp"),
            ("nccl-bootstrap-helper-host", nccl_flags, "plugins/transport-nccl/nccl_bootstrap_main.cpp"),
            ("nccl-transport-provider-host", nccl_flags, "plugins/transport-nccl/entrypoint.cpp"),
            ("nccl-worker-memory-host", nccl_flags, "plugins/model-deepseek-v41/worker_memory.cpp"),
            ("nccl-worker-runtime-host", nccl_flags, "plugins/model-deepseek-v41/worker_runtime.cpp"),
            ("nccl-worker-main-host", nccl_flags + ["-I", str(ROOT / "src"),
                "-DPIH_V41_COMPILED_SM=103"],
                "plugins/model-deepseek-v41/worker_main.cpp"),
            ("nccl-block-sequence-host", nccl_flags, "plugins/model-deepseek-v41/block_sequence.cpp"),
            ("nccl-head-operation-host", nccl_flags, "plugins/model-deepseek-v41/head_operation.cpp"),
            ("nccl-embedding-operation-host", nccl_flags, "plugins/model-deepseek-v41/embedding_operation.cpp"),
            ("sampling-operation-host", nccl_flags, "plugins/model-deepseek-v41/sampling_operation.cpp"),
        ]
    checks += [
        ("model-head-host", [], "plugins/model-deepseek-v41/model_head.cpp"),
        ("token-embedding-host", [], "plugins/model-deepseek-v41/token_embedding.cpp"),
        ("token-embedding-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/token_embedding.cu"),
        ("token-embedding-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/token_embedding.cu"),
        ("sampling-host", [], "plugins/model-deepseek-v41/sampling.cpp"),
        ("token-stop-host", [], "plugins/model-deepseek-v41/token_stop.cpp"),
        ("token-ledger-host", [], "plugins/model-deepseek-v41/token_ledger.cpp"),
        ("token-output-host", [], "plugins/model-deepseek-v41/token_output.cpp"),
        ("sampling-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/sampling.cu"),
        ("sampling-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/sampling.cu"),
        ("model-head-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/model_head.cu"),
        ("model-head-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/model_head.cu"),
        ("norm-launch-host", [], "plugins/model-deepseek-v41/norm_launch.cpp"),
        ("norm-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/norm_cuda.cu"),
        ("norm-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/norm_cuda.cu"),
    ]
    version = subprocess.check_output([args.compiler, "--version"], text=True)
    checks += [
        ("rope-launch-host", [], "plugins/model-deepseek-v41/rope_launch.cpp"),
        ("rope-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/rope_cuda.cu"),
        ("rope-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/rope_cuda.cu"),
    ]
    observations = []
    checks += [
        ("window-launch-host", [], "plugins/model-deepseek-v41/window_kv.cpp"),
        ("window-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/window_kv.cu"),
        ("window-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/window_kv.cu"),
    ]
    checks += [
        ("attention-launch-host", [], "plugins/model-deepseek-v41/sparse_attention.cpp"),
        ("attention-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/sparse_attention.cu"),
        ("attention-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/sparse_attention.cu"),
    ]
    checks += [
        ("attention-output-host", [], "plugins/model-deepseek-v41/attention_output.cpp"),
        ("attention-output-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/attention_output.cu"),
        ("attention-output-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/attention_output.cu"),
    ]
    checks += [
        ("fp8-linear-host", [], "plugins/model-deepseek-v41/linear_fp8.cpp"),
        ("fp8-linear-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/linear_fp8.cu"),
        ("fp8-linear-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/linear_fp8.cu"),
    ]
    checks.append(("attention-input-host", [], "plugins/model-deepseek-v41/attention_input.cpp"))
    checks += [
        ("compressor-pool-host", [], "plugins/model-deepseek-v41/compressor_pool.cpp"),
        ("compressor-pool-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/compressor_pool.cu"),
        ("compressor-pool-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/compressor_pool.cu"),
    ]
    checks += [
        ("compressed-kv-host", [], "plugins/model-deepseek-v41/compressed_kv.cpp"),
        ("compressed-kv-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/compressed_kv.cu"),
        ("compressed-kv-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/compressed_kv.cu"),
    ]
    checks += [
        ("indexer-input-host", [], "plugins/model-deepseek-v41/indexer_input.cpp"),
        ("indexer-input-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/indexer_input.cu"),
        ("indexer-input-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/indexer_input.cu"),
    ]
    checks += [
        ("indexer-score-host", [], "plugins/model-deepseek-v41/indexer_score.cpp"),
        ("indexer-score-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/indexer_score.cu"),
        ("indexer-score-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/indexer_score.cu"),
    ]
    checks += [
        ("indexer-select-host", [], "plugins/model-deepseek-v41/indexer_select.cpp"),
        ("indexer-select-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/indexer_select.cu"),
        ("indexer-select-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/indexer_select.cu"),
    ]
    checks.append(("indexer-chain-host", [], "plugins/model-deepseek-v41/indexer_chain.cpp"))
    checks += [
        ("attention-assemble-host", [], "plugins/model-deepseek-v41/attention_assemble.cpp"),
        ("attention-assemble-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/attention_assemble.cu"),
        ("attention-assemble-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/attention_assemble.cu"),
    ]
    checks.append(("attention-residual-host", [], "plugins/model-deepseek-v41/attention_residual.cpp"))
    checks += [
        ("shared-expert-host", [], "plugins/model-deepseek-v41/expert_fp8.cpp"),
        ("shared-expert-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/expert_fp8.cu"),
        ("shared-expert-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/expert_fp8.cu"),
    ]
    checks.append(("fp4-linear-host", [], "plugins/model-deepseek-v41/linear_fp4.cpp"))
    checks += [
        ("router-host", [], "plugins/model-deepseek-v41/router.cpp"),
        ("router-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/router.cu"),
        ("router-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/router.cu"),
    ]
    checks += [
        ("expert-dispatch-host", [], "plugins/model-deepseek-v41/expert_dispatch.cpp"),
        ("expert-chain-host", [], "plugins/model-deepseek-v41/expert_chain.cpp"),
        ("expert-workspace-host", [], "plugins/model-deepseek-v41/expert_workspace.cpp"),
        ("expert-workspace-owner-host", [], "plugins/model-deepseek-v41/expert_workspace_owner.cpp"),
        ("attention-prepare-host", [], "plugins/model-deepseek-v41/attention_prepare.cpp"),
        ("attention-sources-host", [], "plugins/model-deepseek-v41/attention_sources.cpp"),
        ("step-phases-host", [], "plugins/model-deepseek-v41/step_phases.cpp"),
        ("compressed-prepare-host", [], "plugins/model-deepseek-v41/compressed_prepare.cpp"),
        ("expert-residual-host", [], "plugins/model-deepseek-v41/expert_residual.cpp"),
        ("ffn-route-host", [], "plugins/model-deepseek-v41/ffn_route.cpp"),
        ("expert-dispatch-cuda-host", [*cuda_flags, "--cuda-host-only"], "plugins/model-deepseek-v41/expert_dispatch.cu"),
        ("expert-dispatch-sm90-device-compatibility", [*cuda_flags, "--cuda-device-only", "-Xclang", "-target-sdk-version=12.9"],
         "plugins/model-deepseek-v41/expert_dispatch.cu"),
    ]
    for name, extra, unit in checks:
        print("checking " + name, flush=True)
        command = [args.compiler, *flags, *extra, unit]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(result.stdout + result.stderr, end="", flush=True)
        observations.append({"name": name, "command": command, "source": unit,
                             "source_sha256": hashlib.sha256((ROOT / unit).read_bytes()).hexdigest(),
                             "exit_code": result.returncode, "diagnostics": result.stdout + result.stderr})
    output = ROOT / "out/engram-cuda-syntax-report.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({
        "schema": "pih.engram_cuda_compatibility_syntax_observation.v1", "compiler": version,
        "header_lock_sha256": locks, "cuda_header_version": "13.2.86",
        "clang_device_sdk_feature_level": "12.9", "architecture": "sm_90",
        "nccl_header_version": "2.31.2" if args.nccl_headers else None,
        "compatibility_only": True, "nvcc_qualified": False, "linked": False,
        "hardware_execution": False, "model_tests": False, "checks": observations,
        "local_headers": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "plugins/model-deepseek-v41").glob("*.h"))},
    }, indent=2) + "\n", encoding="utf-8")
    print("compatibility syntax observation: " + str(output))
    return 1 if any(item["exit_code"] for item in observations) else 0


if __name__ == "__main__":
    raise SystemExit(main())
