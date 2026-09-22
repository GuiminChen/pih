#!/usr/bin/env bash
# Development orchestration only. No Python inference, automatic downloads,
# automatic conversion, cgroup delegation, fallback runtime or qualification.
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_directory="$(cd -- "$script_directory/.." && pwd)"
usage() {
    cat >&2 <<'USAGE'
Native V4.1 development entry (Linux only; not a qualified release profile):
  bash deploy/run-v41-native.sh build-service /absolute/build-dir /absolute/new-bundle [jobs]
  bash deploy/run-v41-native.sh seal-rank /absolute/bundle
  bash deploy/run-v41-native.sh seal-service /absolute/bundle /absolute/config.json TRUSTED_SHA256
  bash deploy/run-v41-native.sh serve /absolute/bundle [port]
  bash deploy/run-v41-native.sh probe /absolute/new-observation.json [port] [max_tokens]
  bash deploy/run-v41-native.sh build /absolute/build-dir 103 [jobs]
  bash deploy/run-v41-native.sh build-weights /absolute/build-dir [jobs]
  bash deploy/run-v41-native.sh token-map /absolute/build-dir TOKENIZER_JSON TRUSTED_MAP_SHA256 OUTPUT_DIRECTORY
  bash deploy/run-v41-native.sh convert /absolute/build-dir SOURCE EXPECTATIONS_JSON EXPECTATIONS_SHA256 EXCLUSIONS_JSON EXCLUSIONS_SHA256 DEVICE_BUDGET_BYTES OUTPUT
  bash deploy/run-v41-native.sh reshard /absolute/build-dir CONFIG CONFIG_SHA256 SOURCE SOURCE_MANIFEST_SHA256 WORLD RANK DEVICE_BUDGET_BYTES OUTPUT
  bash deploy/run-v41-native.sh check /absolute/build-dir /absolute/config.json TRUSTED_SHA256
  bash deploy/run-v41-native.sh check-request /absolute/build-dir /absolute/config.json TRUSTED_SHA256
  bash deploy/run-v41-native.sh run   /absolute/build-dir /absolute/config.json TRUSTED_SHA256
build compiles native commands/backend only; it does not run tests or models.
check authenticates configuration and static request budgets; it is not artifact/hardware validation.
check-request also authenticates tokenizer and admits prompt/context/output buffers; no workers, cgroups or GPUs are started.
run starts the configured single-request process group and emits raw token bytes.
build-weights builds the CPU-only conversion/reshard/token-map commands; it does not execute them.
convert explicitly converts admitted HF text-backbone types to canonical TP1; unsupported types fail.
reshard processes already-canonical TP1 weights, not raw HF checkpoints; all paths absolute.
convert/reshard accept optional suffix: --runtime-map MAP_FILE TRUSTED_MAP_SHA256
USAGE
}
die() { printf '%s\n' "$*" >&2; exit 2; }
[[ $# -ge 1 ]] || { usage; exit 2; }
[[ "$(uname -s)" == Linux ]] || die 'Native V4.1 orchestration requires Linux.'
action="$1"
shift
case "$action" in
    build-service)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
        build_directory="$1"; bundle="$2"; jobs="${3:-2}"
        [[ "$build_directory" == /* && "$bundle" == /* ]] || die 'Build and bundle paths must be absolute.'
        build_directory="$(realpath -m -- "$build_directory")"
        bundle="$(realpath -m -- "$bundle")"
        [[ "$build_directory" != / && "$build_directory" != "$repository_directory" ]] || die 'Use a dedicated build directory.'
        [[ "$build_directory" != "$bundle" && "$build_directory" != "$bundle"/* && "$bundle" != "$build_directory"/* ]] || die 'Build and bundle directories must be disjoint.'
        [[ "$bundle" != / && "$bundle" != "$repository_directory" && ! -e "$bundle" && ! -L "$bundle" ]] || die 'Bundle destination must be new.'
        [[ "$jobs" =~ ^[1-9][0-9]{0,2}$ ]] || die 'Jobs must be 1..999.'
        cmake -S "$repository_directory" -B "$build_directory" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
            -DCMAKE_CUDA_ARCHITECTURES=103-real -DPIH_DEPLOYMENT_PROFILE=custom \
            '-DPIH_ENABLED_PLUGINS=pih.platform.linux;pih.backend.nvidia-cuda;pih.transport.nccl;pih.kernels.deepseek-v41.sm103;pih.model.deepseek-v41;pih.surface.text-http' \
            -DPIH_ENABLE_CUDA=ON -DPIH_BUILD_NATIVE_ENGRAM_NCCL=ON \
            -DPIH_ENABLE_NCCL=OFF -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF \
            -DPIH_BUILD_TESTS=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=OFF -DBUILD_TESTING=OFF \
            -DPIH_BUILD_PLUGINS=ON -DPIH_BUILD_WORKER=ON -DPIH_BUILD_TOKENIZER_TOOLS=ON \
            -DPIH_BUILD_ARTIFACT_TOOLS=OFF -DPIH_BUILD_QWEN_ARTIFACT_TOOLS=OFF
        cmake --build "$build_directory" --parallel "$jobs" --target pih-release
        cmake --install "$build_directory" --prefix "$bundle" --component pih-release
        printf '%s\n' 'Built SM103 native service; prepare rank lock and authenticated supervisor configuration before sealing service.' >&2
        ;;
    seal-rank)
        [[ $# == 1 ]] || { usage; exit 2; }
        cmake "-DBUNDLE=$1" -DMODE=rank -P "$repository_directory/cmake/PIHSealV41.cmake"
        ;;
    seal-service)
        [[ $# == 3 ]] || { usage; exit 2; }
        cmake "-DBUNDLE=$1" -DMODE=service "-DCONFIG=$2" "-DCONFIG_SHA256=$3" \
            -P "$repository_directory/cmake/PIHSealV41.cmake"
        ;;
    serve)
        [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
        bundle="$1"; port="${2:-8000}"
        [[ "$bundle" == /* && -f "$bundle/v41-service.lock" ]] || die 'Sealed service bundle is required.'
        [[ "$port" =~ ^[1-9][0-9]{0,4}$ && "$port" -le 65535 ]] || die 'Port must be 1..65535.'
        exec "$bundle/bin/pih-worker" --lock "$bundle/v41-service.lock" --serve "$port"
        ;;
    probe)
        [[ $# -ge 1 && $# -le 3 ]] || { usage; exit 2; }
        observation="$1"; port="${2:-8000}"; max_tokens="${3:-32}"
        [[ "$observation" == /* && ! -e "$observation" && ! -L "$observation" ]] || die 'Observation path must be absolute and new.'
        [[ "$port" =~ ^[1-9][0-9]{0,4}$ && "$port" -le 65535 ]] || die 'Port must be 1..65535.'
        [[ "$max_tokens" =~ ^[1-9][0-9]{0,3}$ && "$max_tokens" -le 4096 ]] || die 'Max tokens must be 1..4096.'
        exec python3 "$repository_directory/deploy/native.py" probe \
            --model 'deepseek-ai/DeepSeek-V4.1-Flash' \
            --url "http://127.0.0.1:$port" --max-tokens "$max_tokens" \
            --output "$observation"
        ;;
    build-weights)
        [[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
        build_directory="$1"
        jobs="${2:-2}"
        [[ "$build_directory" == /* ]] || die 'Build directory must be absolute.'
        build_directory="$(realpath -m -- "$build_directory")"
        [[ "$build_directory" != / && "$build_directory" != "$repository_directory" ]] || die 'Use a dedicated build directory.'
        [[ "$jobs" =~ ^[1-9][0-9]{0,2}$ ]] || die 'Jobs must be an integer from 1 to 999.'
        cmake -S "$repository_directory" -B "$build_directory" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPIH_DEPLOYMENT_PROFILE=custom \
            -DPIH_ENABLE_CUDA=OFF -DPIH_ENABLE_NCCL=OFF -DPIH_BUILD_NATIVE_ENGRAM_NCCL=OFF \
            -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF -DPIH_BUILD_TESTS=OFF \
            -DPIH_BUILD_NATIVE_CONTRACT_TESTS=OFF -DBUILD_TESTING=OFF \
            -DPIH_BUILD_PLUGINS=OFF -DPIH_BUILD_WORKER=OFF -DPIH_BUILD_TOKENIZER_TOOLS=ON \
            -DPIH_BUILD_ARTIFACT_TOOLS=OFF -DPIH_BUILD_QWEN_ARTIFACT_TOOLS=OFF
        cmake --build "$build_directory" --parallel "$jobs" --target pih-v41-reshard pih-v41-convert pih-v41-token-map
        ;;
    reshard|convert|token-map)
        if [[ "$action" == token-map ]]; then
            [[ $# == 4 ]] || { usage; exit 2; }
        elif [[ "$action" == convert ]]; then
            [[ $# == 8 || $# == 11 ]] || { usage; exit 2; }
        else
            [[ $# == 9 || $# == 12 ]] || { usage; exit 2; }
        fi
        build_directory="$1"
        shift
        [[ "$build_directory" == /* ]] || die 'Build directory must be absolute.'
        command_path="$build_directory/pih-v41-$action"
        [[ -f "$command_path" && -x "$command_path" ]] || die 'Native offline command is absent; run build-weights first.'
        exec "$command_path" "$@"
        ;;
    build)
        [[ $# -ge 2 && $# -le 3 ]] || { usage; exit 2; }
        build_directory="$1"
        architecture="$2"
        jobs="${3:-2}"
        [[ "$build_directory" == /* ]] || die 'Build directory must be absolute.'
        build_directory="$(realpath -m -- "$build_directory")"
        [[ "$build_directory" != / && "$build_directory" != "$repository_directory" ]] || die 'Use a dedicated build directory.'
        [[ "$architecture" == 103 ]] || die 'V4.1 rank worker requires B300/SM103.'
        [[ "$jobs" =~ ^[1-9][0-9]{0,2}$ ]] || die 'Jobs must be an integer from 1 to 999.'
        cmake -S "$repository_directory" -B "$build_directory" -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            "-DCMAKE_CUDA_ARCHITECTURES=${architecture}-real" \
            -DPIH_DEPLOYMENT_PROFILE=custom \
            '-DPIH_ENABLED_PLUGINS=pih.platform.linux;pih.backend.nvidia-cuda;pih.transport.nccl;pih.kernels.deepseek-v41.sm103' \
            -DPIH_ENABLE_CUDA=ON -DPIH_BUILD_NATIVE_ENGRAM_NCCL=ON \
            -DPIH_ENABLE_NCCL=OFF -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF \
            -DPIH_BUILD_TESTS=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=OFF -DBUILD_TESTING=OFF \
            -DPIH_BUILD_PLUGINS=ON -DPIH_BUILD_WORKER=ON \
            -DPIH_BUILD_TOKENIZER_TOOLS=ON -DPIH_BUILD_ARTIFACT_TOOLS=OFF -DPIH_BUILD_QWEN_ARTIFACT_TOOLS=OFF
        cmake --build "$build_directory" --parallel "$jobs" --target \
            pih-v41-supervisor pih-v41-rank-worker pih-v41-nccl-bootstrap pih_kernels_deepseek_v41_sm103 pih_plugin_backend_nvidia_cuda pih_plugin_platform_linux pih_plugin_transport_nccl
        printf '%s\n' 'Development binaries built; no model execution or hardware qualification performed.' >&2
        ;;
    check|check-request|run)
        [[ $# == 3 ]] || { usage; exit 2; }
        build_directory="$1"
        configuration="$2"
        trusted_digest="$3"
        [[ "$build_directory" == /* && "$configuration" == /* ]] || die 'Build and configuration paths must be absolute.'
        [[ "$trusted_digest" =~ ^[0-9a-f]{64}$ && "$trusted_digest" != 0000000000000000000000000000000000000000000000000000000000000000 ]] || die 'Provide the nonzero lowercase digest from your trusted deployment record.'
        supervisor="$build_directory/pih-v41-supervisor"
        [[ -f "$supervisor" && -x "$supervisor" ]] || die 'Native supervisor is absent; build the development targets first.'
        if [[ "$action" == check ]]; then
            exec "$supervisor" --check-config "$configuration" --sha256 "$trusted_digest"
        fi
        if [[ "$action" == check-request ]]; then
            exec "$supervisor" --check-request "$configuration" --sha256 "$trusted_digest"
        fi
        exec "$supervisor" "$configuration" --sha256 "$trusted_digest"
        ;;
    *) usage; exit 2 ;;
esac
