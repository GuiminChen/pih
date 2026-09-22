#include "pih/backend/cuda/nvidia_deepseek_dspark_prefill_stage_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

namespace pih {

Result<NvidiaDeepSeekDsparkPrefillStageOperations>
NvidiaDeepSeekDsparkPrefillStageOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(
        result, "cuCtxGetCurrent DeepSeek DSpark prefill");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark prefill requires a current CUDA context");
  }
  return NvidiaDeepSeekDsparkPrefillStageOperations(
      reinterpret_cast<std::uint64_t>(context));
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(
        result, "cuCtxGetCurrent DeepSeek DSpark prefill");
  }
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark prefill CUDA context identity changed");
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::validate_host_error(
    std::uint32_t* host_error) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host_error == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill host error is null");
  }
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(&attributes, host_error);
  if (result != cudaSuccess) {
    return cuda_status(
        result, "cudaPointerGetAttributes DeepSeek DSpark prefill error");
  }
  return attributes.type == cudaMemoryTypeHost
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark prefill error is not pinned host memory");
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill error clear is invalid");
  }
  return cuda_status(
      cudaMemsetAsync(reinterpret_cast<void*>(device), 0,
                      sizeof(std::uint32_t),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek DSpark prefill error");
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::quant(
    DeepSeekFp8ActivationQuantLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_fp8_activation_quant(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::gemm(
    DeepSeekFp8GemmLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_fp8_gemm(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::rms(
    DeepSeekRmsNormLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_rms_norm(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::rotary(
    DeepSeekRotaryLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_rotary(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::kv_fp8_simulate(
    DeepSeekKvFp8SimulateLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_kv_fp8_simulate(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::recent_store(
    DeepSeekDsparkRecentStoreLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_dspark_recent_store(launch) : status;
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill error copy is invalid");
  }
  return cuda_status(
      cudaMemcpyAsync(host, reinterpret_cast<const void*>(device),
                      sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync DeepSeek DSpark prefill error D2H");
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill completion identity is invalid");
  }
  return cuda_status(
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaEventRecord DeepSeek DSpark prefill");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekDsparkPrefillStageOperations::query_event(
    std::uintptr_t event) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill completion event is null");
  }
  const auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (result == cudaSuccess) return DeepSeekExpertAsyncStatus::kSuccess;
  if (result == cudaErrorNotReady) {
    return DeepSeekExpertAsyncStatus::kInProgress;
  }
  return cuda_status(result, "cudaEventQuery DeepSeek DSpark prefill");
}

Status NvidiaDeepSeekDsparkPrefillStageOperations::synchronize_stream(
    std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill synchronization stream is null");
  }
  return cuda_status(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
      "cudaStreamSynchronize DeepSeek DSpark prefill cancel");
}

}  // namespace pih
