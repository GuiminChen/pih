#include "pih/backend/cuda/nvidia_deepseek_dspark_decode_stage_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

namespace pih {

Result<NvidiaDeepSeekDsparkDecodeStageOperations>
NvidiaDeepSeekDsparkDecodeStageOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(
        result, "cuCtxGetCurrent DeepSeek DSpark decode");
  }
  if (context == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark decode requires a current CUDA context");
  }
  return NvidiaDeepSeekDsparkDecodeStageOperations(
      reinterpret_cast<std::uint64_t>(context));
}

Status NvidiaDeepSeekDsparkDecodeStageOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS) {
    return cuda_driver_status(
        result, "cuCtxGetCurrent DeepSeek DSpark decode");
  }
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark decode CUDA context identity changed");
}

Status NvidiaDeepSeekDsparkDecodeStageOperations::validate_host_error(
    std::uint32_t* host_error) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host_error == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode host error is null");
  }
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(&attributes, host_error);
  if (result != cudaSuccess) {
    return cuda_status(
        result, "cudaPointerGetAttributes DeepSeek DSpark decode error");
  }
  return attributes.type == cudaMemoryTypeHost
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark decode error is not pinned host memory");
}

Status NvidiaDeepSeekDsparkDecodeStageOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode error clear is invalid");
  }
  return cuda_status(
      cudaMemsetAsync(reinterpret_cast<void*>(device), 0,
                      sizeof(std::uint32_t),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek DSpark decode error");
}

#define PIH_DSPARK_CONTEXT_LAUNCH(method, launch_function, type) \
  Status NvidiaDeepSeekDsparkDecodeStageOperations::method(type launch) { \
    auto status = require_context();                                    \
    return status.ok() ? launch_function(launch) : status;              \
  }

PIH_DSPARK_CONTEXT_LAUNCH(
    positions, launch_deepseek_dspark_positions,
    DeepSeekDsparkPositionLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    mhc_pre, launch_deepseek_mhc_pre, DeepSeekMhcPreLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    quant, launch_deepseek_fp8_activation_quant,
    DeepSeekFp8ActivationQuantLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    gemm, launch_deepseek_fp8_gemm, DeepSeekFp8GemmLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    rms, launch_deepseek_rms_norm, DeepSeekRmsNormLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    head_rms, launch_deepseek_head_rms, DeepSeekHeadRmsLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    rotary, launch_deepseek_rotary, DeepSeekRotaryLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    kv_simulate, launch_deepseek_kv_fp8_simulate,
    DeepSeekKvFp8SimulateLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    recent_store, launch_deepseek_dspark_recent_store,
    DeepSeekDsparkRecentStoreLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    attention, launch_deepseek_dspark_attention,
    DeepSeekDsparkAttentionLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    grouped_gemm, launch_deepseek_grouped_fp8_gemm,
    DeepSeekGroupedFp8GemmLaunch)
PIH_DSPARK_CONTEXT_LAUNCH(
    mhc_post, launch_deepseek_mhc_post, DeepSeekMhcPostLaunch)

#undef PIH_DSPARK_CONTEXT_LAUNCH

Status NvidiaDeepSeekDsparkDecodeStageOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode error copy is invalid");
  }
  return cuda_status(
      cudaMemcpyAsync(host, reinterpret_cast<const void*>(device),
                      sizeof(std::uint32_t), cudaMemcpyDeviceToHost,
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync DeepSeek DSpark decode error D2H");
}

Status NvidiaDeepSeekDsparkDecodeStageOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode completion identity is invalid");
  }
  return cuda_status(
      cudaEventRecord(reinterpret_cast<cudaEvent_t>(event),
                      reinterpret_cast<cudaStream_t>(stream)),
      "cudaEventRecord DeepSeek DSpark decode");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekDsparkDecodeStageOperations::query_event(
    std::uintptr_t event) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (event == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode completion event is null");
  }
  const auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(event));
  if (result == cudaSuccess) return DeepSeekExpertAsyncStatus::kSuccess;
  if (result == cudaErrorNotReady) {
    return DeepSeekExpertAsyncStatus::kInProgress;
  }
  return cuda_status(result, "cudaEventQuery DeepSeek DSpark decode");
}

Status NvidiaDeepSeekDsparkDecodeStageOperations::synchronize_stream(
    std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark decode synchronization stream is null");
  }
  return cuda_status(
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
      "cudaStreamSynchronize DeepSeek DSpark decode cancel");
}

}  // namespace pih
