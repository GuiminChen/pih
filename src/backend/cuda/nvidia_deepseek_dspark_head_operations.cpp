#include "pih/backend/cuda/nvidia_deepseek_dspark_head_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"

namespace pih {
Result<NvidiaDeepSeekDsparkHeadOperations>
NvidiaDeepSeekDsparkHeadOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark head");
  if (context == nullptr)
    return Status::FailedPrecondition(
        "DeepSeek DSpark head requires a current CUDA context");
  return NvidiaDeepSeekDsparkHeadOperations(
      reinterpret_cast<std::uint64_t>(context));
}
Status NvidiaDeepSeekDsparkHeadOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DeepSeek DSpark head");
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark head CUDA context identity changed");
}
Status NvidiaDeepSeekDsparkHeadOperations::validate_host_error(
    std::uint32_t* host_error) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host_error == nullptr)
    return Status::InvalidArgument("DeepSeek DSpark host error is null");
  cudaPointerAttributes attributes{};
  const auto result = cudaPointerGetAttributes(&attributes, host_error);
  if (result != cudaSuccess)
    return cuda_status(result,
                       "cudaPointerGetAttributes DeepSeek DSpark error");
  return attributes.type == cudaMemoryTypeHost
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DeepSeek DSpark error is not pinned host memory");
}
Status NvidiaDeepSeekDsparkHeadOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek DSpark error clear is invalid");
  return cuda_status(cudaMemsetAsync(
      reinterpret_cast<void*>(device), 0, sizeof(std::uint32_t),
      reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemsetAsync DeepSeek DSpark error");
}
Status NvidiaDeepSeekDsparkHeadOperations::markov(
    DeepSeekDsparkMarkovLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_dspark_markov(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::hc_head(
    DeepSeekHcHeadLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_hc_head(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::rms_norm(
    DeepSeekRmsNormLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_rms_norm(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::lm_head(
    DeepSeekLmHeadLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_lm_head(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::argmax(
    DeepSeekArgmaxLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_argmax(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::confidence(
    DeepSeekDsparkConfidenceLaunch launch) {
  auto status = require_context();
  return status.ok() ? launch_deepseek_dspark_confidence(launch) : status;
}
Status NvidiaDeepSeekDsparkHeadOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek DSpark error copy is invalid");
  return cuda_status(cudaMemcpyAsync(
      host, reinterpret_cast<const void*>(device), sizeof(std::uint32_t),
      cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)),
      "cudaMemcpyAsync DeepSeek DSpark error D2H");
}
}  // namespace pih
