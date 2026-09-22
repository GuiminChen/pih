#include "pih/backend/cuda/nvidia_deepseek_dspark_state_digest_operations.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "pih/backend/cuda/cuda_driver_status.h"
#include "pih/backend/cuda/cuda_status.h"
#include "pih/backend/cuda/deepseek_component_sha256.h"

namespace pih {

Result<NvidiaDeepSeekDsparkStateDigestOperations>
NvidiaDeepSeekDsparkStateDigestOperations::Create() {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DSpark state digest");
  if (context == nullptr)
    return Status::FailedPrecondition(
        "DSpark state digest requires a current CUDA context");
  return NvidiaDeepSeekDsparkStateDigestOperations(
      reinterpret_cast<std::uint64_t>(context));
}

Status NvidiaDeepSeekDsparkStateDigestOperations::require_context() const {
  CUcontext context = nullptr;
  const auto result = cuCtxGetCurrent(&context);
  if (result != CUDA_SUCCESS)
    return cuda_driver_status(result, "cuCtxGetCurrent DSpark state digest");
  return reinterpret_cast<std::uint64_t>(context) == context_identity_
      ? Status::Ok()
      : Status::FailedPrecondition(
            "DSpark state digest CUDA context identity changed");
}

Status NvidiaDeepSeekDsparkStateDigestOperations::launch_component_sha256(
    const DeepSeekDsparkGpuStateDigestSubmission& submission) {
  auto status = require_context();
  if (!status.ok()) return status;
  auto stream = reinterpret_cast<cudaStream_t>(submission.stream);
  status = cuda_status(cudaMemsetAsync(
      reinterpret_cast<void*>(submission.device_error_flag_u32), 0,
      sizeof(std::uint32_t), stream),
      "cudaMemsetAsync DSpark digest error");
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < submission.components.size(); ++index) {
    const auto& component = submission.components[index];
    status = launch_deepseek_component_sha256({
        component.device_address, component.bytes,
        submission.device_digest_workspace + index * sizeof(Sha256Digest),
        submission.device_error_flag_u32, submission.stream});
    if (!status.ok()) return status;
  }
  status = cuda_status(cudaMemcpyAsync(
      submission.host_component_digests.data(),
      reinterpret_cast<const void*>(submission.device_digest_workspace),
      submission.components.size() * sizeof(Sha256Digest),
      cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync DSpark component digests");
  if (!status.ok()) return status;
  status = cuda_status(cudaMemcpyAsync(
      submission.host_error_flag,
      reinterpret_cast<const void*>(submission.device_error_flag_u32),
      sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
      "cudaMemcpyAsync DSpark digest error");
  if (!status.ok()) return status;
  return cuda_status(cudaEventRecord(
      reinterpret_cast<cudaEvent_t>(submission.completion_event), stream),
      "cudaEventRecord DSpark state digest");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekDsparkStateDigestOperations::query_event(
    std::uintptr_t completion_event) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (completion_event == 0)
    return Status::InvalidArgument("DSpark state digest event is invalid");
  const auto result = cudaEventQuery(
      reinterpret_cast<cudaEvent_t>(completion_event));
  if (result == cudaSuccess) return DeepSeekExpertAsyncStatus::kSuccess;
  if (result == cudaErrorNotReady)
    return DeepSeekExpertAsyncStatus::kInProgress;
  return cuda_status(result, "cudaEventQuery DSpark state digest");
}

}  // namespace pih
