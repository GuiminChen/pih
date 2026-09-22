#include "engram_completion.h"
#include <cuda_runtime_api.h>
#include <cstring>
#include <limits>
#include <string>

namespace pih::deepseek_v41 {
Status ValidateEngramCompletionResources(const EngramCompletionResources& x) {
  if (!x.event || !x.host_error_flag.address || x.host_error_flag.address % 4 ||
      x.host_error_flag.bytes != 4 ||
      x.host_error_flag.address > std::numeric_limits<std::uintptr_t>::max() - 4)
    return Status::InvalidArgument("Engram completion requires an event and aligned four-byte host slot");
  cudaPointerAttributes attributes{};
  auto result = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(x.host_error_flag.address));
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  if (attributes.type != cudaMemoryTypeHost)
    return Status::FailedPrecondition("Engram error readback requires pinned host memory");
  result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(x.event));
  if (result == cudaErrorNotReady)
    return Status::FailedPrecondition("Engram completion event is still in use");
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  return Status::Ok();
}
EngramCompletion::EngramCompletion(EngramCompletion&& other) noexcept
    : resources_(other.resources_), failed_(other.failed_), complete_(other.complete_) {
  other.failed_ = true;
  other.resources_ = {};
}
Result<EngramCompletion> EngramCompletion::Record(const EngramGateLaunch& gate,
    const EngramCompletionResources& resources) {
  const auto layout = ValidateEngramGate(gate);
  if (!layout.ok()) return layout;
  return RecordFlag(gate.error_flag, gate.stream, resources);
}
Result<EngramCompletion> EngramCompletion::RecordFlag(EngramDeviceRegion error_flag,
    std::uintptr_t stream_handle, const EngramCompletionResources& resources) {
  if (!stream_handle || !error_flag.address || error_flag.address % 4 || error_flag.bytes != 4 ||
      error_flag.address > std::numeric_limits<std::uintptr_t>::max() - 4)
    return Status::InvalidArgument("Completion requires an aligned device error flag and explicit stream");
  const auto resource_status = ValidateEngramCompletionResources(resources);
  if (!resource_status.ok()) return resource_status;
  cudaPointerAttributes attributes{};
  auto result = cudaPointerGetAttributes(&attributes, reinterpret_cast<void*>(error_flag.address));
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  int device = -1;
  result = cudaGetDevice(&device);
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    return Status::FailedPrecondition("Engram device error flag belongs to a different memory domain");
  result = cudaPeekAtLastError();
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  const auto stream = reinterpret_cast<cudaStream_t>(stream_handle);
  result = cudaMemcpyAsync(reinterpret_cast<void*>(resources.host_error_flag.address),
      reinterpret_cast<const void*>(error_flag.address), 4, cudaMemcpyDeviceToHost, stream);
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  result = cudaEventRecord(reinterpret_cast<cudaEvent_t>(resources.event), stream);
  if (result != cudaSuccess) return Status::Internal(cudaGetErrorString(result));
  EngramCompletion completion;
  completion.resources_ = resources;
  completion.failed_ = false;
  return completion;
}
Result<bool> EngramCompletion::Poll() {
  if (failed_ || !resources_.event)
    return Status::FailedPrecondition("Engram completion failed or was moved from");
  if (complete_) return true;
  const auto result = cudaEventQuery(reinterpret_cast<cudaEvent_t>(resources_.event));
  if (result == cudaErrorNotReady) return false;
  if (result != cudaSuccess) {
    failed_ = true;
    return Status::Internal(cudaGetErrorString(result));
  }
  std::uint32_t device_error = 0;
  std::memcpy(&device_error, reinterpret_cast<const void*>(resources_.host_error_flag.address), 4);
  if (device_error) {
    failed_ = true;
    return Status::FailedPrecondition("Engram device error flag: " + std::to_string(device_error));
  }
  complete_ = true;
  return true;
}
}  // namespace pih::deepseek_v41
