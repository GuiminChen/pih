#include "pih/backend/cuda/nvidia_deepseek_fixed_state_operations.h"


#include <limits>

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {


Result<NvidiaDeepSeekFixedStateOperations>
NvidiaDeepSeekFixedStateOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.record_event == nullptr || async_api.query_event == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekFixedStateOperations(retained_context, &async_api);
}


Status NvidiaDeepSeekFixedStateOperations::copy_d2d_async(
    std::uintptr_t destination, std::uintptr_t source, std::uint64_t bytes,
    std::uintptr_t stream) {
  if (destination == 0 || source == 0 || stream == 0 || bytes == 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument(
        "DeepSeek fixed state D2D copy is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, destination, source, bytes,
        PIH_CUDA_COPY_D2D_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekFixedStateOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek fixed state event record is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->record_event(
        async_api_->context, context_identity_, event, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekFixedStateOperations::query_event(std::uintptr_t event) {
  if (event == 0) {
    return Status::InvalidArgument(
        "DeepSeek fixed state event query is invalid");
  }
  if (async_api_ != nullptr) {
    std::uint32_t event_status = 0;
    auto status = plugin_status(async_api_->query_event(
        async_api_->context, context_identity_, event, &event_status));
    if (!status.ok()) return status;
    if (event_status == PIH_CUDA_EVENT_COMPLETE_V1)
      return DeepSeekExpertAsyncStatus::kSuccess;
    if (event_status == PIH_CUDA_EVENT_PENDING_V1)
      return DeepSeekExpertAsyncStatus::kInProgress;
    return Status::Internal("CUDA async capability returned invalid event state");
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
