#include "pih/backend/cuda/nvidia_deepseek_h2d_runtime.h"


#include <limits>

#include "pih/plugin_sdk/status_bridge.h"


namespace pih {


Result<NvidiaDeepSeekH2dRuntime> NvidiaDeepSeekH2dRuntime::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.copy_async == nullptr ||
      async_api.record_event == nullptr || async_api.query_event == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekH2dRuntime(retained_context, &async_api, nullptr);
}

Result<NvidiaDeepSeekH2dRuntime> NvidiaDeepSeekH2dRuntime::Create(
    std::uintptr_t retained_context,
    const pih_memory_host_spill_api_v1& host_spill_api) {
  if (retained_context == 0 ||
      host_spill_api.struct_size != sizeof(host_spill_api) ||
      host_spill_api.contract_version !=
          PIH_MEMORY_HOST_SPILL_ABI_VERSION_V1 ||
      host_spill_api.context == nullptr ||
      host_spill_api.copy_h2d_async == nullptr ||
      host_spill_api.record_event == nullptr ||
      host_spill_api.query_event == nullptr) {
    return Status::InvalidArgument("Host-spill transfer capability is invalid");
  }
  return NvidiaDeepSeekH2dRuntime(retained_context, nullptr,
                                  &host_spill_api);
}


Status NvidiaDeepSeekH2dRuntime::copy_async(
    std::uintptr_t destination, std::uintptr_t source, std::uint64_t bytes,
    std::uintptr_t stream) {
  if (destination == 0 || source == 0 || bytes == 0 || stream == 0 ||
      bytes > std::numeric_limits<std::size_t>::max()) {
    return Status::InvalidArgument("DeepSeek H2D copy arguments are invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_, destination, source, bytes,
        PIH_CUDA_COPY_H2D_V1, stream));
  }
  if (host_spill_api_ != nullptr) {
    return plugin_status(host_spill_api_->copy_h2d_async(
        host_spill_api_->context, context_, destination, source, bytes,
        stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekH2dRuntime::record_event(std::uintptr_t event,
                                               std::uintptr_t stream) {
  if (event == 0 || stream == 0) {
    return Status::InvalidArgument("DeepSeek H2D event arguments are invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->record_event(
        async_api_->context, context_, event, stream));
  }
  if (host_spill_api_ != nullptr) {
    return plugin_status(host_spill_api_->record_event(
        host_spill_api_->context, context_, event, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Result<DeepSeekTransferEventStatus> NvidiaDeepSeekH2dRuntime::query_event(
    std::uintptr_t event) {
  if (event == 0) return Status::InvalidArgument("DeepSeek H2D event is null");
  if (async_api_ != nullptr) {
    std::uint32_t event_status = 0;
    auto status = plugin_status(async_api_->query_event(
        async_api_->context, context_, event, &event_status));
    if (!status.ok()) return status;
    if (event_status == PIH_CUDA_EVENT_COMPLETE_V1)
      return DeepSeekTransferEventStatus::kSuccess;
    if (event_status == PIH_CUDA_EVENT_PENDING_V1)
      return DeepSeekTransferEventStatus::kNotReady;
    return Status::Internal("CUDA async capability returned invalid event state");
  }
  if (host_spill_api_ != nullptr) {
    std::uint32_t event_status = 0;
    auto status = plugin_status(host_spill_api_->query_event(
        host_spill_api_->context, context_, event, &event_status));
    if (!status.ok()) return status;
    if (event_status == PIH_CUDA_EVENT_COMPLETE_V1)
      return DeepSeekTransferEventStatus::kSuccess;
    if (event_status == PIH_CUDA_EVENT_PENDING_V1)
      return DeepSeekTransferEventStatus::kNotReady;
    return Status::Internal(
        "Host-spill capability returned invalid event state");
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
