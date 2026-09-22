#include "pih/backend/cuda/nvidia_deepseek_recent_state_operations.h"

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {


Result<NvidiaDeepSeekRecentStateOperations>
NvidiaDeepSeekRecentStateOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.copy_async == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekRecentStateOperations(retained_context, &async_api);
}


Status NvidiaDeepSeekRecentStateOperations::copy_d2d_async(
    std::uintptr_t destination, std::uintptr_t source, std::size_t bytes,
    std::uintptr_t stream) {
  if (destination == 0 || source == 0 || bytes == 0 || stream == 0) {
    return Status::InvalidArgument("DeepSeek recent state D2D copy is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, destination, source, bytes,
        PIH_CUDA_COPY_D2D_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
