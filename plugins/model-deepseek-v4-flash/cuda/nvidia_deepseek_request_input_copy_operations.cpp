#include "pih/backend/cuda/nvidia_deepseek_request_input_copy_operations.h"

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {


Result<NvidiaDeepSeekRequestInputCopyOperations>
NvidiaDeepSeekRequestInputCopyOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.copy_async == nullptr ||
      async_api.synchronize_stream == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekRequestInputCopyOperations(retained_context,
                                                  &async_api);
}

Status NvidiaDeepSeekRequestInputCopyOperations::synchronize(
    std::uintptr_t stream) {
  if (stream == 0 || async_api_ == nullptr)
    return Status::FailedPrecondition("CUDA request input stream is unavailable");
  return plugin_status(async_api_->synchronize_stream(
      async_api_->context, context_identity_, stream));
}


Status NvidiaDeepSeekRequestInputCopyOperations::copy_h2d_async(
    std::uintptr_t destination, const void* source, std::uint64_t bytes,
    std::uintptr_t stream) {
  if (destination == 0 || source == nullptr || bytes == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek request input copy extent is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, destination,
        reinterpret_cast<std::uintptr_t>(source), bytes,
        PIH_CUDA_COPY_H2D_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
