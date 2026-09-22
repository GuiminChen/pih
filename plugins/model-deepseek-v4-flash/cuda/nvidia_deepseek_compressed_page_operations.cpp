#include "pih/backend/cuda/nvidia_deepseek_compressed_page_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {
Result<NvidiaDeepSeekCompressedPageOperations>
NvidiaDeepSeekCompressedPageOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels) {
  const auto* kernel_api = &kernels;
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.validate_pinned_host == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.memset_async == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekCompressedPageOperations(retained_context, &async_api,
                                                kernel_api);
}

Status NvidiaDeepSeekCompressedPageOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressedPageOperations::validate_host_error(
    std::uint32_t* host_error) {
  if (host_error == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek compressed page host error is null");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host_error)));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressedPageOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek compressed page error clear is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressedPageOperations::copy_d2d_async(
    std::uintptr_t destination, std::uintptr_t source,
    std::size_t bytes, std::uintptr_t stream) {
  if (destination == 0 || source == 0 || bytes == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek compressed page D2D copy is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, destination, source, bytes,
        PIH_CUDA_COPY_D2D_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressedPageOperations::store(
    DeepSeekCompressorBf16StoreLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_compressor_store_launch_v1 request{
      sizeof(pih_deepseek_compressor_store_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.compressed_f32,
      launch.rms_weight_bf16, launch.cos_sin_cache_f32,
      launch.destination_bf16, launch.error_flag_u32, launch.stream,
      launch.head_dim, launch.rope_head_dim, launch.rope_position,
      launch.rms_epsilon};
  return deepseek_kernel_status(kernels_->launch_compressor_store(&request));
}

Status NvidiaDeepSeekCompressedPageOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek compressed page error copy is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device,
        sizeof(std::uint32_t), PIH_CUDA_COPY_D2H_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
