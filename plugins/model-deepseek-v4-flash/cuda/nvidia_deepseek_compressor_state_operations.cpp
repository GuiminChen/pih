#include "pih/backend/cuda/nvidia_deepseek_compressor_state_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Result<NvidiaDeepSeekCompressorStateOperations>
NvidiaDeepSeekCompressorStateOperations::Create(
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
  return NvidiaDeepSeekCompressorStateOperations(retained_context, &async_api,
                                                 kernel_api);
}

Status NvidiaDeepSeekCompressorStateOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressorStateOperations::validate_host_error(
    std::uint32_t* host_error) {
  if (host_error == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek compressor state host error is null");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host_error)));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressorStateOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek compressor error clear is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekCompressorStateOperations::pooling(
    DeepSeekCompressorPoolingLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_compressor_pooling_launch_v1 request{
      sizeof(pih_deepseek_compressor_pooling_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.kv_projection_f32,
      launch.gate_projection_f32, launch.ape_row_f32, launch.kv_state_f32,
      launch.score_state_f32, launch.output_f32, launch.error_flag_u32,
      launch.stream, launch.batch_count, launch.ratio, launch.head_dim,
      launch.absolute_position};
  return deepseek_kernel_status(kernels_->launch_compressor_pooling(&request));
}

Status NvidiaDeepSeekCompressorStateOperations::projection(
    DeepSeekCompressorBf16ProjectionLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_compressor_projection_launch_v1 request{
      sizeof(pih_deepseek_compressor_projection_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.kv_weight_bf16, launch.gate_weight_bf16, launch.kv_projection_f32,
      launch.gate_projection_f32, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.ratio, launch.head_dim, launch.hidden_size};
  return deepseek_kernel_status(
      kernels_->launch_compressor_projection(&request));
}

Status NvidiaDeepSeekCompressorStateOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek compressor error copy is invalid");
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
