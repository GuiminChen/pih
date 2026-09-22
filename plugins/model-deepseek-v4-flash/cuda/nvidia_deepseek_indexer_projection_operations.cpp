#include "pih/backend/cuda/nvidia_deepseek_indexer_projection_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Result<NvidiaDeepSeekIndexerProjectionOperations>
NvidiaDeepSeekIndexerProjectionOperations::Create(
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
  return NvidiaDeepSeekIndexerProjectionOperations(retained_context,
                                                   &async_api, kernel_api);
}

Status NvidiaDeepSeekIndexerProjectionOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekIndexerProjectionOperations::validate_host_error(
    std::uint32_t* host) {
  if (host == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection host error is null");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host)));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekIndexerProjectionOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection clear is invalid");
  }
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekIndexerProjectionOperations::project(
    DeepSeekIndexerProjectionLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_indexer_projection_launch_v1 request{
      sizeof(pih_deepseek_indexer_projection_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.qr_bf16,
      launch.hidden_bf16, launch.wq_b_e4m3, launch.wq_b_scale_bits,
      launch.qr_e4m3, launch.qr_scale_bits, launch.weights_proj_bf16,
      launch.frequencies_f32, launch.positions_u32, launch.query_bf16,
      launch.head_weight_f32, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.table_position_count};
  return deepseek_kernel_status(kernels_->launch_indexer_projection(&request));
}

Status NvidiaDeepSeekIndexerProjectionOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection error copy is invalid");
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
