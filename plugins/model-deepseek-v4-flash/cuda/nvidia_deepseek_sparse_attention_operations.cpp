#include "pih/backend/cuda/nvidia_deepseek_sparse_attention_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {
namespace {


}  // namespace

Result<NvidiaDeepSeekSparseAttentionOperations>
NvidiaDeepSeekSparseAttentionOperations::Create(
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
  return NvidiaDeepSeekSparseAttentionOperations(retained_context, &async_api,
                                                 kernel_api);
}

Status NvidiaDeepSeekSparseAttentionOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekSparseAttentionOperations::validate_host_staging(
    const DeepSeekSparseAttentionHostStaging& staging) {
  if (staging.index_capacity == 0) {
    return Status::InvalidArgument(
        "DeepSeek sparse attention staging capacity is zero");
  }
  if (async_api_ != nullptr) {
    auto validate = [this](const void* pointer) {
      return plugin_status(async_api_->validate_pinned_host(
          async_api_->context, context_identity_,
          reinterpret_cast<std::uintptr_t>(pointer)));
    };
    auto status = validate(staging.indices);
    if (!status.ok()) return status;
    return validate(staging.error_flag);
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekSparseAttentionOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek sparse attention zero is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekSparseAttentionOperations::copy_h2d_async(
    std::uintptr_t device, const void* host, std::size_t bytes,
    std::uintptr_t stream) {
  if (device == 0 || host == nullptr || bytes == 0 || stream == 0)
    return Status::InvalidArgument(
        "DeepSeek sparse attention H2D copy is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, device,
        reinterpret_cast<std::uintptr_t>(host), bytes,
        PIH_CUDA_COPY_H2D_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekSparseAttentionOperations::attention(
    DeepSeekSparseAttentionLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_sparse_attention_launch_v1 request{
      sizeof(pih_deepseek_sparse_attention_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.query_bf16,
      launch.latent_kv_bf16, launch.attention_sink_f32, launch.indices_i32,
      launch.output_bf16, launch.error_flag_u32, launch.stream,
      launch.query_count, launch.head_count, launch.kv_count,
      launch.index_count, launch.compressed_kv_bf16, launch.page_slots_u32,
      launch.recent_physical_offset, launch.compressed_physical_offset,
      launch.compressed_slot_count, launch.logical_page_count,
      launch.physical_page_count};
  return deepseek_kernel_status(kernels_->launch_sparse_attention(&request));
}

Status NvidiaDeepSeekSparseAttentionOperations::copy_d2h_async(
    void* host, std::uintptr_t device, std::size_t bytes,
    std::uintptr_t stream) {
  if (host == nullptr || device == 0 || bytes == 0 || stream == 0)
    return Status::InvalidArgument(
        "DeepSeek sparse attention D2H copy is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device, bytes,
        PIH_CUDA_COPY_D2H_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
