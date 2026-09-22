#include "pih/backend/cuda/nvidia_deepseek_mhc_sequence_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Result<NvidiaDeepSeekMhcSequenceOperations>
NvidiaDeepSeekMhcSequenceOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    NvidiaDeepSeekMhcBranchDriver& branch,
    const pih_deepseek_kernels_api_v1& kernels) {
  const auto* kernel_api = &kernels;
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.validate_pinned_host == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.memset_async == nullptr)
    return Status::InvalidArgument("CUDA async capability is invalid");
  return NvidiaDeepSeekMhcSequenceOperations(retained_context, &async_api,
                                             branch, kernel_api);
}

Status NvidiaDeepSeekMhcSequenceOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekMhcSequenceOperations::validate_host_error(
    std::uint32_t* host_error) {
  if (host_error == nullptr) {
    return Status::InvalidArgument("DeepSeek mHC host error is null");
  }
  if (async_api_ != nullptr)
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host_error)));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekMhcSequenceOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) {
    return Status::InvalidArgument("DeepSeek mHC error clear is invalid");
  }
  if (async_api_ != nullptr)
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekMhcSequenceOperations::pre(
    DeepSeekMhcPreLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_mhc_pre_launch_v1 request{
      sizeof(pih_deepseek_mhc_pre_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.residual_bf16,
      launch.fn_f32, launch.scale_f32, launch.base_f32,
      launch.norm_weight_bf16, launch.post_mix_f32, launch.residual_mix_f32,
      launch.layer_input_bf16, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.hidden_size, launch.rms_epsilon,
      launch.pre_epsilon, launch.sinkhorn_epsilon, launch.post_multiplier,
      launch.sinkhorn_iterations};
  return deepseek_kernel_status(kernels_->launch_mhc_pre(&request));
}

Status NvidiaDeepSeekMhcSequenceOperations::branch(
    DeepSeekMhcBranchLaunch launch) {
  auto status = require_context();
  return status.ok() ? branch_->launch(launch) : status;
}

Status NvidiaDeepSeekMhcSequenceOperations::post(
    DeepSeekMhcPostLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_mhc_post_launch_v1 request{
      sizeof(pih_deepseek_mhc_post_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.layer_output_bf16,
      launch.residual_bf16, launch.post_mix_f32, launch.residual_mix_f32,
      launch.output_bf16, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.hidden_size};
  return deepseek_kernel_status(kernels_->launch_mhc_post(&request));
}

Status NvidiaDeepSeekMhcSequenceOperations::target_hidden_tap(
    DeepSeekMhcTargetHiddenTapLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_mhc_target_tap_launch_v1 request{
      sizeof(pih_deepseek_mhc_target_tap_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.residual_hc_bf16,
      launch.target_hidden_bf16, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.hidden_size, launch.source_stream_count,
      launch.target_stage_count, launch.target_stage_index};
  return deepseek_kernel_status(kernels_->launch_mhc_target_tap(&request));
}

Status NvidiaDeepSeekMhcSequenceOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument("DeepSeek mHC error copy is invalid");
  }
  if (async_api_ != nullptr)
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device,
        sizeof(std::uint32_t), PIH_CUDA_COPY_D2H_V1, stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
