#include "pih/backend/cuda/nvidia_deepseek_attention_projection_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Result<NvidiaDeepSeekAttentionProjectionOperations>
NvidiaDeepSeekAttentionProjectionOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels) {
  const auto* kernel_api = &kernels;
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.validate_pinned_host == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.memset_async == nullptr)
    return Status::InvalidArgument("CUDA async capability is invalid");
  return NvidiaDeepSeekAttentionProjectionOperations(
      retained_context, &async_api, kernel_api);
}
Status NvidiaDeepSeekAttentionProjectionOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekAttentionProjectionOperations::validate_host_error(
    std::uint32_t* host) {
  if (host == nullptr) return Status::InvalidArgument("DeepSeek projection host error is null");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host)));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekAttentionProjectionOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) return Status::InvalidArgument("DeepSeek projection clear is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekAttentionProjectionOperations::quant(
    DeepSeekFp8ActivationQuantLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_fp8_activation_quant_launch_v1 request{
      sizeof(pih_deepseek_fp8_activation_quant_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.input_bf16, launch.output_e4m3, launch.scale_bits,
      launch.error_flag, launch.stream, launch.token_count, launch.logical_k};
  return deepseek_kernel_status(kernels_->launch_fp8_activation_quant(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::gemm(
    DeepSeekFp8GemmLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_fp8_gemm_launch_v1 request{
      sizeof(pih_deepseek_fp8_gemm_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.activation_e4m3, launch.activation_scale_bits,
      launch.weight_e4m3, launch.weight_scale_bits, launch.output_bf16,
      launch.error_flag, launch.stream, launch.m, launch.n, launch.k,
      static_cast<std::uint32_t>(launch.output_type)};
  return deepseek_kernel_status(kernels_->launch_fp8_gemm(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::head_rms(
    DeepSeekHeadRmsLaunch launch) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_head_rms_launch_v1 request{
      sizeof(pih_deepseek_head_rms_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.output_bf16, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.head_count, launch.head_dimension,
      launch.epsilon};
  return deepseek_kernel_status(kernels_->launch_head_rms(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::rotary(
    DeepSeekRotaryLaunch launch) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_rotary_launch_v1 request{
      sizeof(pih_deepseek_rotary_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.frequencies_f32, launch.error_flag_u32, launch.stream,
      launch.token_count, launch.head_count, launch.head_dimension,
      launch.rope_dimension, launch.inverse ? 1U : 0U,
      launch.positions_u32, launch.table_position_count};
  return deepseek_kernel_status(kernels_->launch_rotary(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::kv_simulate(
    DeepSeekKvFp8SimulateLaunch launch) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_kv_fp8_simulate_launch_v1 request{
      sizeof(pih_deepseek_kv_fp8_simulate_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.kv_bf16,
      launch.error_flag_u32, launch.stream, launch.token_count,
      launch.vector_dimension, launch.quantized_dimension, launch.group_size};
  return deepseek_kernel_status(kernels_->launch_kv_fp8_simulate(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::rms(
    DeepSeekRmsNormLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_rms_norm_launch_v1 request{
      sizeof(pih_deepseek_rms_norm_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.input_bf16, launch.weight_bf16, launch.output_bf16,
      launch.error_flag, launch.stream, launch.rows, launch.hidden_size,
      launch.epsilon};
  return deepseek_kernel_status(kernels_->launch_rms_norm(&request));
}
Status NvidiaDeepSeekAttentionProjectionOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek projection error copy is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device,
        sizeof(std::uint32_t), PIH_CUDA_COPY_D2H_V1, stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
}  // namespace pih
