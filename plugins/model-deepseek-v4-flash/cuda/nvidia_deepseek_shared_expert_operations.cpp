#include "pih/backend/cuda/nvidia_deepseek_shared_expert_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"
#include <limits>

namespace pih {

Result<NvidiaDeepSeekSharedExpertOperations>
NvidiaDeepSeekSharedExpertOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.validate_pinned_host == nullptr || async_api.copy_async == nullptr ||
      async_api.record_event == nullptr || async_api.query_event == nullptr ||
      async_api.memset_async == nullptr || kernels.struct_size != sizeof(kernels) ||
      kernels.contract_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      kernels.launch_fp8_activation_quant == nullptr || kernels.launch_fp8_gemm == nullptr ||
      kernels.launch_shared_swiglu == nullptr || kernels.launch_expert_finalize == nullptr) {
    return Status::InvalidArgument("DeepSeek shared expert plugin capabilities are incomplete");
  }
  return NvidiaDeepSeekSharedExpertOperations(
      retained_context, &async_api, &kernels);
}

Status NvidiaDeepSeekSharedExpertOperations::require_context() const {
  return plugin_status(async_api_->activate_context(async_api_->context, context_identity_));
}

Status NvidiaDeepSeekSharedExpertOperations::validate_host_error(std::uint32_t* host) {
  if (host == nullptr) return Status::InvalidArgument("Shared expert host error is null");
  auto status = require_context();
  if (!status.ok()) return status;
  return plugin_status(async_api_->validate_pinned_host(async_api_->context,
      context_identity_, reinterpret_cast<std::uintptr_t>(host)));
}

Status NvidiaDeepSeekSharedExpertOperations::prepare_moe(
    std::uintptr_t input, std::uintptr_t routed, std::uintptr_t accumulator,
    std::uint32_t tokens, std::uintptr_t stream) {
  if (tokens == 0 || tokens > 4096 || stream == 0 || input == 0 ||
      routed == 0 || accumulator == 0) {
    return Status::InvalidArgument("Invalid MoE preparation geometry");
  }
  const auto input_bytes = static_cast<std::uint64_t>(tokens) * 4096U * 2U;
  const auto accumulator_bytes = input_bytes * 2U;
  const auto limit = std::numeric_limits<std::uintptr_t>::max();
  if (input > limit - input_bytes || routed > limit - input_bytes ||
      accumulator > limit - accumulator_bytes) {
    return Status::InvalidArgument("MoE preparation address overflows");
  }
  const auto overlap = [](std::uintptr_t a, std::uint64_t as,
                          std::uintptr_t b, std::uint64_t bs) {
    return a < b + bs && b < a + as;
  };
  if (overlap(input, input_bytes, routed, input_bytes) ||
      overlap(input, input_bytes, accumulator, accumulator_bytes) ||
      overlap(routed, input_bytes, accumulator, accumulator_bytes)) {
    return Status::InvalidArgument("MoE preparation buffers overlap");
  }
  auto status = require_context();
  if (!status.ok()) return status;
  status = plugin_status(async_api_->copy_async(async_api_->context,
      context_identity_, routed, input, input_bytes, PIH_CUDA_COPY_D2D_V1, stream));
  if (!status.ok()) return status;
  return plugin_status(async_api_->memset_async(async_api_->context,
      context_identity_, accumulator, 0, accumulator_bytes, stream));
}

Status NvidiaDeepSeekSharedExpertOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0) return Status::InvalidArgument("Invalid shared expert error copy");
  auto status = validate_host_error(host);
  if (!status.ok()) return status;
  return plugin_status(async_api_->copy_async(async_api_->context, context_identity_,
      reinterpret_cast<std::uintptr_t>(host), device, sizeof(std::uint32_t),
      PIH_CUDA_COPY_D2H_V1, stream));
}

Status NvidiaDeepSeekSharedExpertOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  if (event == 0 || stream == 0) return Status::InvalidArgument("Invalid shared expert event");
  auto status = require_context();
  if (!status.ok()) return status;
  return plugin_status(async_api_->record_event(async_api_->context,
      context_identity_, event, stream));
}

Result<DeepSeekExpertAsyncStatus> NvidiaDeepSeekSharedExpertOperations::query_event(
    std::uintptr_t event) {
  if (event == 0) return Status::InvalidArgument("Invalid shared expert event");
  auto status = require_context();
  if (!status.ok()) return status;
  std::uint32_t event_status = 0;
  status = plugin_status(async_api_->query_event(async_api_->context,
      context_identity_, event, &event_status));
  if (!status.ok()) return status;
  if (event_status == PIH_CUDA_EVENT_PENDING_V1) return DeepSeekExpertAsyncStatus::kInProgress;
  if (event_status == PIH_CUDA_EVENT_COMPLETE_V1) return DeepSeekExpertAsyncStatus::kSuccess;
  return Status::Internal("Shared expert backend returned an unknown event status");
}

Status NvidiaDeepSeekSharedExpertOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (device == 0 || stream == 0) return Status::InvalidArgument("Invalid shared expert clear");
  return plugin_status(async_api_->memset_async(
      async_api_->context, context_identity_, device, 0, sizeof(std::uint32_t), stream));
}

Status NvidiaDeepSeekSharedExpertOperations::quantize(
    DeepSeekFp8ActivationQuantLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  const pih_deepseek_fp8_activation_quant_launch_v1 request{
      sizeof(pih_deepseek_fp8_activation_quant_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.output_e4m3, launch.scale_bits, launch.error_flag, launch.stream,
      launch.token_count, launch.logical_k};
  return deepseek_kernel_status(kernels_->launch_fp8_activation_quant(&request));
}

Status NvidiaDeepSeekSharedExpertOperations::gemm(
    DeepSeekFp8GemmLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  const pih_deepseek_fp8_gemm_launch_v1 request{
      sizeof(pih_deepseek_fp8_gemm_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.activation_e4m3,
      launch.activation_scale_bits, launch.weight_e4m3,
      launch.weight_scale_bits, launch.output_bf16, launch.error_flag,
      launch.stream, launch.m, launch.n, launch.k,
      static_cast<std::uint32_t>(launch.output_type)};
  return deepseek_kernel_status(kernels_->launch_fp8_gemm(&request));
}

Status NvidiaDeepSeekSharedExpertOperations::swiglu(
    DeepSeekSharedExpertSwiGluLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  const pih_deepseek_shared_swiglu_launch_v1 request{
      sizeof(pih_deepseek_shared_swiglu_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.gate_bf16,
      launch.up_bf16, launch.output_bf16, launch.error_flag, launch.stream,
      launch.token_count};
  return deepseek_kernel_status(kernels_->launch_shared_swiglu(&request));
}

Status NvidiaDeepSeekSharedExpertOperations::finalize(
    DeepSeekExpertFinalizeLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  const pih_deepseek_expert_finalize_launch_v1 request{
      sizeof(pih_deepseek_expert_finalize_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.accumulator_f32,
      launch.shared_output_bf16, launch.output_bf16, launch.error_flag,
      launch.stream, launch.token_count};
  return deepseek_kernel_status(kernels_->launch_expert_finalize(&request));
}

}  // namespace pih
