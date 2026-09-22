#include "pih/backend/cuda/nvidia_deepseek_expert_cuda_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {
namespace {


}  // namespace


Result<NvidiaDeepSeekExpertCudaOperations>
NvidiaDeepSeekExpertCudaOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels) {
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.validate_pinned_host == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.memset_async == nullptr || async_api.record_event == nullptr ||
      async_api.query_event == nullptr)
    return Status::InvalidArgument("CUDA async capability is invalid");
  return NvidiaDeepSeekExpertCudaOperations(
      retained_context, &async_api,
      &kernels);
}

Status NvidiaDeepSeekExpertCudaOperations::validate_host_staging(
    const DeepSeekExpertHostStaging& staging) {
  if (staging.capacity == 0) {
    return Status::InvalidArgument("DeepSeek host staging capacity is zero");
  }
  if (async_api_ != nullptr) {
    auto validate = [this](const void* pointer) {
      return plugin_status(async_api_->validate_pinned_host(
          async_api_->context, context_identity_,
          reinterpret_cast<std::uintptr_t>(pointer)));
    };
    auto status = validate(staging.route_weights);
    if (!status.ok()) return status;
    status = validate(staging.token_indices);
    if (!status.ok()) return status;
    return validate(staging.error_flag);
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekExpertCudaOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekExpertCudaOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek error clear is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekExpertCudaOperations::copy_h2d_async(
    std::uintptr_t device, const void* host, std::size_t bytes,
    std::uintptr_t stream) {
  if (device == 0 || host == nullptr || bytes == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek H2D staging copy is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_, device,
        reinterpret_cast<std::uintptr_t>(host), bytes,
        PIH_CUDA_COPY_H2D_V1, stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekExpertCudaOperations::copy_d2h_async(
    void* host, std::uintptr_t device, std::size_t bytes,
    std::uintptr_t stream) {
  if (host == nullptr || device == 0 || bytes == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek D2H error copy is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device, bytes,
        PIH_CUDA_COPY_D2H_V1, stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekExpertCudaOperations::gather(DeepSeekRouteGatherLaunch v) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_route_gather_launch_v1 request{
      sizeof(pih_deepseek_route_gather_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, v.source_hidden_bf16,
      v.token_indices_u32, v.route_hidden_bf16, v.error_flag, v.stream,
      v.route_count, v.packed_token_count};
  return deepseek_kernel_status(kernels_->launch_route_gather(&request));
}
Status NvidiaDeepSeekExpertCudaOperations::quantize(DeepSeekFp8ActivationQuantLaunch v) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_fp8_activation_quant_launch_v1 request{
      sizeof(pih_deepseek_fp8_activation_quant_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      v.input_bf16, v.output_e4m3, v.scale_bits, v.error_flag, v.stream,
      v.token_count, v.logical_k};
  return deepseek_kernel_status(kernels_->launch_fp8_activation_quant(&request));
}
Status NvidiaDeepSeekExpertCudaOperations::gemm(DeepSeekFp4GemmLaunch v) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_fp4_gemm_launch_v1 request{
      sizeof(pih_deepseek_fp4_gemm_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, v.activation_e4m3,
      v.activation_scale_bits, v.packed_weight, v.weight_scale_bits,
      v.output_bf16, v.error_flag, v.stream, v.m, v.n, v.k};
  return deepseek_kernel_status(kernels_->launch_fp4_gemm(&request));
}
Status NvidiaDeepSeekExpertCudaOperations::swiglu(DeepSeekExpertSwiGluLaunch v) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_expert_swiglu_launch_v1 request{
      sizeof(pih_deepseek_expert_swiglu_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, v.gate_bf16, v.up_bf16,
      v.route_weights_f32, v.output_bf16, v.error_flag, v.stream,
      v.token_count};
  return deepseek_kernel_status(kernels_->launch_expert_swiglu(&request));
}
Status NvidiaDeepSeekExpertCudaOperations::accumulate(DeepSeekExpertAccumulateLaunch v) {
  auto status = require_context(); if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_expert_accumulate_launch_v1 request{
      sizeof(pih_deepseek_expert_accumulate_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, v.expert_output_bf16,
      v.token_indices, v.accumulator_f32, v.error_flag, v.stream,
      v.route_count, v.token_count};
  return deepseek_kernel_status(kernels_->launch_expert_accumulate(&request));
}

Status NvidiaDeepSeekExpertCudaOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  if (event == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek expert event is invalid");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->record_event(
        async_api_->context, context_identity_, event, stream));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekExpertCudaOperations::query_event(std::uintptr_t event) {
  if (event == 0)
    return Status::InvalidArgument("DeepSeek expert event is invalid");
  if (async_api_ != nullptr) {
    std::uint32_t event_status = 0;
    auto status = plugin_status(async_api_->query_event(
        async_api_->context, context_identity_, event, &event_status));
    if (!status.ok()) return status;
    if (event_status == PIH_CUDA_EVENT_COMPLETE_V1)
      return DeepSeekExpertAsyncStatus::kSuccess;
    if (event_status == PIH_CUDA_EVENT_PENDING_V1)
      return DeepSeekExpertAsyncStatus::kInProgress;
    return Status::Internal("CUDA async capability returned invalid event state");
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

}  // namespace pih
