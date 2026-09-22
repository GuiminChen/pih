#include "pih/backend/cuda/nvidia_deepseek_learned_router_operations.h"

#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {
Result<NvidiaDeepSeekLearnedRouterOperations>
NvidiaDeepSeekLearnedRouterOperations::Create(
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api,
    const pih_deepseek_kernels_api_v1& kernels) {
  const auto* kernel_api = &kernels;
  if (retained_context == 0 || async_api.struct_size != sizeof(async_api) ||
      async_api.contract_version != PIH_NVIDIA_CUDA_ASYNC_ABI_VERSION_V1 ||
      async_api.context == nullptr || async_api.activate_context == nullptr ||
      async_api.copy_async == nullptr ||
      async_api.memset_async == nullptr || async_api.record_event == nullptr ||
      async_api.query_event == nullptr) {
    return Status::InvalidArgument("CUDA async capability is invalid");
  }
  return NvidiaDeepSeekLearnedRouterOperations(retained_context, &async_api,
                                               kernel_api);
}
Status NvidiaDeepSeekLearnedRouterOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekLearnedRouterOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek learned router zero is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekLearnedRouterOperations::gemm(
    DeepSeekRouterBf16GemmLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_router_gemm_launch_v1 request{
      sizeof(pih_deepseek_router_gemm_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.weight_bf16, launch.scores_f32, launch.error_flag_u32,
      launch.stream, launch.token_count, launch.expert_count,
      launch.hidden_size};
  return deepseek_kernel_status(kernels_->launch_router_gemm(&request));
}
Status NvidiaDeepSeekLearnedRouterOperations::copy_d2h_async(
    void* host, std::uintptr_t device, std::uint64_t bytes,
    std::uintptr_t stream) {
  if (host == nullptr || device == 0 || bytes == 0 || stream == 0)
    return Status::InvalidArgument(
        "DeepSeek learned router D2H copy is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device, bytes,
        PIH_CUDA_COPY_D2H_V1, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Status NvidiaDeepSeekLearnedRouterOperations::record_event(
    std::uintptr_t event, std::uintptr_t stream) {
  if (event == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek learned router event is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->record_event(
        async_api_->context, context_identity_, event, stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}
Result<DeepSeekExpertAsyncStatus>
NvidiaDeepSeekLearnedRouterOperations::query_event(std::uintptr_t event) {
  if (event == 0)
    return Status::InvalidArgument(
        "DeepSeek learned router event is invalid");
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
