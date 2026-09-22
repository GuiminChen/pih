#include "pih/backend/cuda/nvidia_deepseek_endpoint_sequence_operations.h"


#include <algorithm>
#include "pih/backend/cuda/kernel_pack_status.h"
#include "pih/plugin_sdk/status_bridge.h"

namespace pih {
namespace {

pih_deepseek_suppressed_tokens_v1 SuppressedTokens(
    const DeepSeekSuppressedTokenSet& source) {
  pih_deepseek_suppressed_tokens_v1 value{};
  value.token_count = source.token_count;
  std::copy_n(source.token_ids,
              DeepSeekSuppressedTokenSet::kMaximumTokenIds,
              value.token_ids);
  return value;
}

}  // namespace

Result<NvidiaDeepSeekEndpointSequenceOperations>
NvidiaDeepSeekEndpointSequenceOperations::Create(
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
  return NvidiaDeepSeekEndpointSequenceOperations(retained_context, &async_api,
                                                  kernel_api);
}

Status NvidiaDeepSeekEndpointSequenceOperations::require_context() const {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->activate_context(
        async_api_->context, context_identity_));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekEndpointSequenceOperations::validate_host_error(
    std::uint32_t* host_error) {
  if (host_error == nullptr)
    return Status::InvalidArgument("DeepSeek endpoint host error is null");
  if (async_api_ != nullptr)
    return plugin_status(async_api_->validate_pinned_host(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host_error)));
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekEndpointSequenceOperations::zero_u32_async(
    std::uintptr_t device, std::uintptr_t stream) {
  if (device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek endpoint error clear is invalid");
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->memset_async(
        async_api_->context, context_identity_, device, 0,
        sizeof(std::uint32_t), stream));
  }
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekEndpointSequenceOperations::embedding(
    DeepSeekEmbeddingLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_embedding_launch_v1 request{
      sizeof(pih_deepseek_embedding_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.token_ids_u32,
      launch.weight_bf16, launch.output_hc_bf16, launch.error_flag_u32,
      launch.stream, launch.token_count, launch.vocab_size,
      launch.hidden_size, launch.hc_multiplicity};
  return deepseek_kernel_status(kernels_->launch_embedding(&request));
}
Status NvidiaDeepSeekEndpointSequenceOperations::hc_head(
    DeepSeekHcHeadLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_hc_head_launch_v1 request{
      sizeof(pih_deepseek_hc_head_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_hc_bf16,
      launch.fn_f32, launch.scale_f32, launch.base_f32, launch.output_bf16,
      launch.error_flag_u32, launch.stream, launch.token_count,
      launch.hidden_size, launch.hc_multiplicity, launch.rms_epsilon,
      launch.hc_epsilon};
  return deepseek_kernel_status(kernels_->launch_hc_head(&request));
}
Status NvidiaDeepSeekEndpointSequenceOperations::rms_norm(
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
Status NvidiaDeepSeekEndpointSequenceOperations::lm_head(
    DeepSeekLmHeadLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  const pih_deepseek_lm_head_launch_v1 request{
      sizeof(pih_deepseek_lm_head_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, launch.input_bf16,
      launch.weight_bf16, launch.logits_f32, launch.error_flag_u32,
      launch.stream, launch.output_rows, launch.vocab_size,
      launch.hidden_size};
  return deepseek_kernel_status(kernels_->launch_lm_head(&request));
}
Status NvidiaDeepSeekEndpointSequenceOperations::sample(
    DeepSeekArgmaxLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  pih_deepseek_argmax_launch_v1 request{
      sizeof(pih_deepseek_argmax_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.logits_f32, launch.token_id_u32, launch.error_flag_u32,
      launch.stream, launch.vocab_size, launch.selected_logprob_f32,
      launch.top_logprobs_ids_u32, launch.top_logprobs_f32,
      launch.top_logprobs_count, SuppressedTokens(launch.suppressed_tokens)};
  return deepseek_kernel_status(kernels_->launch_argmax(&request));
}
Status NvidiaDeepSeekEndpointSequenceOperations::stochastic_sample(
    DeepSeekStochasticSampleLaunch launch) {
  auto status = require_context();
  if (!status.ok()) return status;
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  pih_deepseek_stochastic_sample_launch_v1 request{
      sizeof(pih_deepseek_stochastic_sample_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.logits_f32, launch.token_id_u32, launch.selected_logprob_f32,
      launch.rng_word_u32, launch.workspace_values_f32,
      launch.workspace_ids_u32, launch.error_flag_u32, launch.stream,
      launch.vocab_size, launch.workspace_capacity, launch.temperature,
      launch.top_p, launch.top_k, launch.seed, launch.sample_ordinal,
      launch.top_logprobs_ids_u32, launch.top_logprobs_f32,
      launch.top_logprobs_count, SuppressedTokens(launch.suppressed_tokens)};
  return deepseek_kernel_status(kernels_->launch_stochastic_sample(&request));
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_d2h_async(
    void* host, std::uintptr_t device, std::size_t bytes,
    std::uintptr_t stream, const char* operation) {
  if (async_api_ != nullptr) {
    return plugin_status(async_api_->copy_async(
        async_api_->context, context_identity_,
        reinterpret_cast<std::uintptr_t>(host), device, bytes,
        PIH_CUDA_COPY_D2H_V1, stream));
  }
  (void)operation;
  return Status::FailedPrecondition(
      "CUDA async capability is required by the native model plugin");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_token_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0) {
    return Status::InvalidArgument("DeepSeek sampled token copy is invalid");
  }
  return copy_d2h_async(host, device, sizeof(std::uint32_t), stream,
                        "cudaMemcpyAsync DeepSeek sampled token D2H");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_error_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek endpoint error copy is invalid");
  return copy_d2h_async(host, device, sizeof(std::uint32_t), stream,
                        "cudaMemcpyAsync DeepSeek endpoint error D2H");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_logprob_d2h_async(
    float* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek logprob copy is invalid");
  return copy_d2h_async(host, device, sizeof(float), stream,
                        "cudaMemcpyAsync DeepSeek logprob D2H");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_rng_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0)
    return Status::InvalidArgument("DeepSeek RNG copy is invalid");
  return copy_d2h_async(host, device, sizeof(std::uint32_t), stream,
                        "cudaMemcpyAsync DeepSeek RNG D2H");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_top_ids_d2h_async(
    std::uint32_t* host, std::uintptr_t device, std::uint32_t count,
    std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0 || count == 0 ||
      count > 20) {
    return Status::InvalidArgument("DeepSeek top-token copy is invalid");
  }
  return copy_d2h_async(host, device, count * sizeof(std::uint32_t), stream,
                        "cudaMemcpyAsync DeepSeek top-token IDs D2H");
}

Status NvidiaDeepSeekEndpointSequenceOperations::copy_top_logprobs_d2h_async(
    float* host, std::uintptr_t device, std::uint32_t count,
    std::uintptr_t stream) {
  if (host == nullptr || device == 0 || stream == 0 || count == 0 ||
      count > 20) {
    return Status::InvalidArgument("DeepSeek top-logprob copy is invalid");
  }
  return copy_d2h_async(host, device, count * sizeof(float), stream,
                        "cudaMemcpyAsync DeepSeek top-logprobs D2H");
}

}  // namespace pih
