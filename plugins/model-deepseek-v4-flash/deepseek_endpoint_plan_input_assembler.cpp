#include "pih/model/deepseek_endpoint_plan_input_assembler.h"

#include <limits>

namespace pih {

Result<DeepSeekEndpointStageSequenceWork>
DeepSeekEndpointPlanInputAssembler::Assemble(
    DeepSeekStagePlan stage, std::uint32_t token_count,
    std::uintptr_t token_ids_u32, std::uintptr_t final_hc_bf16,
    const DeepSeekEndpointWeightBindings& weights,
    DeepSeekEndpointDeviceView device,
    DeepSeekEndpointSequenceExecutor* executor,
    std::uintptr_t stream, std::uint32_t maximum_tokens,
    std::optional<DeepSeekPreparedSamplingInput> sampling) {
  if ((!stage.owns_embedding && !stage.owns_lm_head) || executor == nullptr ||
      stream == 0 || token_count == 0 || token_count > maximum_tokens ||
      maximum_tokens == 0 || maximum_tokens > 4096 ||
      weights.generation == 0 || device.error_flag_u32 == 0) {
    return Status::InvalidArgument(
        "DeepSeek endpoint plan input identity is invalid");
  }
  if (stage.owns_lm_head && device.selected_logprob_f32 == 0) {
    return Status::InvalidArgument(
        "DeepSeek main head logprob storage is missing");
  }
  DeepSeekEndpointStageSequenceWork work;
  work.executor = executor;
  if (stage.owns_embedding) {
    work.embedding = {
        token_ids_u32, weights.embedding_weight_bf16,
        device.embedding_output_hc_bf16, device.error_flag_u32, stream,
        token_count, 129280, 4096, 4};
    auto status = validate_deepseek_embedding_launch(work.embedding);
    if (!status.ok()) return status;
  }
  if (stage.owns_lm_head) {
    // The head emits the next token for the final position of this chunk.
    constexpr std::uintptr_t kTokenBytes = 4U * 4096U * sizeof(std::uint16_t);
    const auto offset = static_cast<std::uintptr_t>(token_count - 1) * kTokenBytes;
    if (final_hc_bf16 == 0 ||
        final_hc_bf16 > std::numeric_limits<std::uintptr_t>::max() - offset) {
      return Status::InvalidArgument("DeepSeek final head input range is invalid");
    }
    work.head = {
        {final_hc_bf16 + offset, weights.hc_head_fn_f32,
         weights.hc_head_scale_f32, weights.hc_head_base_f32,
         device.head_output_bf16, device.error_flag_u32, stream,
         1, 4096, 4, 1.0e-6F, 1.0e-6F},
        {device.head_output_bf16, weights.norm_weight_bf16,
         device.normalized_bf16, device.error_flag_u32, stream,
         1, 4096, 1.0e-6F},
        {device.normalized_bf16, weights.head_weight_bf16,
         device.logits_f32, device.error_flag_u32, stream,
         1, 129280, 4096},
        {device.logits_f32, device.sampled_token_u32,
         device.error_flag_u32, stream, 129280,
         device.selected_logprob_f32}};
    auto status = validate_deepseek_hc_head_launch(work.head.hc);
    if (!status.ok()) return status;
    status = validate_deepseek_rms_norm_launch(work.head.rms);
    if (!status.ok()) return status;
    status = validate_deepseek_lm_head_launch(work.head.lm);
    if (!status.ok()) return status;
    status = validate_deepseek_argmax_launch(work.head.sample);
    if (!status.ok()) return status;
    if (sampling.has_value()) {
      if (sampling->config_id == 0) {
        return Status::InvalidArgument(
            "DeepSeek sampling config identity is invalid");
      }
      const auto& descriptor = sampling->descriptor;
      if (sampling->mode == DeepSeekSamplingMode::kStochastic) {
        work.head.sample = {};
        work.head.stochastic_sample = DeepSeekStochasticSampleLaunch{
            device.logits_f32, device.sampled_token_u32,
            device.selected_logprob_f32, device.rng_word_u32,
            device.sampling_workspace_values_f32,
            device.sampling_workspace_ids_u32, device.error_flag_u32, stream,
            129280, 129280, descriptor.temperature, descriptor.top_p,
            descriptor.top_k.value_or(0), descriptor.seed,
            descriptor.sample_ordinal, device.top_logprobs_ids_u32,
            device.top_logprobs_f32, descriptor.top_logprobs_count,
            descriptor.suppressed_tokens};
        status = validate_deepseek_stochastic_sample_launch(
            *work.head.stochastic_sample);
      } else {
        if (descriptor.temperature != 0.0F || descriptor.top_p != 1.0F ||
            descriptor.top_k.has_value()) {
          return Status::InvalidArgument(
              "DeepSeek greedy sampling descriptor is invalid");
        }
        work.head.sample.top_logprobs_ids_u32 =
            device.top_logprobs_ids_u32;
        work.head.sample.top_logprobs_f32 = device.top_logprobs_f32;
        work.head.sample.top_logprobs_count = descriptor.top_logprobs_count;
        work.head.sample.suppressed_tokens = descriptor.suppressed_tokens;
        status = validate_deepseek_argmax_launch(work.head.sample);
      }
      if (!status.ok()) return status;
    }
  }
  return work;
}

}  // namespace pih
