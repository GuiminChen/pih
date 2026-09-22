#include "pih/backend/cuda/deepseek_endpoint.h"

#include <cmath>

#include <cmath>
#include <initializer_list>

namespace pih { namespace {
bool present(std::initializer_list<std::uintptr_t> values) {
  for (const auto value : values) if (value == 0) return false;
  return true;
}

bool valid_suppression(const DeepSeekSuppressedTokenSet& suppressed,
                       std::uint32_t vocabulary_size) {
  if (suppressed.token_count > DeepSeekSuppressedTokenSet::kMaximumTokenIds ||
      suppressed.token_count >= vocabulary_size) return false;
  for (std::uint32_t index = 0;
       index < DeepSeekSuppressedTokenSet::kMaximumTokenIds; ++index) {
    if (index >= suppressed.token_count) {
      if (suppressed.token_ids[index] != 0U) return false;
      continue;
    }
    if (suppressed.token_ids[index] >= vocabulary_size) return false;
    for (std::uint32_t prior = 0; prior < index; ++prior)
      if (suppressed.token_ids[prior] == suppressed.token_ids[index])
        return false;
  }
  return true;
}
}  // namespace pih::<anonymous>

Status validate_deepseek_embedding_launch(const DeepSeekEmbeddingLaunch& v) {
  if (!present({v.token_ids_u32, v.weight_bf16, v.output_hc_bf16,
                v.error_flag_u32, v.stream}) ||
      v.token_count == 0 || v.token_count > 4096 ||
      v.vocab_size != 129280 || v.hidden_size != 4096 ||
      v.hc_multiplicity != 4 || v.output_hc_bf16 == v.weight_bf16 ||
      v.output_hc_bf16 == v.token_ids_u32) {
    return Status::InvalidArgument("DeepSeek embedding launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_hc_head_launch(const DeepSeekHcHeadLaunch& v) {
  if (!present({v.input_hc_bf16, v.fn_f32, v.scale_f32, v.base_f32,
                v.output_bf16, v.error_flag_u32, v.stream}) ||
      v.token_count == 0 || v.token_count > 4096 || v.hidden_size != 4096 ||
      v.hc_multiplicity != 4 || v.input_hc_bf16 == v.output_bf16 ||
      !std::isfinite(v.rms_epsilon) || !std::isfinite(v.hc_epsilon) ||
      v.rms_epsilon != 1.0e-6F || v.hc_epsilon != 1.0e-6F) {
    return Status::InvalidArgument("DeepSeek HC head launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_lm_head_launch(const DeepSeekLmHeadLaunch& v) {
  if (!present({v.input_bf16, v.weight_bf16, v.logits_f32,
                v.error_flag_u32, v.stream}) ||
      v.output_rows == 0 || v.output_rows > 4096 ||
      v.vocab_size != 129280 || v.hidden_size != 4096 ||
      v.logits_f32 == v.input_bf16 || v.logits_f32 == v.weight_bf16) {
    return Status::InvalidArgument("DeepSeek LM head launch is invalid");
  }
  return Status::Ok();
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_dspark_markov_launch(
    const DeepSeekDsparkMarkovLaunch& v) {
  if (!present({v.token_ids_u32, v.embedding_weight_bf16,
                v.head_weight_bf16, v.raw_logits_f32,
                v.markov_embeddings_bf16, v.biased_logits_f32,
                v.error_flag_u32, v.stream}) ||
      v.row_count == 0 || v.row_count > 5 || v.vocab_size != 129280 ||
      v.markov_rank != 256 ||
      v.raw_logits_f32 == v.biased_logits_f32 ||
      v.markov_embeddings_bf16 == v.embedding_weight_bf16) {
    return Status::InvalidArgument("DeepSeek DSpark Markov launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_dspark_confidence_launch(
    const DeepSeekDsparkConfidenceLaunch& v) {
  if (!present({v.hidden_bf16, v.markov_embeddings_bf16,
                v.projection_weight_bf16, v.confidence_f32,
                v.error_flag_u32, v.stream}) ||
      v.row_count == 0 || v.row_count > 5 || v.hidden_size != 4096 ||
      v.markov_rank != 256) {
    return Status::InvalidArgument(
        "DeepSeek DSpark confidence launch is invalid");
  }
  return Status::Ok();
}
#endif

Status validate_deepseek_argmax_launch(
    const DeepSeekArgmaxLaunch& v) {
  if (!present({v.logits_f32, v.token_id_u32, v.error_flag_u32, v.stream}) ||
      v.vocab_size != 129280 || v.logits_f32 == v.token_id_u32 ||
      (v.selected_logprob_f32 != 0 &&
       (v.selected_logprob_f32 == v.logits_f32 ||
        v.selected_logprob_f32 == v.token_id_u32 ||
        v.selected_logprob_f32 == v.error_flag_u32)) ||
      v.top_logprobs_count > 20U ||
      !valid_suppression(v.suppressed_tokens, v.vocab_size) ||
      v.top_logprobs_count >
          v.vocab_size - v.suppressed_tokens.token_count ||
      (v.top_logprobs_count != 0U &&
       (!present({v.selected_logprob_f32, v.top_logprobs_ids_u32,
                  v.top_logprobs_f32}) ||
        v.top_logprobs_ids_u32 == v.token_id_u32 ||
        v.top_logprobs_f32 == v.selected_logprob_f32))) {
    return Status::InvalidArgument("DeepSeek argmax launch is invalid");
  }
  return Status::Ok();
}

Status validate_deepseek_stochastic_sample_launch(
    const DeepSeekStochasticSampleLaunch& v) {
  if (!present({v.logits_f32, v.token_id_u32, v.selected_logprob_f32,
                v.rng_word_u32, v.workspace_values_f32,
                v.workspace_ids_u32, v.error_flag_u32, v.stream}) ||
      v.vocab_size != 129280 || v.workspace_capacity < v.vocab_size ||
      !std::isfinite(v.temperature) || v.temperature <= 0.0F ||
      v.temperature > 2.0F || !std::isfinite(v.top_p) || v.top_p <= 0.0F ||
      v.top_p > 1.0F || v.top_k > v.vocab_size ||
      v.top_logprobs_count > 20U ||
      !valid_suppression(v.suppressed_tokens, v.vocab_size) ||
      v.top_logprobs_count >
          v.vocab_size - v.suppressed_tokens.token_count ||
      (v.top_logprobs_count != 0U &&
       !present({v.top_logprobs_ids_u32, v.top_logprobs_f32})) ||
      v.logits_f32 == v.workspace_values_f32 ||
      v.token_id_u32 == v.workspace_ids_u32 ||
      (v.top_logprobs_ids_u32 != 0U &&
       v.top_logprobs_ids_u32 == v.workspace_ids_u32) ||
      (v.top_logprobs_f32 != 0U &&
       v.top_logprobs_f32 == v.workspace_values_f32)) {
    return Status::InvalidArgument(
        "DeepSeek stochastic sample launch is invalid");
  }
  return Status::Ok();
}

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_dspark_draft_init_launch(
    const DeepSeekDsparkDraftInitLaunch& v) {
  if (!present({v.input_token_ids_u32, v.embedding_weight_bf16,
                v.draft_token_ids_u32, v.output_hc_bf16,
                v.error_flag_u32, v.stream}) ||
      v.sequence_count == 0 || v.sequence_count > 4096 ||
      v.noise_token_id >= v.vocab_size || v.block_size != 5 ||
      v.vocab_size != 129280 || v.hidden_size != 4096 ||
      v.hc_multiplicity != 4 ||
      v.output_hc_bf16 == v.embedding_weight_bf16 ||
      v.draft_token_ids_u32 == v.input_token_ids_u32) {
    return Status::InvalidArgument(
        "DeepSeek DSpark draft init launch is invalid");
  }
  return Status::Ok();
}
#endif

}  // namespace pih
