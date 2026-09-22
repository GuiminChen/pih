#pragma once

#include <cstdint>

#include "pih/core/status.h"
#include "pih/model/deepseek_token_suppression.h"

namespace pih {

struct DeepSeekEmbeddingLaunch final {
  std::uintptr_t token_ids_u32 = 0, weight_bf16 = 0, output_hc_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, vocab_size = 0, hidden_size = 0;
  std::uint32_t hc_multiplicity = 0;
};
struct DeepSeekHcHeadLaunch final {
  std::uintptr_t input_hc_bf16 = 0, fn_f32 = 0, scale_f32 = 0, base_f32 = 0;
  std::uintptr_t output_bf16 = 0, error_flag_u32 = 0, stream = 0;
  std::uint32_t token_count = 0, hidden_size = 0, hc_multiplicity = 0;
  float rms_epsilon = 0.0F, hc_epsilon = 0.0F;
};
struct DeepSeekLmHeadLaunch final {
  std::uintptr_t input_bf16 = 0, weight_bf16 = 0, logits_f32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t output_rows = 0, vocab_size = 0, hidden_size = 0;
};
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
struct DeepSeekDsparkMarkovLaunch final {
  std::uintptr_t token_ids_u32 = 0, embedding_weight_bf16 = 0;
  std::uintptr_t head_weight_bf16 = 0, raw_logits_f32 = 0;
  std::uintptr_t markov_embeddings_bf16 = 0, biased_logits_f32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t row_count = 0, vocab_size = 0, markov_rank = 0;
};
struct DeepSeekDsparkConfidenceLaunch final {
  std::uintptr_t hidden_bf16 = 0, markov_embeddings_bf16 = 0;
  std::uintptr_t projection_weight_bf16 = 0, confidence_f32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t row_count = 0, hidden_size = 0, markov_rank = 0;
};
#endif
struct DeepSeekArgmaxLaunch final {
  std::uintptr_t logits_f32 = 0, token_id_u32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t vocab_size = 0;
  // Optional for DSpark drafts; required by the main greedy head.
  std::uintptr_t selected_logprob_f32 = 0;
  std::uintptr_t top_logprobs_ids_u32 = 0, top_logprobs_f32 = 0;
  std::uint32_t top_logprobs_count = 0;
  DeepSeekSuppressedTokenSet suppressed_tokens;
};
struct DeepSeekStochasticSampleLaunch final {
  std::uintptr_t logits_f32 = 0, token_id_u32 = 0;
  std::uintptr_t selected_logprob_f32 = 0, rng_word_u32 = 0;
  std::uintptr_t workspace_values_f32 = 0, workspace_ids_u32 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t vocab_size = 0, workspace_capacity = 0;
  float temperature = 1.0F, top_p = 1.0F;
  std::uint32_t top_k = 0;
  std::uint64_t seed = 0, sample_ordinal = 0;
  std::uintptr_t top_logprobs_ids_u32 = 0, top_logprobs_f32 = 0;
  std::uint32_t top_logprobs_count = 0;
  DeepSeekSuppressedTokenSet suppressed_tokens;
};
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
struct DeepSeekDsparkDraftInitLaunch final {
  std::uintptr_t input_token_ids_u32 = 0, embedding_weight_bf16 = 0;
  std::uintptr_t draft_token_ids_u32 = 0, output_hc_bf16 = 0;
  std::uintptr_t error_flag_u32 = 0, stream = 0;
  std::uint32_t sequence_count = 0, noise_token_id = 0;
  std::uint32_t block_size = 0, vocab_size = 0, hidden_size = 0;
  std::uint32_t hc_multiplicity = 0;
};
#endif

Status validate_deepseek_embedding_launch(const DeepSeekEmbeddingLaunch& launch);
Status validate_deepseek_hc_head_launch(const DeepSeekHcHeadLaunch& launch);
Status validate_deepseek_lm_head_launch(const DeepSeekLmHeadLaunch& launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_dspark_markov_launch(
    const DeepSeekDsparkMarkovLaunch& launch);
Status validate_deepseek_dspark_confidence_launch(
    const DeepSeekDsparkConfidenceLaunch& launch);
#endif
Status validate_deepseek_argmax_launch(
    const DeepSeekArgmaxLaunch& launch);
Status validate_deepseek_stochastic_sample_launch(
    const DeepSeekStochasticSampleLaunch& launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status validate_deepseek_dspark_draft_init_launch(
    const DeepSeekDsparkDraftInitLaunch& launch);
#endif
Status launch_deepseek_embedding(DeepSeekEmbeddingLaunch launch);
Status launch_deepseek_hc_head(DeepSeekHcHeadLaunch launch);
Status launch_deepseek_lm_head(DeepSeekLmHeadLaunch launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_dspark_markov(DeepSeekDsparkMarkovLaunch launch);
Status launch_deepseek_dspark_confidence(
    DeepSeekDsparkConfidenceLaunch launch);
#endif
Status launch_deepseek_argmax(DeepSeekArgmaxLaunch launch);
Status launch_deepseek_stochastic_sample(
    DeepSeekStochasticSampleLaunch launch);
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
Status launch_deepseek_dspark_draft_init(
    DeepSeekDsparkDraftInitLaunch launch);
#endif

}  // namespace pih
