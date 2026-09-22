#pragma once

#include <stdint.h>

#include "pih/plugin_sdk/kernel_pack.h"
#include "pih/plugin_sdk/status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 1U

typedef struct pih_deepseek_rope_table_launch_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uintptr_t output_f32;
  uintptr_t error_flag_u32;
  uintptr_t stream;
  uint32_t position_count;
  uint32_t rope_dimension;
  double theta;
  double scaling_factor;
  uint32_t original_maximum_positions;
  uint32_t beta_fast;
  uint32_t beta_slow;
  uint32_t yarn;
} pih_deepseek_rope_table_launch_v1;

typedef pih_status_v1 (*pih_deepseek_launch_rope_table_v1)(
    const pih_deepseek_rope_table_launch_v1* launch);

typedef struct pih_deepseek_rms_norm_launch_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uintptr_t input_bf16;
  uintptr_t weight_bf16;
  uintptr_t output_bf16;
  uintptr_t error_flag;
  uintptr_t stream;
  uint32_t rows;
  uint32_t hidden_size;
  float epsilon;
} pih_deepseek_rms_norm_launch_v1;

typedef pih_status_v1 (*pih_deepseek_launch_rms_norm_v1)(
    const pih_deepseek_rms_norm_launch_v1* launch);

typedef struct pih_deepseek_fp8_activation_quant_launch_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uintptr_t input_bf16;
  uintptr_t output_e4m3;
  uintptr_t scale_bits;
  uintptr_t error_flag;
  uintptr_t stream;
  uint32_t token_count;
  uint32_t logical_k;
} pih_deepseek_fp8_activation_quant_launch_v1;

typedef pih_status_v1 (*pih_deepseek_launch_fp8_activation_quant_v1)(
    const pih_deepseek_fp8_activation_quant_launch_v1* launch);

typedef struct pih_deepseek_fp8_gemm_launch_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uintptr_t activation_e4m3;
  uintptr_t activation_scale_bits;
  uintptr_t weight_e4m3;
  uintptr_t weight_scale_bits;
  uintptr_t output;
  uintptr_t error_flag;
  uintptr_t stream;
  uint32_t m;
  uint32_t n;
  uint32_t k;
  uint32_t output_type;
} pih_deepseek_fp8_gemm_launch_v1;

typedef pih_status_v1 (*pih_deepseek_launch_fp8_gemm_v1)(
    const pih_deepseek_fp8_gemm_launch_v1* launch);

typedef struct pih_deepseek_head_rms_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, output_bf16, error_flag_u32, stream;
  uint32_t token_count, head_count, head_dimension;
  float epsilon;
} pih_deepseek_head_rms_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_head_rms_v1)(
    const pih_deepseek_head_rms_launch_v1* launch);

typedef struct pih_deepseek_rotary_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, frequencies_f32, error_flag_u32, stream;
  uint32_t token_count, head_count, head_dimension, rope_dimension;
  uint32_t inverse;
  uintptr_t positions_u32;
  uint32_t table_position_count;
} pih_deepseek_rotary_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_rotary_v1)(
    const pih_deepseek_rotary_launch_v1* launch);

typedef struct pih_deepseek_kv_fp8_simulate_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t kv_bf16, error_flag_u32, stream;
  uint32_t token_count, vector_dimension, quantized_dimension, group_size;
} pih_deepseek_kv_fp8_simulate_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_kv_fp8_simulate_v1)(
    const pih_deepseek_kv_fp8_simulate_launch_v1* launch);

typedef struct pih_deepseek_grouped_fp8_gemm_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, activation_e4m3, activation_scale_bits;
  uintptr_t weight_e4m3, weight_scale_bits, output_bf16;
  uintptr_t error_flag_u32, stream;
  uint32_t token_count, group_count, output_per_group, input_per_group;
} pih_deepseek_grouped_fp8_gemm_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_grouped_fp8_gemm_v1)(
    const pih_deepseek_grouped_fp8_gemm_launch_v1* launch);

typedef struct pih_deepseek_route_gather_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t source_hidden_bf16, token_indices_u32, route_hidden_bf16;
  uintptr_t error_flag, stream;
  uint32_t route_count, packed_token_count;
} pih_deepseek_route_gather_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_route_gather_v1)(
    const pih_deepseek_route_gather_launch_v1* launch);

typedef struct pih_deepseek_fp4_gemm_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t activation_e4m3, activation_scale_bits, packed_weight;
  uintptr_t weight_scale_bits, output_bf16, error_flag, stream;
  uint32_t m, n, k;
} pih_deepseek_fp4_gemm_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_fp4_gemm_v1)(
    const pih_deepseek_fp4_gemm_launch_v1* launch);

typedef struct pih_deepseek_expert_swiglu_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t gate_bf16, up_bf16, route_weights_f32, output_bf16;
  uintptr_t error_flag, stream;
  uint32_t token_count;
} pih_deepseek_expert_swiglu_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_expert_swiglu_v1)(
    const pih_deepseek_expert_swiglu_launch_v1* launch);

typedef struct pih_deepseek_expert_accumulate_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t expert_output_bf16, token_indices, accumulator_f32;
  uintptr_t error_flag, stream;
  uint32_t route_count, token_count;
} pih_deepseek_expert_accumulate_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_expert_accumulate_v1)(
    const pih_deepseek_expert_accumulate_launch_v1* launch);

typedef struct pih_deepseek_embedding_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t token_ids_u32, weight_bf16, output_hc_bf16;
  uintptr_t error_flag_u32, stream;
  uint32_t token_count, vocab_size, hidden_size, hc_multiplicity;
} pih_deepseek_embedding_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_embedding_v1)(
    const pih_deepseek_embedding_launch_v1* launch);

typedef struct pih_deepseek_hc_head_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_hc_bf16, fn_f32, scale_f32, base_f32;
  uintptr_t output_bf16, error_flag_u32, stream;
  uint32_t token_count, hidden_size, hc_multiplicity;
  float rms_epsilon, hc_epsilon;
} pih_deepseek_hc_head_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_hc_head_v1)(
    const pih_deepseek_hc_head_launch_v1* launch);

typedef struct pih_deepseek_lm_head_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, weight_bf16, logits_f32, error_flag_u32, stream;
  uint32_t output_rows, vocab_size, hidden_size;
} pih_deepseek_lm_head_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_lm_head_v1)(
    const pih_deepseek_lm_head_launch_v1* launch);

#define PIH_DEEPSEEK_MAXIMUM_SUPPRESSED_TOKENS_V1 17U
typedef struct pih_deepseek_suppressed_tokens_v1 {
  uint32_t token_ids[PIH_DEEPSEEK_MAXIMUM_SUPPRESSED_TOKENS_V1];
  uint32_t token_count;
} pih_deepseek_suppressed_tokens_v1;

typedef struct pih_deepseek_argmax_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t logits_f32, token_id_u32, error_flag_u32, stream;
  uint32_t vocab_size;
  uintptr_t selected_logprob_f32, top_logprobs_ids_u32, top_logprobs_f32;
  uint32_t top_logprobs_count;
  pih_deepseek_suppressed_tokens_v1 suppressed_tokens;
} pih_deepseek_argmax_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_argmax_v1)(
    const pih_deepseek_argmax_launch_v1* launch);

typedef struct pih_deepseek_stochastic_sample_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t logits_f32, token_id_u32, selected_logprob_f32, rng_word_u32;
  uintptr_t workspace_values_f32, workspace_ids_u32, error_flag_u32, stream;
  uint32_t vocab_size, workspace_capacity;
  float temperature, top_p;
  uint32_t top_k;
  uint64_t seed, sample_ordinal;
  uintptr_t top_logprobs_ids_u32, top_logprobs_f32;
  uint32_t top_logprobs_count;
  pih_deepseek_suppressed_tokens_v1 suppressed_tokens;
} pih_deepseek_stochastic_sample_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_stochastic_sample_v1)(
    const pih_deepseek_stochastic_sample_launch_v1* launch);

typedef struct pih_deepseek_compressor_pooling_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t kv_projection_f32, gate_projection_f32, ape_row_f32;
  uintptr_t kv_state_f32, score_state_f32, output_f32;
  uintptr_t error_flag_u32, stream;
  uint32_t batch_count, ratio, head_dim, absolute_position;
} pih_deepseek_compressor_pooling_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_compressor_pooling_v1)(
    const pih_deepseek_compressor_pooling_launch_v1* launch);

typedef struct pih_deepseek_compressor_projection_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, kv_weight_bf16, gate_weight_bf16, kv_projection_f32;
  uintptr_t gate_projection_f32, error_flag_u32, stream;
  uint32_t token_count, ratio, head_dim, hidden_size;
} pih_deepseek_compressor_projection_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_compressor_projection_v1)(
    const pih_deepseek_compressor_projection_launch_v1* launch);

typedef struct pih_deepseek_compressor_store_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t compressed_f32, rms_weight_bf16, cos_sin_cache_f32;
  uintptr_t destination_bf16, error_flag_u32, stream;
  uint32_t head_dim, rope_head_dim, rope_position;
  float rms_epsilon;
} pih_deepseek_compressor_store_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_compressor_store_v1)(
    const pih_deepseek_compressor_store_launch_v1* launch);

typedef struct pih_deepseek_indexer_projection_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t qr_bf16, hidden_bf16, wq_b_e4m3, wq_b_scale_bits;
  uintptr_t qr_e4m3, qr_scale_bits, weights_proj_bf16;
  uintptr_t frequencies_f32, positions_u32, query_bf16, head_weight_f32;
  uintptr_t error_flag_u32, stream;
  uint32_t token_count, table_position_count;
} pih_deepseek_indexer_projection_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_indexer_projection_v1)(
    const pih_deepseek_indexer_projection_launch_v1* launch);

typedef struct pih_deepseek_index_score_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t query_bf16, index_kv_bf16, head_weight_f32, score_f32;
  uintptr_t error_flag_u32, stream;
  uint32_t query_count, head_count, slot_count;
  uintptr_t page_slots_u32;
  uint32_t slot_base, logical_page_count, physical_page_count;
} pih_deepseek_index_score_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_index_score_v1)(
    const pih_deepseek_index_score_launch_v1* launch);

typedef struct pih_deepseek_sparse_attention_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t query_bf16, latent_kv_bf16, attention_sink_f32, indices_i32;
  uintptr_t output_bf16, error_flag_u32, stream;
  uint32_t query_count, head_count, kv_count, index_count;
  uintptr_t compressed_kv_bf16, page_slots_u32;
  int32_t recent_physical_offset, compressed_physical_offset;
  uint32_t compressed_slot_count, logical_page_count, physical_page_count;
} pih_deepseek_sparse_attention_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_sparse_attention_v1)(
    const pih_deepseek_sparse_attention_launch_v1* launch);

typedef struct pih_deepseek_router_gemm_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t input_bf16, weight_bf16, scores_f32, error_flag_u32, stream;
  uint32_t token_count, expert_count, hidden_size;
} pih_deepseek_router_gemm_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_router_gemm_v1)(
    const pih_deepseek_router_gemm_launch_v1* launch);

typedef struct pih_deepseek_mhc_pre_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t residual_bf16, fn_f32, scale_f32, base_f32, norm_weight_bf16;
  uintptr_t post_mix_f32, residual_mix_f32, layer_input_bf16;
  uintptr_t error_flag_u32, stream;
  uint32_t token_count, hidden_size;
  float rms_epsilon, pre_epsilon, sinkhorn_epsilon, post_multiplier;
  uint32_t sinkhorn_iterations;
} pih_deepseek_mhc_pre_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_mhc_pre_v1)(
    const pih_deepseek_mhc_pre_launch_v1* launch);

typedef struct pih_deepseek_mhc_post_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t layer_output_bf16, residual_bf16, post_mix_f32;
  uintptr_t residual_mix_f32, output_bf16, error_flag_u32, stream;
  uint32_t token_count, hidden_size;
} pih_deepseek_mhc_post_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_mhc_post_v1)(
    const pih_deepseek_mhc_post_launch_v1* launch);

typedef struct pih_deepseek_mhc_target_tap_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t residual_hc_bf16, target_hidden_bf16, error_flag_u32, stream;
  uint32_t token_count, hidden_size, source_stream_count;
  uint32_t target_stage_count, target_stage_index;
} pih_deepseek_mhc_target_tap_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_mhc_target_tap_v1)(
    const pih_deepseek_mhc_target_tap_launch_v1* launch);

typedef struct pih_deepseek_shared_swiglu_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t gate_bf16, up_bf16, output_bf16, error_flag, stream;
  uint32_t token_count;
} pih_deepseek_shared_swiglu_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_shared_swiglu_v1)(
    const pih_deepseek_shared_swiglu_launch_v1* launch);

typedef struct pih_deepseek_expert_finalize_launch_v1 {
  uint32_t struct_size, abi_version;
  uintptr_t accumulator_f32, shared_output_bf16, output_bf16;
  uintptr_t error_flag, stream;
  uint32_t token_count;
} pih_deepseek_expert_finalize_launch_v1;
typedef pih_status_v1 (*pih_deepseek_launch_expert_finalize_v1)(
    const pih_deepseek_expert_finalize_launch_v1* launch);

typedef struct pih_deepseek_kernels_api_v1 {
  uint32_t struct_size;
  uint32_t contract_version;
  const pih_kernel_pack_identity_v1* identity;
  pih_deepseek_launch_rope_table_v1 launch_rope_table;
  pih_deepseek_launch_rms_norm_v1 launch_rms_norm;
  pih_deepseek_launch_fp8_activation_quant_v1 launch_fp8_activation_quant;
  pih_deepseek_launch_fp8_gemm_v1 launch_fp8_gemm;
  pih_deepseek_launch_head_rms_v1 launch_head_rms;
  pih_deepseek_launch_rotary_v1 launch_rotary;
  pih_deepseek_launch_kv_fp8_simulate_v1 launch_kv_fp8_simulate;
  pih_deepseek_launch_grouped_fp8_gemm_v1 launch_grouped_fp8_gemm;
  pih_deepseek_launch_route_gather_v1 launch_route_gather;
  pih_deepseek_launch_fp4_gemm_v1 launch_fp4_gemm;
  pih_deepseek_launch_expert_swiglu_v1 launch_expert_swiglu;
  pih_deepseek_launch_expert_accumulate_v1 launch_expert_accumulate;
  pih_deepseek_launch_embedding_v1 launch_embedding;
  pih_deepseek_launch_hc_head_v1 launch_hc_head;
  pih_deepseek_launch_lm_head_v1 launch_lm_head;
  pih_deepseek_launch_argmax_v1 launch_argmax;
  pih_deepseek_launch_stochastic_sample_v1 launch_stochastic_sample;
  pih_deepseek_launch_compressor_pooling_v1 launch_compressor_pooling;
  pih_deepseek_launch_compressor_projection_v1 launch_compressor_projection;
  pih_deepseek_launch_compressor_store_v1 launch_compressor_store;
  pih_deepseek_launch_indexer_projection_v1 launch_indexer_projection;
  pih_deepseek_launch_index_score_v1 launch_index_score;
  pih_deepseek_launch_sparse_attention_v1 launch_sparse_attention;
  pih_deepseek_launch_router_gemm_v1 launch_router_gemm;
  pih_deepseek_launch_mhc_pre_v1 launch_mhc_pre;
  pih_deepseek_launch_mhc_post_v1 launch_mhc_post;
  pih_deepseek_launch_mhc_target_tap_v1 launch_mhc_target_tap;
  pih_deepseek_launch_shared_swiglu_v1 launch_shared_swiglu;
  pih_deepseek_launch_expert_finalize_v1 launch_expert_finalize;
} pih_deepseek_kernels_api_v1;

#ifdef __cplusplus
}
#endif

