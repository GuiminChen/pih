#include "pih/plugin_sdk/abi.h"
#include "pih/plugin_sdk/kernel_pack.h"
#include "pih/contracts/deepseek_kernels_v1.h"
#include "pih/backend/cuda/deepseek_rope_table.h"
#include "pih/backend/cuda/deepseek_rms_norm.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/backend/cuda/deepseek_attention_dense_primitives.h"
#include "pih/backend/cuda/deepseek_route_gather.h"
#include "pih/backend/cuda/deepseek_fp4_gemm.h"
#include "pih/backend/cuda/deepseek_expert_swiglu.h"
#include "pih/backend/cuda/deepseek_expert_accumulate.h"
#include "pih/backend/cuda/deepseek_endpoint.h"
#include "pih/backend/cuda/deepseek_compressor_pooling.h"
#include "pih/backend/cuda/deepseek_compressor_bf16_projection.h"
#include "pih/backend/cuda/deepseek_compressor_bf16_store.h"
#include "pih/backend/cuda/deepseek_indexer_projection.h"
#include "pih/backend/cuda/deepseek_index_score.h"
#include "pih/backend/cuda/deepseek_sparse_attention.h"
#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"
#include "pih/backend/cuda/deepseek_mhc.h"

#include <algorithm>
#include <cstring>
#include <new>

namespace {

#ifndef PIH_DEEPSEEK_PACK_ID
#define PIH_DEEPSEEK_PACK_ID "pih.kernels.deepseek-v4.sm89"
#endif
#ifndef PIH_DEEPSEEK_PACK_ABI
#define PIH_DEEPSEEK_PACK_ABI "pih.deepseek-sm89-kernel-pack.v1"
#endif
#ifndef PIH_DEEPSEEK_PACK_ARCH
#define PIH_DEEPSEEK_PACK_ARCH "sm89"
#endif

static_assert(pih::DeepSeekSuppressedTokenSet::kMaximumTokenIds ==
              PIH_DEEPSEEK_MAXIMUM_SUPPRESSED_TOKENS_V1);

constexpr pih_kernel_pack_identity_v1 kIdentity{
    sizeof(pih_kernel_pack_identity_v1),
    PIH_KERNEL_PACK_ABI_VERSION_V1,
    PIH_DEEPSEEK_PACK_ID,
    "1.0.0",
    PIH_DEEPSEEK_PACK_ABI,
    PIH_DEEPSEEK_PACK_ARCH,
};

pih_status_v1 Status(const pih::Status& status) {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = PIH_STATUS_INTERNAL_V1;
  switch (status.code()) {
    case pih::StatusCode::kOk: value.code = PIH_STATUS_OK_V1; break;
    case pih::StatusCode::kInvalidArgument:
      value.code = PIH_STATUS_INVALID_ARGUMENT_V1; break;
    case pih::StatusCode::kFailedPrecondition:
      value.code = PIH_STATUS_FAILED_PRECONDITION_V1; break;
    case pih::StatusCode::kResourceExhausted:
      value.code = PIH_STATUS_RESOURCE_EXHAUSTED_V1; break;
    case pih::StatusCode::kUnavailable:
      value.code = PIH_STATUS_UNAVAILABLE_V1; break;
    case pih::StatusCode::kDeadlineExceeded:
      value.code = PIH_STATUS_DEADLINE_EXCEEDED_V1; break;
    case pih::StatusCode::kInternal:
      value.code = PIH_STATUS_INTERNAL_V1; break;
  }
  if (!status.ok()) {
    const auto message = status.message();
    const auto bytes = std::min(message.size(), sizeof(value.message) - 1);
    std::memcpy(value.message, message.data(), bytes);
  }
  return value;
}

pih_status_v1 LaunchRopeTable(
    const pih_deepseek_rope_table_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      launch->yarn > 1) {
    pih_status_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = PIH_STATUS_ABI_VERSION_V1;
    value.code = PIH_STATUS_INVALID_ARGUMENT_V1;
    std::strncpy(value.message, "rope_table_request_invalid",
                 sizeof(value.message) - 1);
    return value;
  }
  return Status(pih::launch_deepseek_rope_table(
      {launch->output_f32, launch->error_flag_u32, launch->stream,
       launch->position_count, launch->rope_dimension, launch->theta,
       launch->scaling_factor, launch->original_maximum_positions,
       launch->beta_fast, launch->beta_slow, launch->yarn != 0}));
}

pih_status_v1 LaunchRmsNorm(
    const pih_deepseek_rms_norm_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1) {
    pih_status_v1 value{};
    value.struct_size = sizeof(value);
    value.abi_version = PIH_STATUS_ABI_VERSION_V1;
    value.code = PIH_STATUS_INVALID_ARGUMENT_V1;
    std::strncpy(value.message, "rms_norm_request_invalid",
                 sizeof(value.message) - 1);
    return value;
  }
  return Status(pih::launch_deepseek_rms_norm(
      {launch->input_bf16, launch->weight_bf16, launch->output_bf16,
       launch->error_flag, launch->stream, launch->rows,
       launch->hidden_size, launch->epsilon}));
}

pih_status_v1 LaunchFp8ActivationQuant(
    const pih_deepseek_fp8_activation_quant_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1) {
    return Status(pih::Status::InvalidArgument("fp8_quant_request_invalid"));
  }
  return Status(pih::launch_deepseek_fp8_activation_quant(
      {launch->input_bf16, launch->output_e4m3, launch->scale_bits,
       launch->error_flag, launch->stream, launch->token_count,
       launch->logical_k}));
}

pih_status_v1 LaunchFp8Gemm(
    const pih_deepseek_fp8_gemm_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      (launch->output_type != 1 && launch->output_type != 2)) {
    return Status(pih::Status::InvalidArgument("fp8_gemm_request_invalid"));
  }
  return Status(pih::launch_deepseek_fp8_gemm(
      {launch->activation_e4m3, launch->activation_scale_bits,
       launch->weight_e4m3, launch->weight_scale_bits, launch->output,
       launch->error_flag, launch->stream, launch->m, launch->n, launch->k,
       static_cast<pih::DeepSeekFp8GemmOutputType>(launch->output_type)}));
}

pih_status_v1 LaunchHeadRms(const pih_deepseek_head_rms_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("head_rms_request_invalid"));
  return Status(pih::launch_deepseek_head_rms(
      {launch->input_bf16, launch->output_bf16, launch->error_flag_u32,
       launch->stream, launch->token_count, launch->head_count,
       launch->head_dimension, launch->epsilon}));
}

pih_status_v1 LaunchRotary(const pih_deepseek_rotary_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      launch->inverse > 1)
    return Status(pih::Status::InvalidArgument("rotary_request_invalid"));
  return Status(pih::launch_deepseek_rotary(
      {launch->input_bf16, launch->frequencies_f32, launch->error_flag_u32,
       launch->stream, launch->token_count, launch->head_count,
       launch->head_dimension, launch->rope_dimension, launch->inverse != 0,
       launch->positions_u32, launch->table_position_count}));
}

pih_status_v1 LaunchKvFp8Simulate(
    const pih_deepseek_kv_fp8_simulate_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("kv_fp8_request_invalid"));
  return Status(pih::launch_deepseek_kv_fp8_simulate(
      {launch->kv_bf16, launch->error_flag_u32, launch->stream,
       launch->token_count, launch->vector_dimension,
       launch->quantized_dimension, launch->group_size}));
}

pih_status_v1 LaunchGroupedFp8Gemm(
    const pih_deepseek_grouped_fp8_gemm_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("grouped_fp8_request_invalid"));
  return Status(pih::launch_deepseek_grouped_fp8_gemm(
      {launch->input_bf16, launch->activation_e4m3,
       launch->activation_scale_bits, launch->weight_e4m3,
       launch->weight_scale_bits, launch->output_bf16,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->group_count, launch->output_per_group,
       launch->input_per_group}));
}

pih_status_v1 LaunchRouteGather(
    const pih_deepseek_route_gather_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("route_gather_request_invalid"));
  return Status(pih::launch_deepseek_route_gather(
      {launch->source_hidden_bf16, launch->token_indices_u32,
       launch->route_hidden_bf16, launch->error_flag, launch->stream,
       launch->route_count, launch->packed_token_count}));
}
pih_status_v1 LaunchFp4Gemm(
    const pih_deepseek_fp4_gemm_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("fp4_gemm_request_invalid"));
  return Status(pih::launch_deepseek_fp4_gemm(
      {launch->activation_e4m3, launch->activation_scale_bits,
       launch->packed_weight, launch->weight_scale_bits, launch->output_bf16,
       launch->error_flag, launch->stream, launch->m, launch->n, launch->k}));
}
pih_status_v1 LaunchExpertSwiGlu(
    const pih_deepseek_expert_swiglu_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("expert_swiglu_request_invalid"));
  return Status(pih::launch_deepseek_expert_swiglu(
      {launch->gate_bf16, launch->up_bf16, launch->route_weights_f32,
       launch->output_bf16, launch->error_flag, launch->stream,
       launch->token_count}));
}
pih_status_v1 LaunchExpertAccumulate(
    const pih_deepseek_expert_accumulate_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("expert_accumulate_request_invalid"));
  return Status(pih::launch_deepseek_ordered_expert_accumulate(
      {launch->expert_output_bf16, launch->token_indices,
       launch->accumulator_f32, launch->error_flag, launch->stream,
       launch->route_count, launch->token_count}));
}

pih_status_v1 LaunchEmbedding(
    const pih_deepseek_embedding_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("embedding_request_invalid"));
  return Status(pih::launch_deepseek_embedding(
      {launch->token_ids_u32, launch->weight_bf16, launch->output_hc_bf16,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->vocab_size, launch->hidden_size, launch->hc_multiplicity}));
}
pih_status_v1 LaunchHcHead(const pih_deepseek_hc_head_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("hc_head_request_invalid"));
  return Status(pih::launch_deepseek_hc_head(
      {launch->input_hc_bf16, launch->fn_f32, launch->scale_f32,
       launch->base_f32, launch->output_bf16, launch->error_flag_u32,
       launch->stream, launch->token_count, launch->hidden_size,
       launch->hc_multiplicity, launch->rms_epsilon, launch->hc_epsilon}));
}
pih_status_v1 LaunchLmHead(const pih_deepseek_lm_head_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("lm_head_request_invalid"));
  return Status(pih::launch_deepseek_lm_head(
      {launch->input_bf16, launch->weight_bf16, launch->logits_f32,
       launch->error_flag_u32, launch->stream, launch->output_rows,
       launch->vocab_size, launch->hidden_size}));
}

pih::DeepSeekSuppressedTokenSet SuppressedTokens(
    const pih_deepseek_suppressed_tokens_v1& source) {
  pih::DeepSeekSuppressedTokenSet value{};
  value.token_count = source.token_count;
  std::copy_n(source.token_ids,
              pih::DeepSeekSuppressedTokenSet::kMaximumTokenIds,
              value.token_ids);
  return value;
}
pih_status_v1 LaunchArgmax(const pih_deepseek_argmax_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      launch->suppressed_tokens.token_count >
          PIH_DEEPSEEK_MAXIMUM_SUPPRESSED_TOKENS_V1)
    return Status(pih::Status::InvalidArgument("argmax_request_invalid"));
  return Status(pih::launch_deepseek_argmax(
      {launch->logits_f32, launch->token_id_u32, launch->error_flag_u32,
       launch->stream, launch->vocab_size, launch->selected_logprob_f32,
       launch->top_logprobs_ids_u32, launch->top_logprobs_f32,
       launch->top_logprobs_count, SuppressedTokens(launch->suppressed_tokens)}));
}
pih_status_v1 LaunchStochasticSample(
    const pih_deepseek_stochastic_sample_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1 ||
      launch->suppressed_tokens.token_count >
          PIH_DEEPSEEK_MAXIMUM_SUPPRESSED_TOKENS_V1)
    return Status(pih::Status::InvalidArgument("sampling_request_invalid"));
  return Status(pih::launch_deepseek_stochastic_sample(
      {launch->logits_f32, launch->token_id_u32,
       launch->selected_logprob_f32, launch->rng_word_u32,
       launch->workspace_values_f32, launch->workspace_ids_u32,
       launch->error_flag_u32, launch->stream, launch->vocab_size,
       launch->workspace_capacity, launch->temperature, launch->top_p,
       launch->top_k, launch->seed, launch->sample_ordinal,
       launch->top_logprobs_ids_u32, launch->top_logprobs_f32,
       launch->top_logprobs_count, SuppressedTokens(launch->suppressed_tokens)}));
}

pih_status_v1 LaunchCompressorPooling(
    const pih_deepseek_compressor_pooling_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("compressor_pooling_invalid"));
  return Status(pih::launch_deepseek_compressor_pooling(
      {launch->kv_projection_f32, launch->gate_projection_f32,
       launch->ape_row_f32, launch->kv_state_f32, launch->score_state_f32,
       launch->output_f32, launch->error_flag_u32, launch->stream,
       launch->batch_count, launch->ratio, launch->head_dim,
       launch->absolute_position}));
}
pih_status_v1 LaunchCompressorProjection(
    const pih_deepseek_compressor_projection_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("compressor_projection_invalid"));
  return Status(pih::launch_deepseek_compressor_bf16_projection(
      {launch->input_bf16, launch->kv_weight_bf16, launch->gate_weight_bf16,
       launch->kv_projection_f32, launch->gate_projection_f32,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->ratio, launch->head_dim, launch->hidden_size}));
}
pih_status_v1 LaunchCompressorStore(
    const pih_deepseek_compressor_store_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("compressor_store_invalid"));
  return Status(pih::launch_deepseek_compressor_bf16_store(
      {launch->compressed_f32, launch->rms_weight_bf16,
       launch->cos_sin_cache_f32, launch->destination_bf16,
       launch->error_flag_u32, launch->stream, launch->head_dim,
       launch->rope_head_dim, launch->rope_position, launch->rms_epsilon}));
}

pih_status_v1 LaunchIndexerProjection(
    const pih_deepseek_indexer_projection_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("indexer_projection_invalid"));
  return Status(pih::launch_deepseek_indexer_projection(
      {launch->qr_bf16, launch->hidden_bf16, launch->wq_b_e4m3,
       launch->wq_b_scale_bits, launch->qr_e4m3, launch->qr_scale_bits,
       launch->weights_proj_bf16, launch->frequencies_f32,
       launch->positions_u32, launch->query_bf16, launch->head_weight_f32,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->table_position_count}));
}
pih_status_v1 LaunchIndexScore(
    const pih_deepseek_index_score_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("index_score_invalid"));
  return Status(pih::launch_deepseek_index_score(
      {launch->query_bf16, launch->index_kv_bf16, launch->head_weight_f32,
       launch->score_f32, launch->error_flag_u32, launch->stream,
       launch->query_count, launch->head_count, launch->slot_count,
       launch->page_slots_u32, launch->slot_base, launch->logical_page_count,
       launch->physical_page_count}));
}
pih_status_v1 LaunchSparseAttention(
    const pih_deepseek_sparse_attention_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("sparse_attention_invalid"));
  return Status(pih::launch_deepseek_sparse_attention(
      {launch->query_bf16, launch->latent_kv_bf16,
       launch->attention_sink_f32, launch->indices_i32, launch->output_bf16,
       launch->error_flag_u32, launch->stream, launch->query_count,
       launch->head_count, launch->kv_count, launch->index_count,
       launch->compressed_kv_bf16, launch->page_slots_u32,
       launch->recent_physical_offset, launch->compressed_physical_offset,
       launch->compressed_slot_count, launch->logical_page_count,
       launch->physical_page_count}));
}

pih_status_v1 LaunchRouterGemm(
    const pih_deepseek_router_gemm_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("router_gemm_invalid"));
  return Status(pih::launch_deepseek_router_bf16_gemm(
      {launch->input_bf16, launch->weight_bf16, launch->scores_f32,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->expert_count, launch->hidden_size}));
}
pih_status_v1 LaunchMhcPre(const pih_deepseek_mhc_pre_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("mhc_pre_invalid"));
  return Status(pih::launch_deepseek_mhc_pre(
      {launch->residual_bf16, launch->fn_f32, launch->scale_f32,
       launch->base_f32, launch->norm_weight_bf16, launch->post_mix_f32,
       launch->residual_mix_f32, launch->layer_input_bf16,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->hidden_size, launch->rms_epsilon, launch->pre_epsilon,
       launch->sinkhorn_epsilon, launch->post_multiplier,
       launch->sinkhorn_iterations}));
}
pih_status_v1 LaunchMhcPost(const pih_deepseek_mhc_post_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("mhc_post_invalid"));
  return Status(pih::launch_deepseek_mhc_post(
      {launch->layer_output_bf16, launch->residual_bf16,
       launch->post_mix_f32, launch->residual_mix_f32, launch->output_bf16,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->hidden_size}));
}
pih_status_v1 LaunchMhcTargetTap(
    const pih_deepseek_mhc_target_tap_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("mhc_target_tap_invalid"));
  return Status(pih::launch_deepseek_mhc_target_hidden_tap(
      {launch->residual_hc_bf16, launch->target_hidden_bf16,
       launch->error_flag_u32, launch->stream, launch->token_count,
       launch->hidden_size, launch->source_stream_count,
       launch->target_stage_count, launch->target_stage_index}));
}

pih_status_v1 LaunchSharedSwiGlu(
    const pih_deepseek_shared_swiglu_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("shared_swiglu_invalid"));
  return Status(pih::launch_deepseek_shared_expert_swiglu(
      {launch->gate_bf16, launch->up_bf16, launch->output_bf16,
       launch->error_flag, launch->stream, launch->token_count}));
}
pih_status_v1 LaunchExpertFinalize(
    const pih_deepseek_expert_finalize_launch_v1* launch) {
  if (launch == nullptr || launch->struct_size != sizeof(*launch) ||
      launch->abi_version != PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1)
    return Status(pih::Status::InvalidArgument("expert_finalize_invalid"));
  return Status(pih::launch_deepseek_expert_finalize(
      {launch->accumulator_f32, launch->shared_output_bf16,
       launch->output_bf16, launch->error_flag, launch->stream,
      launch->token_count}));
}

pih_status_v1 CallbackFailure(uint32_t code, const char* message) noexcept {
  pih_status_v1 value{};
  value.struct_size = sizeof(value);
  value.abi_version = PIH_STATUS_ABI_VERSION_V1;
  value.code = code;
  std::strncpy(value.message, message, sizeof(value.message) - 1);
  return value;
}

#define PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(wrapper, operation, request_type) \
  pih_status_v1 wrapper(const request_type* request) noexcept {               \
    try {                                                                      \
      return operation(request);                                               \
    } catch (const std::bad_alloc&) {                                          \
      return CallbackFailure(PIH_STATUS_RESOURCE_EXHAUSTED_V1,                 \
                             "kernel_pack_allocation_failed");                \
    } catch (...) {                                                            \
      return CallbackFailure(PIH_STATUS_INTERNAL_V1,                           \
                             "kernel_pack_callback_failed");                  \
    }                                                                          \
  }

PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchRopeTable, LaunchRopeTable,
    pih_deepseek_rope_table_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchRmsNorm, LaunchRmsNorm, pih_deepseek_rms_norm_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchFp8ActivationQuant, LaunchFp8ActivationQuant,
    pih_deepseek_fp8_activation_quant_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchFp8Gemm, LaunchFp8Gemm, pih_deepseek_fp8_gemm_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchHeadRms, LaunchHeadRms, pih_deepseek_head_rms_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchRotary, LaunchRotary, pih_deepseek_rotary_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchKvFp8Simulate, LaunchKvFp8Simulate,
    pih_deepseek_kv_fp8_simulate_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchGroupedFp8Gemm, LaunchGroupedFp8Gemm,
    pih_deepseek_grouped_fp8_gemm_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchRouteGather, LaunchRouteGather,
    pih_deepseek_route_gather_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchFp4Gemm, LaunchFp4Gemm, pih_deepseek_fp4_gemm_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchExpertSwiGlu, LaunchExpertSwiGlu,
    pih_deepseek_expert_swiglu_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchExpertAccumulate, LaunchExpertAccumulate,
    pih_deepseek_expert_accumulate_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchEmbedding, LaunchEmbedding,
    pih_deepseek_embedding_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchHcHead, LaunchHcHead, pih_deepseek_hc_head_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchLmHead, LaunchLmHead, pih_deepseek_lm_head_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchArgmax, LaunchArgmax, pih_deepseek_argmax_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchStochasticSample, LaunchStochasticSample,
    pih_deepseek_stochastic_sample_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchCompressorPooling, LaunchCompressorPooling,
    pih_deepseek_compressor_pooling_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchCompressorProjection, LaunchCompressorProjection,
    pih_deepseek_compressor_projection_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchCompressorStore, LaunchCompressorStore,
    pih_deepseek_compressor_store_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchIndexerProjection, LaunchIndexerProjection,
    pih_deepseek_indexer_projection_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchIndexScore, LaunchIndexScore,
    pih_deepseek_index_score_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchSparseAttention, LaunchSparseAttention,
    pih_deepseek_sparse_attention_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchRouterGemm, LaunchRouterGemm,
    pih_deepseek_router_gemm_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchMhcPre, LaunchMhcPre, pih_deepseek_mhc_pre_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchMhcPost, LaunchMhcPost, pih_deepseek_mhc_post_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchMhcTargetTap, LaunchMhcTargetTap,
    pih_deepseek_mhc_target_tap_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchSharedSwiGlu, LaunchSharedSwiGlu,
    pih_deepseek_shared_swiglu_launch_v1)
PIH_DEFINE_CONTAINED_KERNEL_CALLBACK(
    ContainedLaunchExpertFinalize, LaunchExpertFinalize,
    pih_deepseek_expert_finalize_launch_v1)

#undef PIH_DEFINE_CONTAINED_KERNEL_CALLBACK

constexpr pih_deepseek_kernels_api_v1 kContractApi{
    sizeof(pih_deepseek_kernels_api_v1),
    PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1, &kIdentity,
    &ContainedLaunchRopeTable, &ContainedLaunchRmsNorm,
    &ContainedLaunchFp8ActivationQuant, &ContainedLaunchFp8Gemm,
    &ContainedLaunchHeadRms, &ContainedLaunchRotary,
    &ContainedLaunchKvFp8Simulate, &ContainedLaunchGroupedFp8Gemm,
    &ContainedLaunchRouteGather, &ContainedLaunchFp4Gemm,
    &ContainedLaunchExpertSwiGlu, &ContainedLaunchExpertAccumulate,
    &ContainedLaunchEmbedding, &ContainedLaunchHcHead,
    &ContainedLaunchLmHead, &ContainedLaunchArgmax,
    &ContainedLaunchStochasticSample, &ContainedLaunchCompressorPooling,
    &ContainedLaunchCompressorProjection, &ContainedLaunchCompressorStore,
    &ContainedLaunchIndexerProjection, &ContainedLaunchIndexScore,
    &ContainedLaunchSparseAttention, &ContainedLaunchRouterGemm,
    &ContainedLaunchMhcPre, &ContainedLaunchMhcPost,
    &ContainedLaunchMhcTargetTap, &ContainedLaunchSharedSwiGlu,
    &ContainedLaunchExpertFinalize};

// Embedded device images have no external artifacts, but use the same
// mandatory host binding contract. No driver or device call occurs here.
pih_status_v1 BindOrigin(const char* path) noexcept {
  pih_status_v1 result{};
  result.struct_size = sizeof(result); result.abi_version = PIH_STATUS_ABI_VERSION_V1;
  result.code = path && path[0] == '/' ? PIH_STATUS_OK_V1 : PIH_STATUS_INVALID_ARGUMENT_V1;
  return result;
}
constexpr pih_kernel_pack_api_v1 kApi{
    sizeof(pih_kernel_pack_api_v1), PIH_KERNEL_PACK_ABI_VERSION_V1,
    &kIdentity, &kContractApi, BindOrigin};

}  // namespace

extern "C" PIH_PLUGIN_EXPORT
const pih_kernel_pack_identity_v1* pih_kernel_pack_identity_v1_get(void) {
  return &kIdentity;
}

extern "C" PIH_PLUGIN_EXPORT
const pih_kernel_pack_api_v1* pih_kernel_pack_api_v1_get(void) {
  return &kApi;
}
