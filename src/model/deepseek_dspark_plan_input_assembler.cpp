#include "pih/model/deepseek_dspark_plan_input_assembler.h"

namespace pih {
namespace {

DeepSeekDsparkEmbedSubmission assemble_main_projection(
    std::uint32_t token_count,
    const DeepSeekDsparkStage0BoundaryWeightBindings& boundary,
    DeepSeekDsparkDeviceView device, std::uintptr_t stream) {
  DeepSeekDsparkEmbedSubmission embed;
  embed.main_quant = {
      device.target_hidden_bf16, device.main_quant_fp8,
      device.main_quant_scale_ue8m0, device.error_flag_u32, stream,
      token_count, 12288};
  embed.main_proj = {
      device.main_quant_fp8, device.main_quant_scale_ue8m0,
      boundary.main_proj_fp8, boundary.main_proj_scale_ue8m0,
      device.main_projected_bf16, device.error_flag_u32, stream,
      token_count, 4096, 12288, DeepSeekFp8GemmOutputType::kBf16};
  embed.main_norm = {
      device.main_projected_bf16, boundary.main_norm_weight_bf16,
      device.main_normalized_bf16, device.error_flag_u32, stream,
      token_count, 4096, 1.0e-6F};
  return embed;
}

Status validate_main_projection(const DeepSeekDsparkEmbedSubmission& embed) {
  auto status = validate_deepseek_fp8_activation_quant_launch(
      embed.main_quant);
  if (!status.ok()) return status;
  status = validate_deepseek_fp8_gemm_launch(embed.main_proj);
  if (!status.ok()) return status;
  return validate_deepseek_rms_norm_launch(embed.main_norm);
}

}  // namespace

Result<DeepSeekDsparkStageWork> DeepSeekDsparkPlanInputAssembler::Assemble(
    DeepSeekStagePlan stage, std::uintptr_t input_token_ids_u32,
    const DeepSeekEndpointWeightBindings& endpoint_weights,
    const DeepSeekDsparkWeightBindings& weights,
    DeepSeekDsparkDeviceView device,
    DeepSeekDsparkEmbedCoordinator* embed_coordinator,
    DeepSeekDsparkHeadExecutor* head_executor,
    DeepSeekAttentionSequenceTransaction* unbound_transaction,
    std::uintptr_t stream, std::uint32_t maximum_tokens) {
  if (!stage.owns_dspark || !stage.owns_lm_head ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer != 42 || device.target_hidden_bf16 == 0 ||
      input_token_ids_u32 == 0 || endpoint_weights.generation == 0 ||
      endpoint_weights.generation != weights.generation() ||
      endpoint_weights.embedding_weight_bf16 == 0 ||
      endpoint_weights.head_weight_bf16 == 0 ||
      embed_coordinator == nullptr || head_executor == nullptr ||
      stream == 0 || maximum_tokens == 0 || maximum_tokens > 4096 ||
      device.error_flag_u32 == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark plan input identity is invalid");
  }
  DeepSeekDsparkStageWork work;
  const auto& boundary = weights.boundary();
  const auto& head = weights.head();
  work.embed_coordinator = embed_coordinator;
  work.head_executor = head_executor;
  // The rank compiler replaces this optional placeholder with the unique
  // plan-owned attention transaction before publishing work.
  work.transaction = unbound_transaction;
  work.embed = assemble_main_projection(1, boundary, device, stream);
  work.embed.draft_init = {
      input_token_ids_u32, endpoint_weights.embedding_weight_bf16,
      device.draft_token_ids_u32, device.draft_input_hc_bf16,
      device.error_flag_u32, stream, 1, 128799, 5, 129280, 4096, 4};
  work.head.hc = {
      device.draft_input_hc_bf16, head.hc_head_fn_f32,
      head.hc_head_scale_f32, head.hc_head_base_f32,
      device.head_hidden_bf16, device.error_flag_u32, stream,
      5, 4096, 4, 1.0e-6F, 1.0e-6F};
  work.head.rms = {
      device.head_hidden_bf16, head.norm_weight_bf16,
      device.normalized_bf16, device.error_flag_u32, stream,
      5, 4096, 1.0e-6F};
  work.head.lm = {
      device.normalized_bf16, endpoint_weights.head_weight_bf16,
      device.raw_logits_f32, device.error_flag_u32, stream,
      5, 129280, 4096};
  constexpr std::uintptr_t kLogitStride = 129280ULL * sizeof(float);
  constexpr std::uintptr_t kEmbedStride = 256ULL * sizeof(std::uint16_t);
  for (std::size_t index = 0; index < work.head.kBlockSize; ++index) {
    work.head.markov[index] = {
        device.draft_token_ids_u32 + index * sizeof(std::uint32_t),
        head.markov_embedding_bf16, head.markov_head_bf16,
        device.raw_logits_f32 + index * kLogitStride,
        device.markov_embeddings_bf16 + index * kEmbedStride,
        device.biased_logits_f32 + index * kLogitStride,
        device.error_flag_u32, stream, 1, 129280, 256};
    work.head.argmax[index] = {
        work.head.markov[index].biased_logits_f32,
        device.draft_token_ids_u32 + (index + 1) * sizeof(std::uint32_t),
        device.error_flag_u32, stream, 129280};
  }
  work.head.confidence = {
      device.head_hidden_bf16, device.markov_embeddings_bf16,
      head.confidence_weight_bf16, device.confidence_f32,
      device.error_flag_u32, stream, 5, 4096, 256};

  auto status = validate_main_projection(work.embed);
  if (!status.ok()) return status;
  status = validate_deepseek_dspark_draft_init_launch(work.embed.draft_init);
  if (!status.ok()) return status;
  status = validate_deepseek_hc_head_launch(work.head.hc);
  if (!status.ok()) return status;
  status = validate_deepseek_rms_norm_launch(work.head.rms);
  if (!status.ok()) return status;
  status = validate_deepseek_lm_head_launch(work.head.lm);
  if (!status.ok()) return status;
  for (std::size_t index = 0; index < work.head.kBlockSize; ++index) {
    status = validate_deepseek_dspark_markov_launch(work.head.markov[index]);
    if (!status.ok()) return status;
    status = validate_deepseek_argmax_launch(work.head.argmax[index]);
    if (!status.ok()) return status;
  }
  status = validate_deepseek_dspark_confidence_launch(work.head.confidence);
  if (!status.ok()) return status;
  return work;
}

Result<DeepSeekDsparkStageWork>
DeepSeekDsparkPlanInputAssembler::AssemblePrefill(
    DeepSeekStagePlan stage, std::uint32_t token_count,
    const DeepSeekDsparkWeightBindings& weights,
    DeepSeekDsparkDeviceView device,
    DeepSeekDsparkEmbedCoordinator* embed_coordinator,
    DeepSeekAttentionSequenceTransaction* unbound_transaction,
    std::uintptr_t stream, std::uint32_t maximum_tokens) {
  const auto& boundary = weights.boundary();
  if (!stage.owns_dspark || !stage.owns_lm_head ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer != 42 || token_count == 0 ||
      token_count > maximum_tokens || maximum_tokens > 4096 ||
      weights.generation() == 0 || device.target_hidden_bf16 == 0 ||
      device.error_flag_u32 == 0 || embed_coordinator == nullptr ||
      stream == 0) {
    return Status::InvalidArgument(
        "DeepSeek DSpark prefill plan input identity is invalid");
  }
  DeepSeekDsparkStageWork work;
  work.kind = DeepSeekDsparkStageWorkKind::kPrefillStateInitialization;
  work.embed_coordinator = embed_coordinator;
  work.transaction = unbound_transaction;
  work.embed = assemble_main_projection(
      token_count, boundary, device, stream);
  auto status = validate_main_projection(work.embed);
  if (!status.ok()) return status;
  return work;
}

}  // namespace pih
