#include "pih/model/deepseek_dense_mhc_plan_input_assembler.h"

namespace pih {

Result<DeepSeekDenseMhcPlanInput>
DeepSeekDenseMhcPlanInputAssembler::Assemble(
    DeepSeekStageRange owned_layers,
    std::span<const DeepSeekDenseMhcLayerSubmissionInput> layers,
    DeepSeekDenseMhcRuntimeResources& runtime,
    DeepSeekAttentionSequenceTransaction& transaction) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 ||
      layers.size() != owned_layers.last_layer - owned_layers.first_layer + 1 ||
      runtime.layer_count() != layers.size()) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC plan input coverage is invalid");
  }
  DeepSeekDenseMhcPlanInput result;
  result.dense_attention.reserve(layers.size());
  result.mhc_attention.reserve(layers.size());
  result.mhc_feed_forward.reserve(layers.size());
  for (std::size_t offset = 0; offset < layers.size(); ++offset) {
    const auto& source = layers[offset];
    const auto expected = owned_layers.first_layer +
                          static_cast<std::uint32_t>(offset);
    if (source.layer != expected || source.sparse_query_bf16 == 0 ||
        source.sparse_kv_bf16 == 0 || source.sparse_output_bf16 == 0 ||
        source.mhc_attention.kind != DeepSeekMhcBranchKind::kAttention ||
        source.mhc_feed_forward.kind !=
            DeepSeekMhcBranchKind::kFeedForward ||
        source.mhc_attention.layer_id != expected ||
        source.mhc_feed_forward.layer_id != expected) {
      return Status::InvalidArgument(
          "DeepSeek dense mHC layer submission is invalid");
    }
    auto status = validate_deepseek_attention_projection_submission(
        source.attention_input);
    if (!status.ok()) return status;
    status = validate_deepseek_attention_output_projection_submission(
        source.attention_output);
    if (!status.ok()) return status;
    status = validate_deepseek_mhc_sequence_submission(
        source.mhc_attention);
    if (!status.ok()) return status;
    status = validate_deepseek_mhc_sequence_submission(
        source.mhc_feed_forward);
    if (!status.ok()) return status;
    auto borrowed = runtime.borrow(expected);
    if (!borrowed.ok()) return borrowed.status();
    DeepSeekDenseAttentionStageSequenceWork dense;
    dense.input_coordinator = borrowed->input;
    dense.output_coordinator = borrowed->output;
    dense.transaction = &transaction;
    dense.input = source.attention_input;
    dense.output = source.attention_output;
    dense.sparse_query_bf16 = source.sparse_query_bf16;
    dense.sparse_kv_bf16 = source.sparse_kv_bf16;
    dense.sparse_output_bf16 = source.sparse_output_bf16;
    result.dense_attention.push_back({expected, {std::move(dense)}});
    result.mhc_attention.push_back(
        {expected, {{borrowed->mhc_attention, &transaction,
                     source.mhc_attention}}});
    result.mhc_feed_forward.push_back(
        {expected, {{borrowed->mhc_feed_forward, &transaction,
                     source.mhc_feed_forward}}});
  }
  return result;
}

}  // namespace pih
