#include "pih/model/deepseek_dense_mhc_work_factory.h"

#include <array>
#include <limits>
#include <unordered_set>

namespace pih {
namespace {

template <typename Layer, typename Validate, typename Append>
Status append_layers(
    DeepSeekStageRange owned, std::uint32_t sequence_count,
    std::vector<Layer> layers, Validate validate, Append append) {
  const auto expected = owned.last_layer - owned.first_layer + 1;
  if (layers.size() != expected) {
    return Status::InvalidArgument(
        "DeepSeek dense or mHC work does not cover every owned layer");
  }
  std::array<bool, 43> seen{};
  for (auto& layer : layers) {
    if (layer.layer < owned.first_layer || layer.layer > owned.last_layer ||
        seen[layer.layer] || layer.sequences.size() != sequence_count) {
      return Status::InvalidArgument(
          "DeepSeek dense or mHC layer is invalid or duplicated");
    }
    seen[layer.layer] = true;
    const auto status = validate(layer);
    if (!status.ok()) return status;
    const auto appended = append(layer.layer, std::move(layer.sequences));
    if (!appended.ok()) return appended;
  }
  return Status::Ok();
}

Status validate_mhc(
    const DeepSeekMhcLayerPlanInput& layer, DeepSeekMhcBranchKind kind,
    std::uint32_t token_count) {
  std::unordered_set<DeepSeekMhcSequenceExecutor*> executors;
  std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
  for (const auto& sequence : layer.sequences) {
    if (sequence.executor == nullptr || sequence.transaction == nullptr ||
        sequence.submission.kind != kind ||
        sequence.submission.layer_id != layer.layer ||
        sequence.submission.token_count != token_count ||
        !executors.insert(sequence.executor).second ||
        !transactions.insert(sequence.transaction).second) {
      return Status::InvalidArgument(
          "DeepSeek mHC packed sequence is inconsistent or aliased");
    }
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekDenseMhcWorkFactory>
DeepSeekDenseMhcWorkFactory::Create(
    DeepSeekStageRange owned_layers, std::uint32_t maximum_sequences,
    std::uint32_t maximum_tokens) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || maximum_sequences == 0 ||
      maximum_tokens == 0) {
    return Status::InvalidArgument(
        "DeepSeek dense and mHC work factory capacity is invalid");
  }
  DeepSeekDenseMhcWorkFactory result;
  result.owned_layers_ = owned_layers;
  result.maximum_sequences_ = maximum_sequences;
  result.maximum_tokens_ = maximum_tokens;
  return result;
}

Status DeepSeekDenseMhcWorkFactory::append_plan_work(
    std::uint32_t token_count, std::uint32_t sequence_count,
    std::vector<DeepSeekDenseAttentionLayerPlanInput> dense,
    std::vector<DeepSeekMhcLayerPlanInput> mhc_attention,
    std::vector<DeepSeekMhcLayerPlanInput> mhc_feed_forward,
    DeepSeekRankComputeWorkBuilder& builder) const {
  if (token_count == 0 || token_count > maximum_tokens_ ||
      sequence_count == 0 || sequence_count > maximum_sequences_) {
    return Status::InvalidArgument(
        "DeepSeek dense and mHC plan capacity is invalid");
  }
  auto status = append_layers(
      owned_layers_, sequence_count, std::move(dense),
      [token_count](const auto& layer) {
        std::uint64_t total = 0;
        std::unordered_set<DeepSeekAttentionProjectionCoordinator*> inputs;
        std::unordered_set<DeepSeekAttentionOutputProjectionCoordinator*>
            outputs;
        std::unordered_set<DeepSeekAttentionSequenceTransaction*> transactions;
        for (const auto& sequence : layer.sequences) {
          const auto tokens = sequence.input.input_quant.token_count;
          if (sequence.input_coordinator == nullptr ||
              sequence.output_coordinator == nullptr ||
              sequence.transaction == nullptr || tokens == 0 ||
              sequence.sparse_query_bf16 == 0 ||
              sequence.sparse_kv_bf16 == 0 ||
              sequence.sparse_output_bf16 == 0 ||
              !inputs.insert(sequence.input_coordinator).second ||
              !outputs.insert(sequence.output_coordinator).second ||
              !transactions.insert(sequence.transaction).second) {
            return Status::InvalidArgument(
                "DeepSeek dense packed sequence is incomplete or aliased");
          }
          total += tokens;
        }
        return total == token_count
                   ? Status::Ok()
                   : Status::InvalidArgument(
                         "DeepSeek dense packed tokens differ from plan");
      },
      [&builder](std::uint32_t layer, auto sequences) {
        return builder.add_dense_attention(layer, std::move(sequences));
      });
  if (!status.ok()) return status;
  status = append_layers(
      owned_layers_, sequence_count, std::move(mhc_attention),
      [token_count](const auto& layer) {
        return validate_mhc(
            layer, DeepSeekMhcBranchKind::kAttention, token_count);
      },
      [&builder](std::uint32_t layer, auto sequences) {
        return builder.add_mhc_attention(layer, std::move(sequences));
      });
  if (!status.ok()) return status;
  return append_layers(
      owned_layers_, sequence_count, std::move(mhc_feed_forward),
      [token_count](const auto& layer) {
        return validate_mhc(
            layer, DeepSeekMhcBranchKind::kFeedForward, token_count);
      },
      [&builder](std::uint32_t layer, auto sequences) {
        return builder.add_mhc_feed_forward(layer, std::move(sequences));
      });
}

}  // namespace pih
