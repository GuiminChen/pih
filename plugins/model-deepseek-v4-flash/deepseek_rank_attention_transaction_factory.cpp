#include "pih/model/deepseek_rank_attention_transaction_factory.h"

#include <unordered_set>

namespace pih {

Result<std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>>
DeepSeekRankAttentionTransactionFactory::Create(
    std::span<const DeepSeekAttentionSequenceBinding> bindings) {
  if (state_pool_ == nullptr || bindings.empty() ||
      bindings.size() > state_pool_->sequence_capacity()) {
    return Status::InvalidArgument(
        "DeepSeek attention transaction bindings are invalid");
  }
  std::unordered_set<std::uint32_t> sequences;
  std::unordered_set<std::uint32_t> slots;
  for (const auto& binding : bindings) {
    if (binding.sequence == 0 ||
        binding.state_slot >= state_pool_->sequence_capacity() ||
        !sequences.insert(binding.sequence).second ||
        !slots.insert(binding.state_slot).second) {
      return Status::InvalidArgument(
          "DeepSeek attention sequence or state slot is invalid or duplicated");
    }
  }

  std::uint32_t ratio4_layers = 0;
  std::uint32_t ratio128_layers = 0;
  for (const auto& descriptor : state_pool_->fixed_layout().descriptors()) {
    ratio4_layers += descriptor.kind == DeepSeekFixedLayerKind::kRatio4;
    ratio128_layers += descriptor.kind == DeepSeekFixedLayerKind::kRatio128;
  }
  std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>> result;
  result.reserve(bindings.size());
  for (const auto& binding : bindings) {
    auto transaction = DeepSeekAttentionSequenceTransaction::Create(
        binding.sequence, ratio4_layers, ratio128_layers,
        state_pool_->sequence(binding.state_slot).fixed_banks(),
        state_pool_->ratio4_pool(), state_pool_->ratio128_pool());
    if (!transaction.ok()) return transaction.status();
    result.push_back(std::make_unique<DeepSeekAttentionSequenceTransaction>(
        std::move(*transaction)));
  }
  return result;
}

}  // namespace pih
