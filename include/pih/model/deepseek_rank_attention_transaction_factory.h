#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "pih/model/deepseek_attention_sequence_transaction.h"
#include "pih/model/deepseek_rank_attention_state_pool.h"

namespace pih {

struct DeepSeekAttentionSequenceBinding final {
  std::uint32_t sequence = 0;
  std::uint32_t state_slot = 0;
};

class DeepSeekRankAttentionTransactionFactory final {
 public:
  explicit DeepSeekRankAttentionTransactionFactory(
      DeepSeekRankAttentionStatePool& state_pool) noexcept
      : state_pool_(&state_pool) {}

  Result<std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>>
  Create(std::span<const DeepSeekAttentionSequenceBinding> bindings);

 private:
  DeepSeekRankAttentionStatePool* state_pool_ = nullptr;
};

}  // namespace pih
