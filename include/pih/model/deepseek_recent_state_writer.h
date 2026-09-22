#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/model/deepseek_attention_sequence_transaction.h"
#include "pih/model/deepseek_fixed_state_layout.h"

namespace pih {

class DeepSeekRecentStateOperations {
 public:
  virtual ~DeepSeekRecentStateOperations() = default;
  virtual Status copy_d2d_async(std::uintptr_t destination,
                                std::uintptr_t source, std::size_t bytes,
                                std::uintptr_t stream) = 0;
};

struct DeepSeekRecentStateSubmission final {
  std::uint32_t layer_id = 0;
  std::uintptr_t latent_bf16 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t batch_count = 0;
  std::uint32_t absolute_position = 0;
};

class DeepSeekRecentStateWriter final {
 public:
  static Result<DeepSeekRecentStateWriter> Create(
      DeepSeekFixedStateLayout layout,
      DeepSeekRecentStateOperations& operations);

  Status validate(const DeepSeekRecentStateSubmission& submission,
                  const DeepSeekAttentionSequenceTransaction& transaction)
      const;

  Status launch(const DeepSeekRecentStateSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekFixedStateLayout layout_;
  DeepSeekRecentStateOperations* operations_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
