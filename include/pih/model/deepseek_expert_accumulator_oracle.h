#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_expert_subwave_plan.h"

namespace pih {

class DeepSeekExpertAccumulatorOracle final {
 public:
  static constexpr std::uint32_t kHiddenSize = 4096;
  static Result<DeepSeekExpertAccumulatorOracle> Create(
      std::uint32_t token_count);
  Status add_expert(std::uint16_t expert_id,
                    std::span<const DeepSeekExpertRoute> routes,
                    std::span<const float> expert_output);
  Result<std::vector<float>> finalize(
      std::span<const float> shared_expert_output);
  [[nodiscard]] std::span<const float> accumulator() const noexcept {
    return accumulator_;
  }

 private:
  std::uint32_t token_count_ = 0;
  std::uint16_t next_minimum_expert_ = 0;
  std::vector<float> accumulator_;
  bool finalized_ = false;
};

}  // namespace pih
