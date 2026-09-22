#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

class DeepSeekOnlineTopK final {
 public:
  static constexpr std::uint32_t kCapacity = 512;

  Status consume(std::uint32_t first_ordinal, std::span<const float> scores);
  Result<std::vector<std::uint32_t>> finish() const;
  std::uint32_t consumed() const { return consumed_; }

 private:
  std::array<float, kCapacity> scores_{};
  std::array<std::uint32_t, kCapacity> ordinals_{};
  std::uint32_t size_ = 0;
  std::uint32_t consumed_ = 0;
  float best_rejected_score_ = 0.0F;
  bool has_rejected_ = false;
};

}  // namespace pih
