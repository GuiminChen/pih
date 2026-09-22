#pragma once

#include <cstdint>
#include <vector>

#include "pih/core/sha256.h"
#include "pih/model/deepseek_expert_pager.h"

namespace pih {

struct DeepSeekPagerCostKey final {
  std::uint64_t request_id = 0;
  std::uint64_t request_generation = 0;
  std::uint64_t plan_sequence = 0;
  std::uint16_t layer = 0;
  std::uint16_t expert = 0;
  friend bool operator==(const DeepSeekPagerCostKey&,
                         const DeepSeekPagerCostKey&) = default;
};

class DeepSeekPagerCostLedger final {
 public:
  static constexpr std::uint32_t kMaximumEntries = 4096;
  static Result<DeepSeekPagerCostLedger> Create(std::uint32_t maximum_entries);

  Result<std::uint64_t> debit(DeepSeekPagerCostKey key,
                              DeepSeekExpertDemandDisposition disposition,
                              std::uint64_t observed_cost_ns);
  [[nodiscard]] std::uint32_t entry_count() const noexcept {
    return static_cast<std::uint32_t>(entries_.size());
  }
  [[nodiscard]] std::uint64_t total_debit_ns() const noexcept {
    return total_debit_ns_;
  }
  [[nodiscard]] const Sha256Digest& identity() const noexcept {
    return identity_;
  }

 private:
  struct Entry final {
    DeepSeekPagerCostKey key;
    std::uint64_t observed_cost_ns = 0;
  };
  Status recompute_identity();

  std::uint32_t maximum_entries_ = 0;
  std::uint64_t total_debit_ns_ = 0;
  std::vector<Entry> entries_;
  Sha256Digest identity_{};
};

}  // namespace pih
