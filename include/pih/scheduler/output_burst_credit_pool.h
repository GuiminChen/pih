#pragma once

#include <cstdint>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class OutputPlanKind : std::uint8_t { kQwen = 1, kDeepSeek = 2 };
enum class OutputBurstState : std::uint8_t {
  kFree = 0, kPrepared, kCommitted, kTransferred
};

struct OutputBurstCreditLimits final {
  std::uint32_t credit_count;
  std::uint32_t maximum_records_qwen;
  std::uint32_t maximum_records_deepseek;
  std::uint32_t maximum_slots_per_credit;
  std::uint64_t maximum_bytes_per_credit;
};

struct OutputBurstLease final {
  std::uint32_t credit_index;
  std::uint64_t credit_generation;
  std::uint64_t plan_sequence;
  std::uint32_t records;
  std::uint32_t slots;
  std::uint64_t bytes;
};

class OutputBurstCreditPool final {
 public:
  static constexpr std::string_view kAbi = "output_commit_burst_credit_v1";
  static Result<OutputBurstCreditPool> Create(OutputBurstCreditLimits limits);
  Result<OutputBurstLease> acquire(std::uint64_t plan_sequence,
                                   OutputPlanKind kind,
                                   std::uint32_t records,
                                   std::uint32_t slots,
                                   std::uint64_t bytes);
  [[nodiscard]] Status validate_commit(const OutputBurstLease& lease) const;
  [[nodiscard]] Status validate_abort(const OutputBurstLease& lease) const;
  [[nodiscard]] Status validate_transfer(const OutputBurstLease& lease) const;
  Status commit(const OutputBurstLease& lease);
  Status abort(const OutputBurstLease& lease);
  Status transfer(const OutputBurstLease& lease);
  Status release(const OutputBurstLease& lease);
  [[nodiscard]] std::uint32_t available() const noexcept;

 private:
  struct Credit final {
    OutputBurstState state = OutputBurstState::kFree;
    std::uint64_t generation = 0;
    std::uint64_t plan_sequence = 0;
    std::uint32_t records = 0;
    std::uint32_t slots = 0;
    std::uint64_t bytes = 0;
  };
  explicit OutputBurstCreditPool(OutputBurstCreditLimits limits)
      : limits_(limits), credits_(limits.credit_count) {}
  Status validate(const OutputBurstLease& lease, OutputBurstState state) const;
  OutputBurstCreditLimits limits_{};
  std::vector<Credit> credits_;
};

}  // namespace pih
