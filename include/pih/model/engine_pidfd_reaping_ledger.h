#pragma once

#include <cstdint>

#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class EngineSupervisedProcessRole : std::uint8_t { kController, kRank };

struct EngineSupervisedProcessIdentity final {
  EngineSupervisedProcessRole role = EngineSupervisedProcessRole::kController;
  std::int32_t rank = -1;
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  bool operator==(const EngineSupervisedProcessIdentity&) const = default;
};

struct EnginePidfdReapedReceipt final {
  std::uint64_t engine_generation = 0;
  std::uint64_t event_identity = 0;
  EngineSupervisedProcessIdentity process;
  bool exited = false;
  bool reaped = false;
};

class EnginePidfdReapingLedger final {
 public:
  static Result<EnginePidfdReapingLedger> Create(
      std::uint64_t engine_generation,
      std::span<const EngineSupervisedProcessIdentity> processes);
  Status accept(const EnginePidfdReapedReceipt& receipt);
  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] std::size_t remaining() const noexcept;
  [[nodiscard]] bool poisoned() const noexcept { return poisoned_; }

 private:
  EnginePidfdReapingLedger(
      std::uint64_t generation,
      std::vector<EngineSupervisedProcessIdentity> processes) noexcept
      : generation_(generation), processes_(std::move(processes)),
        reaped_(processes_.size(), false) {}
  Status poison() noexcept;
  std::uint64_t generation_ = 0;
  std::vector<EngineSupervisedProcessIdentity> processes_;
  std::vector<bool> reaped_;
  std::uint64_t last_event_identity_ = 0;
  bool poisoned_ = false;
};

}  // namespace pih
