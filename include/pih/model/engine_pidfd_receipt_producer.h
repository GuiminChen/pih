#pragma once

#include <optional>

#include "pih/model/engine_pidfd_reaping_ledger.h"

namespace pih {

enum class EnginePidfdReaperMode : std::uint8_t {
  kDirectParent,
  kDesignatedSubreaper,
};

struct EnginePidfdReaperAuthority final {
  std::uint64_t engine_generation = 0;
  std::uint64_t producer_process_identity = 0;
  EnginePidfdReaperMode mode = EnginePidfdReaperMode::kDirectParent;
  bool kernel_subreaper_enabled = false;
};

struct EnginePidfdReapTarget final {
  EngineSupervisedProcessIdentity process;
  std::uint64_t creator_process_identity = 0;
  std::uint64_t adopted_reaper_identity = 0;
};

struct EnginePidfdKernelReapObservation final {
  std::uint64_t process_identity = 0;
  std::uint64_t pidfd_identity = 0;
  bool exited = false;
  bool reaped = false;
};

class EnginePidfdReapOperations {
 public:
  virtual ~EnginePidfdReapOperations() = default;
  virtual Result<std::optional<EnginePidfdKernelReapObservation>> poll_and_reap(
      const EnginePidfdReapTarget& target) = 0;
};

class EnginePidfdReceiptProducer final {
 public:
  static Result<EnginePidfdReceiptProducer> Create(
      EnginePidfdReaperAuthority authority,
      std::span<const EnginePidfdReapTarget> targets,
      EnginePidfdReapOperations& operations);
  Result<std::optional<EnginePidfdReapedReceipt>> poll(
      const EnginePidfdReapTarget& target, std::uint64_t event_identity);

 private:
  EnginePidfdReceiptProducer(EnginePidfdReaperAuthority authority,
                             std::vector<EnginePidfdReapTarget> targets,
                             EnginePidfdReapOperations& operations) noexcept
      : authority_(authority), targets_(std::move(targets)),
        operations_(&operations), produced_(targets_.size(), false) {}
  Status validate_target(const EnginePidfdReapTarget& target) const;
  EnginePidfdReaperAuthority authority_{};
  std::vector<EnginePidfdReapTarget> targets_;
  EnginePidfdReapOperations* operations_ = nullptr;
  std::vector<bool> produced_;
  bool poisoned_ = false;
};

}  // namespace pih
