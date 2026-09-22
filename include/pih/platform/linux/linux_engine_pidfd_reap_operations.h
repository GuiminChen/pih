#pragma once

#include <span>
#include <vector>

#include "pih/model/engine_pidfd_receipt_producer.h"

namespace pih {

struct LinuxEnginePidfdBinding final {
  std::uint64_t pidfd_identity = 0;
  std::int32_t pidfd = -1;
};

class LinuxEnginePidfdReapOperations final
    : public EnginePidfdReapOperations {
 public:
  static Result<LinuxEnginePidfdReapOperations> Create(
      std::span<const LinuxEnginePidfdBinding> bindings);
  Result<std::optional<EnginePidfdKernelReapObservation>> poll_and_reap(
      const EnginePidfdReapTarget& target) override;

 private:
  explicit LinuxEnginePidfdReapOperations(
      std::vector<LinuxEnginePidfdBinding> bindings) noexcept
      : bindings_(std::move(bindings)) {}
  std::vector<LinuxEnginePidfdBinding> bindings_;
};

}  // namespace pih
