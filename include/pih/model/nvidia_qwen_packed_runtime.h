#pragma once

#include <cstdint>
#include <optional>

#include "pih/scheduler/controller_mailbox.h"
#include "pih/scheduler/controller_packed_driver.h"

namespace pih {

enum class NvidiaQwenPackedStep : std::uint8_t {
  kIdle = 1,
  kCommandProcessed = 2,
  kBatchCompleted = 3,
  kOutputBackpressured = 4,
};

struct NvidiaQwenPackedOutputEvent final {
  ControllerOutputEvent event;
  std::optional<ControllerPackedSamplingReceipt> sampling_receipt;
};

}  // namespace pih
