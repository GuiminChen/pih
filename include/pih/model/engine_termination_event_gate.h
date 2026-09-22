#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class EngineTerminationEventKind : std::uint8_t { kSigterm, kSigint };
enum class EngineTerminationEventDisposition : std::uint8_t {
  kFirstRequest,
  kRepeatedRequest,
};

struct EngineTerminationEvent final {
  std::uint64_t engine_generation = 0;
  std::uint64_t event_identity = 0;
  EngineTerminationEventKind kind = EngineTerminationEventKind::kSigterm;
};

class EngineTerminationEventGate final {
 public:
  static Result<EngineTerminationEventGate> Create(
      std::uint64_t engine_generation);
  Result<EngineTerminationEventDisposition> accept(
      const EngineTerminationEvent& event);

 private:
  explicit EngineTerminationEventGate(std::uint64_t generation) noexcept
      : generation_(generation) {}
  std::uint64_t generation_ = 0;
  std::uint64_t last_event_identity_ = 0;
  bool accepted_first_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
