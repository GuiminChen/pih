#include "pih/model/engine_termination_event_gate.h"

#include <limits>

namespace pih {

Result<EngineTerminationEventGate> EngineTerminationEventGate::Create(
    std::uint64_t engine_generation) {
  if (engine_generation == 0)
    return Status::InvalidArgument(
        "engine termination event generation is invalid");
  return EngineTerminationEventGate(engine_generation);
}

Result<EngineTerminationEventDisposition>
EngineTerminationEventGate::accept(const EngineTerminationEvent& event) {
  if (poisoned_)
    return Status::FailedPrecondition(
        "engine termination event gate is poisoned");
  const bool valid_kind =
      event.kind == EngineTerminationEventKind::kSigterm ||
      event.kind == EngineTerminationEventKind::kSigint;
  if (event.engine_generation != generation_ || !valid_kind ||
      event.event_identity == 0 ||
      last_event_identity_ == std::numeric_limits<std::uint64_t>::max() ||
      event.event_identity != last_event_identity_ + 1) {
    poisoned_ = true;
    return Status::FailedPrecondition(
        "engine termination event identity drifted");
  }
  last_event_identity_ = event.event_identity;
  if (!accepted_first_) {
    accepted_first_ = true;
    return EngineTerminationEventDisposition::kFirstRequest;
  }
  return EngineTerminationEventDisposition::kRepeatedRequest;
}

}  // namespace pih
