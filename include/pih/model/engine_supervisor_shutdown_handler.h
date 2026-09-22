#pragma once

#include <utility>

#include "pih/model/engine_shutdown_controller.h"
#include "pih/model/engine_generation_domain_termination.h"
#include "pih/model/engine_supervision_coordinator.h"
#include "pih/model/engine_supervisor_shutdown_codec.h"

namespace pih {

struct EngineSupervisorShutdownHandlingResult final {
  EngineSupervisorShutdownAck ack{};
  EngineShutdownAction action = EngineShutdownAction::kNone;
};

class EngineSupervisorShutdownHandler final {
 public:
  static Result<EngineSupervisorShutdownHandler> Create(
      std::uint64_t engine_generation, std::uint64_t engine_epoch);

  Result<EngineSupervisorShutdownHandlingResult> accept(
      const EngineSupervisorShutdownRequest& request, std::uint64_t now_ns,
      EngineSupervisionCoordinator& coordinator,
      EngineShutdownController& shutdown,
      EngineGenerationDomainTermination& termination);

 private:
  EngineSupervisorShutdownHandler(
      std::uint64_t engine_generation, std::uint64_t engine_epoch,
      EngineSupervisorShutdownRequestGate gate) noexcept
      : engine_generation_(engine_generation), engine_epoch_(engine_epoch),
        gate_(std::move(gate)) {}

  std::uint64_t engine_generation_ = 0;
  std::uint64_t engine_epoch_ = 0;
  EngineSupervisorShutdownRequestGate gate_;
};

}  // namespace pih
