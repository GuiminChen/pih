#pragma once

#include <array>
#include <optional>
#include <span>
#include <vector>

#include "pih/model/engine_supervisor_shutdown_handler.h"

namespace pih {

class EngineSupervisorShutdownServerDriver {
 public:
  virtual ~EngineSupervisorShutdownServerDriver() = default;
  virtual Result<std::optional<std::vector<std::byte>>> poll_request() = 0;
  virtual Status send_ack(std::span<const std::byte> frame) = 0;
};

class EngineSupervisorShutdownServer final {
 public:
  static Result<EngineSupervisorShutdownServer> Create(
      EngineSupervisorShutdownServerDriver& driver);

  Result<bool> advance(
      std::uint64_t now_ns, EngineSupervisorShutdownHandler& handler,
      EngineSupervisionCoordinator& coordinator,
      EngineShutdownController& shutdown,
      EngineGenerationDomainTermination& termination);

  [[nodiscard]] bool request_acknowledged() const noexcept {
    return complete_;
  }

 private:
  explicit EngineSupervisorShutdownServer(
      EngineSupervisorShutdownServerDriver& driver) noexcept
      : driver_(&driver) {}

  Result<bool> flush_ack();

  EngineSupervisorShutdownServerDriver* driver_ = nullptr;
  std::optional<EngineSupervisorShutdownRequest> pending_request_;
  std::optional<
      std::array<std::byte, kEngineSupervisorShutdownFrameBytes>>
      pending_ack_;
  bool complete_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
