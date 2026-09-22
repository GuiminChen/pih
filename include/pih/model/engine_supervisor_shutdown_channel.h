#pragma once

#include <optional>
#include <span>
#include <vector>

#include "pih/model/engine_supervisor_shutdown_codec.h"

namespace pih {

class EngineSupervisorShutdownChannelDriver {
 public:
  virtual ~EngineSupervisorShutdownChannelDriver() = default;
  virtual Status send_request(std::span<const std::byte> frame) = 0;
  virtual Result<std::optional<std::vector<std::byte>>> poll_ack() = 0;
};

class EngineSupervisorShutdownChannel final {
 public:
  static Result<EngineSupervisorShutdownChannel> Create(
      EngineSupervisorShutdownRequest request,
      EngineSupervisorShutdownChannelDriver& driver);

  Result<bool> advance(std::uint64_t now_ns);

  [[nodiscard]] bool request_acknowledged() const noexcept {
    return request_acknowledged_;
  }

 private:
  EngineSupervisorShutdownChannel(
      EngineSupervisorShutdownRequest request,
      EngineSupervisorShutdownAckGate ack_gate,
      EngineSupervisorShutdownChannelDriver& driver) noexcept
      : request_(request), ack_gate_(std::move(ack_gate)), driver_(&driver) {}

  EngineSupervisorShutdownRequest request_{};
  EngineSupervisorShutdownAckGate ack_gate_;
  EngineSupervisorShutdownChannelDriver* driver_ = nullptr;
  std::uint64_t last_now_ns_ = 0;
  bool request_sent_ = false;
  bool request_acknowledged_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
