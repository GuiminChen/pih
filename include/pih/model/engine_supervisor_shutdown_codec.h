#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/core/result.h"

namespace pih {

inline constexpr std::size_t kEngineSupervisorShutdownFrameBytes = 48;

enum class EngineSupervisorShutdownRequestKind : std::uint8_t {
  kForceStop = 1,
};

enum class EngineSupervisorShutdownAckDisposition : std::uint8_t {
  kAccepted = 1,
  kAlreadyStopping = 2,
  kRejected = 3,
};

struct EngineSupervisorShutdownRequest final {
  std::uint64_t engine_generation = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t request_identity = 0;
  std::uint64_t deadline_ns = 0;
  EngineSupervisorShutdownRequestKind kind =
      EngineSupervisorShutdownRequestKind::kForceStop;
};

struct EngineSupervisorShutdownAck final {
  std::uint64_t engine_generation = 0;
  std::uint64_t engine_epoch = 0;
  std::uint64_t request_identity = 0;
  std::uint64_t acknowledged_ns = 0;
  EngineSupervisorShutdownAckDisposition disposition =
      EngineSupervisorShutdownAckDisposition::kRejected;
};

std::array<std::byte, kEngineSupervisorShutdownFrameBytes>
encode_engine_supervisor_shutdown_request(
    const EngineSupervisorShutdownRequest& value);
Result<EngineSupervisorShutdownRequest>
decode_engine_supervisor_shutdown_request(std::span<const std::byte> bytes);

std::array<std::byte, kEngineSupervisorShutdownFrameBytes>
encode_engine_supervisor_shutdown_ack(
    const EngineSupervisorShutdownAck& value);
Result<EngineSupervisorShutdownAck> decode_engine_supervisor_shutdown_ack(
    std::span<const std::byte> bytes);

class EngineSupervisorShutdownRequestGate final {
 public:
  static Result<EngineSupervisorShutdownRequestGate> Create(
      std::uint64_t engine_generation, std::uint64_t engine_epoch);
  Result<EngineSupervisorShutdownAckDisposition> accept(
      const EngineSupervisorShutdownRequest& request, std::uint64_t now_ns);

  [[nodiscard]] bool accepted() const noexcept { return accepted_; }

 private:
  EngineSupervisorShutdownRequestGate(std::uint64_t engine_generation,
                                      std::uint64_t engine_epoch) noexcept
      : engine_generation_(engine_generation), engine_epoch_(engine_epoch) {}

  std::uint64_t engine_generation_ = 0;
  std::uint64_t engine_epoch_ = 0;
  EngineSupervisorShutdownRequest accepted_request_{};
  bool accepted_ = false;
  bool poisoned_ = false;
};

class EngineSupervisorShutdownAckGate final {
 public:
  static Result<EngineSupervisorShutdownAckGate> Create(
      std::uint64_t engine_generation, std::uint64_t engine_epoch,
      std::uint64_t request_identity);
  Status accept(const EngineSupervisorShutdownAck& ack);

  [[nodiscard]] bool accepted() const noexcept { return accepted_; }

 private:
  EngineSupervisorShutdownAckGate(std::uint64_t engine_generation,
                                  std::uint64_t engine_epoch,
                                  std::uint64_t request_identity) noexcept
      : engine_generation_(engine_generation), engine_epoch_(engine_epoch),
        request_identity_(request_identity) {}

  std::uint64_t engine_generation_ = 0;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t request_identity_ = 0;
  bool accepted_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
