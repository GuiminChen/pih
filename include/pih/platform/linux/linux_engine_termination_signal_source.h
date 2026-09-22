#pragma once

#include <cstdint>

#include <optional>

#include "pih/model/engine_termination_event_gate.h"

namespace pih {

class LinuxEngineTerminationSignalSource final {
 public:
  static Result<LinuxEngineTerminationSignalSource> Create(
      std::uint64_t engine_generation);
  ~LinuxEngineTerminationSignalSource();
  LinuxEngineTerminationSignalSource(
      const LinuxEngineTerminationSignalSource&) = delete;
  LinuxEngineTerminationSignalSource& operator=(
      const LinuxEngineTerminationSignalSource&) = delete;
  LinuxEngineTerminationSignalSource(
      LinuxEngineTerminationSignalSource&& other) noexcept;
  LinuxEngineTerminationSignalSource& operator=(
      LinuxEngineTerminationSignalSource&& other) noexcept;
  Result<std::optional<EngineTerminationEvent>> poll();

 private:
  LinuxEngineTerminationSignalSource(std::uint64_t generation,
                                     std::int32_t fd) noexcept
      : generation_(generation), fd_(fd) {}
  void reset() noexcept;
  std::uint64_t generation_ = 0;
  std::uint64_t next_event_identity_ = 1;
  std::int32_t fd_ = -1;
};

}  // namespace pih
