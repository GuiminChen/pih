#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "pih/model/engine_supervisor_shutdown_channel.h"

namespace pih {

class LinuxEngineSupervisorShutdownChannelDriver final
    : public EngineSupervisorShutdownChannelDriver {
 public:
  static Result<LinuxEngineSupervisorShutdownChannelDriver> Create(
      std::int32_t inherited_fd, std::uint64_t expected_supervisor_pid,
      std::uint32_t expected_supervisor_uid);

  LinuxEngineSupervisorShutdownChannelDriver(
      const LinuxEngineSupervisorShutdownChannelDriver&) = delete;
  LinuxEngineSupervisorShutdownChannelDriver& operator=(
      const LinuxEngineSupervisorShutdownChannelDriver&) = delete;
  LinuxEngineSupervisorShutdownChannelDriver(
      LinuxEngineSupervisorShutdownChannelDriver&& other) noexcept;
  LinuxEngineSupervisorShutdownChannelDriver& operator=(
      LinuxEngineSupervisorShutdownChannelDriver&& other) noexcept;
  ~LinuxEngineSupervisorShutdownChannelDriver() override;

  Status send_request(std::span<const std::byte> frame) override;
  Result<std::optional<std::vector<std::byte>>> poll_ack() override;

 private:
  explicit LinuxEngineSupervisorShutdownChannelDriver(std::int32_t fd) noexcept
      : fd_(fd) {}
  void close_owned() noexcept;

  std::int32_t fd_ = -1;
};

class LinuxEngineSupervisorShutdownClient final {
 public:
  static Result<std::unique_ptr<LinuxEngineSupervisorShutdownClient>> Create(
      std::int32_t inherited_fd, std::uint64_t expected_supervisor_pid,
      std::uint32_t expected_supervisor_uid, std::uint64_t engine_generation,
      std::uint64_t engine_epoch, std::uint64_t deadline_ns);

  LinuxEngineSupervisorShutdownClient(
      const LinuxEngineSupervisorShutdownClient&) = delete;
  LinuxEngineSupervisorShutdownClient& operator=(
      const LinuxEngineSupervisorShutdownClient&) = delete;

  Result<bool> advance(std::uint64_t now_ns) { return channel_->advance(now_ns); }
  [[nodiscard]] bool request_acknowledged() const noexcept {
    return channel_->request_acknowledged();
  }
  [[nodiscard]] std::uint64_t engine_generation() const noexcept {
    return engine_generation_;
  }
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }

 private:
  LinuxEngineSupervisorShutdownClient(
      std::uint64_t engine_generation, std::uint64_t engine_epoch,
      std::unique_ptr<LinuxEngineSupervisorShutdownChannelDriver> driver,
      std::unique_ptr<EngineSupervisorShutdownChannel> channel) noexcept
      : engine_generation_(engine_generation),
        engine_epoch_(engine_epoch),
        driver_(std::move(driver)),
        channel_(std::move(channel)) {}

  std::uint64_t engine_generation_ = 0;
  std::uint64_t engine_epoch_ = 0;
  std::unique_ptr<LinuxEngineSupervisorShutdownChannelDriver> driver_;
  std::unique_ptr<EngineSupervisorShutdownChannel> channel_;
};

}  // namespace pih
