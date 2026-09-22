#pragma once

#include <cstdint>
#include <memory>
#include <utility>

#include "pih/model/engine_supervisor_shutdown_server.h"

namespace pih {

class LinuxEngineSupervisorShutdownServerDriver final
    : public EngineSupervisorShutdownServerDriver {
 public:
  static Result<LinuxEngineSupervisorShutdownServerDriver> Create(
      std::int32_t inherited_fd, std::uint64_t expected_engine_pid,
      std::uint32_t expected_engine_uid);

  LinuxEngineSupervisorShutdownServerDriver(
      const LinuxEngineSupervisorShutdownServerDriver&) = delete;
  LinuxEngineSupervisorShutdownServerDriver& operator=(
      const LinuxEngineSupervisorShutdownServerDriver&) = delete;
  LinuxEngineSupervisorShutdownServerDriver(
      LinuxEngineSupervisorShutdownServerDriver&& other) noexcept;
  LinuxEngineSupervisorShutdownServerDriver& operator=(
      LinuxEngineSupervisorShutdownServerDriver&& other) noexcept;
  ~LinuxEngineSupervisorShutdownServerDriver() override;

  Result<std::optional<std::vector<std::byte>>> poll_request() override;
  Status send_ack(std::span<const std::byte> frame) override;

 private:
  explicit LinuxEngineSupervisorShutdownServerDriver(
      std::int32_t fd, std::uint64_t expected_engine_pid,
      std::uint32_t expected_engine_uid) noexcept
      : fd_(fd), expected_engine_pid_(expected_engine_pid),
        expected_engine_uid_(expected_engine_uid) {}
  void close_owned() noexcept;

  std::int32_t fd_ = -1;
  std::uint64_t expected_engine_pid_ = 0;
  std::uint32_t expected_engine_uid_ = 0;
};

class LinuxEngineSupervisorShutdownService final {
 public:
  static Result<std::unique_ptr<LinuxEngineSupervisorShutdownService>> Create(
      std::int32_t inherited_fd, std::uint64_t expected_engine_pid,
      std::uint32_t expected_engine_uid, std::uint64_t engine_generation,
      std::uint64_t engine_epoch);

  LinuxEngineSupervisorShutdownService(
      const LinuxEngineSupervisorShutdownService&) = delete;
  LinuxEngineSupervisorShutdownService& operator=(
      const LinuxEngineSupervisorShutdownService&) = delete;

  Result<bool> advance(
      std::uint64_t now_ns, EngineSupervisionCoordinator& coordinator,
      EngineShutdownController& shutdown,
      EngineGenerationDomainTermination& termination) {
    return server_->advance(now_ns, *handler_, coordinator, shutdown,
                            termination);
  }
  [[nodiscard]] bool request_acknowledged() const noexcept {
    return server_->request_acknowledged();
  }
  [[nodiscard]] std::uint64_t engine_generation() const noexcept {
    return engine_generation_;
  }
  [[nodiscard]] std::uint64_t engine_epoch() const noexcept {
    return engine_epoch_;
  }

 private:
  LinuxEngineSupervisorShutdownService(
      std::uint64_t engine_generation, std::uint64_t engine_epoch,
      std::unique_ptr<LinuxEngineSupervisorShutdownServerDriver> driver,
      std::unique_ptr<EngineSupervisorShutdownServer> server,
      std::unique_ptr<EngineSupervisorShutdownHandler> handler) noexcept
      : engine_generation_(engine_generation), engine_epoch_(engine_epoch),
        driver_(std::move(driver)), server_(std::move(server)),
        handler_(std::move(handler)) {}

  std::uint64_t engine_generation_ = 0;
  std::uint64_t engine_epoch_ = 0;
  std::unique_ptr<LinuxEngineSupervisorShutdownServerDriver> driver_;
  std::unique_ptr<EngineSupervisorShutdownServer> server_;
  std::unique_ptr<EngineSupervisorShutdownHandler> handler_;
};

}  // namespace pih
