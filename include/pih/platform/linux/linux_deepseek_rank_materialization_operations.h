#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "pih/model/deepseek_rank_materialization_exchange.h"
#include "pih/model/deepseek_rank_materialization_warmup.h"
#include "pih/model/deepseek_rank_control_codec.h"
#include "pih/model/deepseek_rank_serving_plan_executor.h"
#include "pih/model/deepseek_rank_serving_coordinator.h"
#include "pih/platform/linux/linux_deepseek_rank_process_driver.h"

namespace pih {

inline constexpr std::string_view
    kLinuxDeepSeekRankMaterializationControllerOperationsAbi =
        "pih_linux_deepseek_rank_materialization_controller_operations_v1";
inline constexpr std::string_view
    kLinuxDeepSeekRankMaterializationReceiverOperationsAbi =
        "pih_linux_deepseek_rank_materialization_receiver_operations_v1";
inline constexpr std::string_view
    kLinuxDeepSeekRankServingWorkerOperationsAbi =
        "pih_linux_deepseek_rank_serving_worker_operations_v1";
inline constexpr std::string_view
    kLinuxDeepSeekRankServingControllerOperationsAbi =
        "pih_linux_deepseek_rank_serving_controller_operations_v1";

class LinuxDeepSeekRankMaterializationControllerOperations final
    : public DeepSeekRankMaterializationGrantChannel,
      public DeepSeekRankMaterializationWarmupOperations {
 public:
  static Result<LinuxDeepSeekRankMaterializationControllerOperations>
  Create(LinuxDeepSeekRankProcessDriver& driver,
         DeepSeekRankProcessSupervisor& supervisor,
         std::span<const DeepSeekRankProcessHandle> ordered_handles);

  Status send_grant(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override;
  Result<std::optional<std::vector<std::byte>>> poll_ack(
      const DeepSeekRankProcessHandle& handle) override;
  Result<std::optional<std::vector<std::byte>>> poll_completion(
      const DeepSeekRankProcessHandle& handle) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) override;

 private:
  LinuxDeepSeekRankMaterializationControllerOperations(
      LinuxDeepSeekRankProcessDriver& driver,
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessHandle> handles,
      std::uint64_t engine_epoch,
      std::uint64_t worker_generation) noexcept
      : driver_(&driver), supervisor_(&supervisor),
        handles_(std::move(handles)), engine_epoch_(engine_epoch),
        worker_generation_(worker_generation) {}

  Result<std::uint32_t> rank_for(
      const DeepSeekRankProcessHandle& handle) const;
  Result<std::int32_t> control_fd(std::uint32_t rank) const;

  LinuxDeepSeekRankProcessDriver* driver_ = nullptr;
  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessHandle> handles_;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
};

class LinuxDeepSeekRankMaterializationReceiverOperations final
    : public DeepSeekRankMaterializationGrantReceiverOperations,
      public DeepSeekRankMaterializationCompletionSenderOperations {
 public:
  static Result<LinuxDeepSeekRankMaterializationReceiverOperations>
  Create(std::int32_t control_fd,
         std::uint64_t expected_controller_process_identity);

  Result<std::optional<std::vector<std::byte>>> receive_grant(
      std::int32_t control_fd) override;
  Result<std::uint64_t> monotonic_now_ns() override;
  Status send_ack(
      std::int32_t control_fd, std::span<const std::byte> frame) override;
  Status send_completion(
      std::int32_t control_fd, std::span<const std::byte> frame) override;

 private:
  LinuxDeepSeekRankMaterializationReceiverOperations(
      std::int32_t control_fd,
      std::uint64_t controller_process_identity) noexcept
      : control_fd_(control_fd),
        controller_process_identity_(controller_process_identity) {}

  Status validate_bound_control(std::int32_t control_fd) const;

  std::int32_t control_fd_ = -1;
  std::uint64_t controller_process_identity_ = 0;
};

// Reuses the worker's authenticated control socket after materialization. The
// service gate validates session/order semantics; this adapter validates the
// kernel peer and exact frame type before bytes enter or leave that gate.
class LinuxDeepSeekRankServingWorkerOperations final {
 public:
  static Result<LinuxDeepSeekRankServingWorkerOperations> Create(
      std::int32_t control_fd,
      std::uint64_t expected_controller_process_identity);

  Result<std::optional<std::vector<std::byte>>> receive_command();
  Status send_completion(std::span<const std::byte> frame);

 private:
  LinuxDeepSeekRankServingWorkerOperations(
      std::int32_t control_fd,
      std::uint64_t controller_process_identity) noexcept
      : control_fd_(control_fd),
        controller_process_identity_(controller_process_identity) {}

  Status validate_bound_control() const;

  std::int32_t control_fd_ = -1;
  std::uint64_t controller_process_identity_ = 0;
};

class LinuxDeepSeekRankServingWorkerTransport final
    : public DeepSeekRankServingWorkerTransport {
 public:
  static Result<LinuxDeepSeekRankServingWorkerTransport> Create(
      std::int32_t control_fd,
      std::uint64_t expected_controller_process_identity);
  Result<std::optional<DeepSeekRankServingCommand>> receive() override;
  Status send(const DeepSeekRankServingCompletion& completion) override;
 private:
  explicit LinuxDeepSeekRankServingWorkerTransport(
      LinuxDeepSeekRankServingWorkerOperations operations) noexcept
      : operations_(std::move(operations)) {}
  LinuxDeepSeekRankServingWorkerOperations operations_;
};

// Owns the worker-side service graph after startup/materialization has derived
// the warm-sealed session. Its factory, rather than this transport, is the
// only component allowed to resolve opaque lease identities into resources.
class LinuxDeepSeekRankServingWorker final {
 public:
  static Result<LinuxDeepSeekRankServingWorker> Create(
      DeepSeekRankServingSessionBinding session, std::int32_t control_fd,
      std::uint64_t expected_controller_process_identity,
      DeepSeekRankServingPlanExecutionFactory& factory,
      DeepSeekRankServingLeaseResolver& resolver);

  LinuxDeepSeekRankServingWorker(const LinuxDeepSeekRankServingWorker&) =
      delete;
  LinuxDeepSeekRankServingWorker& operator=(
      const LinuxDeepSeekRankServingWorker&) = delete;
  LinuxDeepSeekRankServingWorker(LinuxDeepSeekRankServingWorker&&) noexcept =
      default;
  LinuxDeepSeekRankServingWorker& operator=(
      LinuxDeepSeekRankServingWorker&&) noexcept = default;

  Status advance();
  [[nodiscard]] bool failed() const noexcept { return loop_.failed(); }

 private:
  LinuxDeepSeekRankServingWorker(
      LinuxDeepSeekRankServingWorkerTransport transport,
      DeepSeekRankServingPlanWorkerExecutor executor,
      DeepSeekRankServingWorkerLoop loop) noexcept
      : transport_(std::move(transport)), executor_(std::move(executor)),
        loop_(std::move(loop)) {}

  LinuxDeepSeekRankServingWorkerTransport transport_;
  DeepSeekRankServingPlanWorkerExecutor executor_;
  DeepSeekRankServingWorkerLoop loop_;
};

// Controller-side bridge for DeepSeekRankServingCoordinator. It owns neither
// workers nor sockets: the supervisor remains the sole process-domain owner.
class LinuxDeepSeekRankServingControllerOperations final
    : public DeepSeekRankServingChannel {
 public:
  static Result<LinuxDeepSeekRankServingControllerOperations> Create(
      LinuxDeepSeekRankProcessDriver& driver,
      DeepSeekRankProcessSupervisor& supervisor,
      std::span<const DeepSeekRankProcessHandle> ordered_handles);

  Status send_command(std::uint32_t rank,
                      std::span<const std::byte> frame) override;
  Result<std::optional<std::vector<std::byte>>> poll_completion(
      std::uint32_t rank) override;
  Status abort_generation(std::uint64_t engine_epoch,
                          std::uint64_t worker_generation,
                          const Status& cause) override;

 private:
  LinuxDeepSeekRankServingControllerOperations(
      LinuxDeepSeekRankProcessDriver& driver,
      DeepSeekRankProcessSupervisor& supervisor,
      std::vector<DeepSeekRankProcessHandle> handles,
      std::uint64_t engine_epoch,
      std::uint64_t worker_generation) noexcept
      : driver_(&driver), supervisor_(&supervisor), handles_(std::move(handles)),
        engine_epoch_(engine_epoch), worker_generation_(worker_generation) {}
  Result<std::int32_t> control_fd(std::uint32_t rank) const;
  LinuxDeepSeekRankProcessDriver* driver_ = nullptr;
  DeepSeekRankProcessSupervisor* supervisor_ = nullptr;
  std::vector<DeepSeekRankProcessHandle> handles_;
  std::uint64_t engine_epoch_ = 0;
  std::uint64_t worker_generation_ = 0;
};

}  // namespace pih
