#pragma once

#include <string_view>

#include "pih/core/result.h"

namespace pih {

// This lifecycle is deliberately independent of the Linux socket adapter. It
// makes the worker-only ownership cut testable: the controller can drive the
// wire protocol, but it cannot build a rank's CUDA resources or manufacture a
// materialization completion on that rank's behalf.
inline constexpr std::string_view
    kDeepSeekRankWorkerMaterializationLifecycleAbi =
        "pih_deepseek_rank_worker_materialization_lifecycle_v1";

class DeepSeekRankWorkerStartupProtocol {
 public:
  virtual ~DeepSeekRankWorkerStartupProtocol() = default;

  // Each operation is allowed to return Unavailable while the authenticated
  // control channel is backpressured. Any other failure is terminal and the
  // hosting worker process must exit so the controller tears down the group.
  virtual Status run_resource_barrier() = 0;
  virtual Status run_artifact_transfer() = 0;
  virtual Status run_materialization_completion() = 0;
  [[nodiscard]] virtual bool materialization_completion_sent() const noexcept =
      0;
};

class DeepSeekRankWorkerMaterializationApplication {
 public:
  virtual ~DeepSeekRankWorkerMaterializationApplication() = default;

  // Implementations own the rank-local resources. materialize() must consume
  // the worker's prefaulted grant inputs and construct CUDA/pinned/NCCL state
  // only in that worker. begin_completion() must bind those same retained
  // resources to LinuxDeepSeekRankWorkerStartup::begin_materialization_completion.
  virtual Status materialize() = 0;
  virtual Status begin_completion() = 0;
};

enum class DeepSeekRankWorkerMaterializationState : std::uint8_t {
  kAwaitingResourceBarrier,
  kAwaitingArtifactTransfer,
  kMaterializing,
  kSendingCompletion,
  kReady,
  kFailed,
};

class DeepSeekRankWorkerMaterializationLifecycle final {
 public:
  static Result<DeepSeekRankWorkerMaterializationLifecycle> Create(
      DeepSeekRankWorkerStartupProtocol& startup,
      DeepSeekRankWorkerMaterializationApplication& application);

  DeepSeekRankWorkerMaterializationLifecycle(
      const DeepSeekRankWorkerMaterializationLifecycle&) = delete;
  DeepSeekRankWorkerMaterializationLifecycle& operator=(
      const DeepSeekRankWorkerMaterializationLifecycle&) = delete;
  DeepSeekRankWorkerMaterializationLifecycle(
      DeepSeekRankWorkerMaterializationLifecycle&&) noexcept = default;
  DeepSeekRankWorkerMaterializationLifecycle& operator=(
      DeepSeekRankWorkerMaterializationLifecycle&&) noexcept = default;

  // Advances exactly one durable phase. Callers may retry only after an
  // Unavailable result; all other failures leave the lifecycle terminal.
  Status advance();
  [[nodiscard]] DeepSeekRankWorkerMaterializationState state() const noexcept {
    return state_;
  }
  [[nodiscard]] bool ready() const noexcept {
    return state_ == DeepSeekRankWorkerMaterializationState::kReady;
  }
  [[nodiscard]] bool failed() const noexcept {
    return state_ == DeepSeekRankWorkerMaterializationState::kFailed;
  }

 private:
  DeepSeekRankWorkerMaterializationLifecycle(
      DeepSeekRankWorkerStartupProtocol& startup,
      DeepSeekRankWorkerMaterializationApplication& application) noexcept
      : startup_(&startup), application_(&application) {}

  Status fail(Status cause) noexcept;

  DeepSeekRankWorkerStartupProtocol* startup_ = nullptr;
  DeepSeekRankWorkerMaterializationApplication* application_ = nullptr;
  DeepSeekRankWorkerMaterializationState state_ =
      DeepSeekRankWorkerMaterializationState::kAwaitingResourceBarrier;
  bool completion_begun_ = false;
};

}  // namespace pih
