#pragma once
#include "generation_loop.h"
#include "lifecycle_channel.h"
#include "worker_cgroup.h"
#include "nccl_bootstrap_broker.h"

namespace pih::deepseek_v41 {
enum class GenerationSessionState { kWaitingReady, kReapingBootstrap, kRunning, kOutputReady, kSendingRetire,
  kWaitingReleased, kReaping, kRetiring, kCleaningCgroups, kComplete, kFailedRetired, kFailedUnreconciled };
// Controller parent/subreaper session. Owns execution/retirement operations,
// not worker handles, channels, ledger or publication queue. No background work.
class GenerationSession final {
 private:
  friend class WorkerGroup;
  static Result<std::unique_ptr<GenerationSession>> Create(std::unique_ptr<GenerationLoop> generation,
      std::span<RankLifecycleChannel* const> commands, std::span<RankLifecycleChannel* const> notices,
      std::span<WorkerCgroup* const> cgroups, NcclBootstrapBroker& broker,
      GenerationLoop::Clock::time_point startup_deadline,
      std::chrono::milliseconds terminate_grace, std::chrono::milliseconds retirement_timeout);
 public:
  GenerationSession(const GenerationSession&) = delete;
  GenerationSession& operator=(const GenerationSession&) = delete;
  Result<GenerationSessionState> Poll();
  Status Cancel(Status reason);
  Result<TokenOutputLease> TakeOutput();
  const Status& failure() const noexcept { return failure_; }
  const Status& retirement_failure() const noexcept { return retirement_failure_; }
  std::uint32_t reaped_mask() const noexcept {
    return retirement_ ? retirement_->reaped_mask() : generation_->processes_->reaped_mask();
  }
  std::uint32_t removed_cgroups_mask() const noexcept;
 private:
  GenerationSession() = default;
  Status BeginRetirement(Status reason);
  Status BeginNormalRetirement();
  Status KillCgroups();
  Status BeginBrokerRetirement();
  Result<bool> CleanCgroups();
  Result<GenerationSessionState> PollImpl();
  std::unique_ptr<GenerationLoop> generation_;
  std::unique_ptr<RankProcessRetirement> retirement_;
  std::optional<TokenOutputLease> output_;
  std::chrono::milliseconds grace_{}, timeout_{};
  std::array<RankLifecycleChannel*, 8> commands_{}, notices_{};
  std::array<WorkerCgroup*, 8> cgroups_{};
  NcclBootstrapBroker* broker_ = nullptr;
  GenerationLoop::Clock::time_point lifecycle_deadline_{};
  GenerationLoop::Clock::time_point cgroup_kill_after_{};
  std::uint32_t lifecycle_mask_ = 0;
  GenerationSessionState state_ = GenerationSessionState::kWaitingReady;
  Status failure_ = Status::Ok(), retirement_failure_ = Status::Ok();
};
}  // namespace pih::deepseek_v41
