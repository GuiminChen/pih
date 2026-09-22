#pragma once
#include "supervisor_request.h"
#include "worker_group.h"

namespace pih::deepseek_v41 {
enum class SupervisorOperationState { kEmpty, kStartingBroker, kConnectingRanks, kSession,
  kOutputReady, kRetiringStartup, kComplete, kFailedRetired, kFailedUnreconciled };
// Serialized CPU orchestration. Borrowed admissions/request/cgroups must outlive
// this owner and outstanding output leases. No destructor-driven process work.
class SupervisorOperation final {
 public:
  using Clock = WorkerSpawn::Clock;
  SupervisorOperation() = default;
  SupervisorOperation(const SupervisorOperation&) = delete;
  SupervisorOperation& operator=(const SupervisorOperation&) = delete;
  // Placements supply artifact/device/endpoint/deadline fields; NCCL IDs must
  // be empty. Sequence fields are filled from the authenticated request.
  Status Start(const WorkerExecutable& rank_executable, const WorkerExecutable& helper_executable,
      const WorkerEnvironment& environment, SupervisorRequest& request,
      std::span<const WorkerBootstrap> placements, std::span<WorkerCgroup* const> rank_cgroups,
      WorkerCgroup& helper_cgroup, std::uint64_t first_plan, std::chrono::milliseconds terminate_grace);
  Result<SupervisorOperationState> Poll();
  Status Cancel(Status reason);
  Result<TokenOutputLease> TakeOutput();
  SupervisorOperationState state() const noexcept { return state_; }
  const Status& failure() const noexcept { return failure_; }
  const Status& cleanup_failure() const noexcept { return cleanup_failure_; }
 private:
  Status Fail(Status reason);
  Result<SupervisorOperationState> PollImpl();
  Result<bool> RetireUnlaunched();
  const WorkerExecutable* executable_ = nullptr;
  const WorkerEnvironment* environment_ = nullptr;
  SupervisorRequest* request_ = nullptr;
  WorkerCgroup* helper_group_ = nullptr;
  std::vector<WorkerBootstrap> placements_;
  std::array<WorkerCgroup*, 8> groups_{};
  std::uint32_t world_ = 0;
  std::uint64_t first_plan_ = 0;
  std::chrono::milliseconds grace_{}, timeout_{};
  Clock::time_point retirement_{};
  NcclBootstrapBroker broker_;
  WorkerGroup group_;
  std::unique_ptr<GenerationSession> session_;
  SupervisorOperationState state_ = SupervisorOperationState::kEmpty;
  Status failure_ = Status::Ok(), cleanup_failure_ = Status::Ok();
};
}  // namespace pih::deepseek_v41
