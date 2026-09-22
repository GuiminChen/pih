#pragma once
#include "nccl_bootstrap_channel.h"
#include "worker_spawn.h"
#include "worker_cgroup.h"

namespace pih::deepseek_v41 {
enum class NcclBrokerState { kEmpty, kStarting, kReady, kFailed, kRetiring, kRetired };
// Single-threaded CPU supervisor owner. Keep its dedicated cgroup alive and
// explicitly retire/reap this helper; destruction never kills a process.
class NcclBootstrapBroker final {
 public:
  using Clock = WorkerSpawn::Clock;
  NcclBootstrapBroker() = default;
  NcclBootstrapBroker(const NcclBootstrapBroker&) = delete;
  NcclBootstrapBroker& operator=(const NcclBootstrapBroker&) = delete;
  ~NcclBootstrapBroker();
  Status Start(const WorkerExecutable& helper, WorkerCgroup& cgroup,
      const WorkerEnvironment& environment, Clock::time_point startup, Clock::time_point lifetime);
  Result<bool> Poll();
  Result<NcclBootstrapChannel::Id> BorrowId();
  // Only after all ranks finished communicator initialization, or after startup
  // has been cancelled. The service must coordinate this with rank lifecycle.
  Status BeginRetirement(Clock::time_point deadline);
  Result<bool> PollRetirement();
  NcclBrokerState state() const noexcept { return state_; }
  Clock::time_point lifetime_deadline() const noexcept { return lifetime_; }
  const WorkerCgroup* cgroup_owner() const noexcept { return cgroup_; }
 private:
  Status Live() const;
  void Clear() noexcept;
  WorkerSpawn process_;
  WorkerCgroup* cgroup_ = nullptr;
  int socket_ = -1;
  NcclBootstrapChannel::Id id_{};
  Clock::time_point startup_{}, lifetime_{}, retirement_{};
  NcclBrokerState state_ = NcclBrokerState::kEmpty;
};
}  // namespace pih::deepseek_v41
