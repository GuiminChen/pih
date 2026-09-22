#pragma once
#include "rank_process_watch.h"
#include "worker_executable.h"
#include "worker_environment.h"
#include <string>

namespace pih::deepseek_v41 {
enum class WorkerSpawnState { kEmpty, kCheckingExec, kAwaitingChannels, kFailed, kKilled, kReaped };
// Linux clone3/execveat launcher. Call only from a dedicated single-threaded
// supervisor before CUDA initialization. Owns pidfd and exec-status pipe, not
// a process-killing destructor. Admit executable/environment and seal bootstrap first.
class WorkerSpawn final {
 public:
  using Clock = std::chrono::steady_clock;
  WorkerSpawn() = default;
  WorkerSpawn(const WorkerSpawn&) = delete;
  WorkerSpawn& operator=(const WorkerSpawn&) = delete;
  ~WorkerSpawn();
  // bootstrap is mapped to FD 3. Other FDs >=4 are CLOEXEC. stdio is inherited.
  // cgroup_fd must name the already configured delegated worker cgroup v2.
  Status Start(const WorkerExecutable& executable, int bootstrap_fd, int cgroup_fd,
      std::span<const std::string> arguments, const WorkerEnvironment& environment,
      Clock::time_point deadline);
  // Private inherited packet channel at FD 3, not a rank bootstrap envelope.
  Status StartNcclBroker(const WorkerExecutable& executable, int socket_fd, int cgroup_fd,
      const WorkerEnvironment& environment, Clock::time_point startup_deadline,
      Clock::time_point lifetime_deadline);
  Result<bool> PollExec();
  // Usable immediately after successful clone, including failed exec. Watch
  // Attach duplicates pidfds; this owner must not reap after handing off custody.
  RankProcessBinding binding() const noexcept { return binding_; }
  WorkerSpawnState state() const noexcept { return state_; }
  int child_errno() const noexcept { return child_errno_; }
  // For partial startup before a full rank watch exists; exclusive parent only.
  Status Kill();
  Result<bool> PollKilled(Clock::time_point deadline);
 private:
  Status StartImpl(const WorkerExecutable& executable, int bootstrap_fd, int cgroup_fd,
      std::span<const std::string> arguments, const WorkerEnvironment& environment,
      Clock::time_point deadline, bool broker);
  RankProcessBinding binding_{};
  int errors_ = -1, child_errno_ = 0;
  std::array<unsigned char, sizeof(int)> error_bytes_{};
  std::size_t received_ = 0;
  Clock::time_point deadline_{};
  WorkerSpawnState state_ = WorkerSpawnState::kEmpty;
};
}  // namespace pih::deepseek_v41
