#pragma once
#include "worker_spawn.h"
#include "worker_bootstrap.h"
#include "rank_channels.h"
#include "worker_cgroup.h"
#include "generation_session.h"
#include "packed_generation_session.h"

namespace pih::deepseek_v41 {
enum class WorkerGroupState { kEmpty, kConnecting, kReady, kHandedOff, kFailed, kRetiring, kRetired };
// Supervisor-only, single-threaded, no CUDA state. Retains all partial launch
// records; destruction closes descriptors but never signals/reaps children.
class WorkerGroup final {
 public:
  using Clock = WorkerSpawn::Clock;
  WorkerGroup() = default;
  WorkerGroup(const WorkerGroup&) = delete;
  WorkerGroup& operator=(const WorkerGroup&) = delete;
  Status Start(const WorkerExecutable& executable, std::span<WorkerCgroup* const> cgroups,
      std::span<const WorkerBootstrap> bootstraps, const WorkerEnvironment& environment, NcclBootstrapBroker& broker);
  Result<bool> Poll();
  // Transfer custody only after successful session construction. Failed
  // construction leaves startup retirement with this group. Keep group, ledger,
  // token byte table and cgroup owners alive until the returned session finishes.
  Result<std::unique_ptr<GenerationSession>> StartGeneration(TokenLedger& ledger,
      std::span<const std::uint32_t> prompt, std::span<const std::string_view> token_bytes,
      std::uint64_t first_plan, std::chrono::milliseconds terminate_grace);
  // Persistent packed execution retains process custody in this group. Caller
  // must retire the group before destroying session borrowers or cgroups.
  Result<bool> PollPackedReady();
  Result<std::unique_ptr<PackedGenerationSession>> StartPackedGeneration(
      std::uint64_t epoch, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, std::uint32_t maximum_pending,
      std::chrono::milliseconds retirement_timeout, std::uint64_t first_plan = 1);
  Status BeginStartupRetirement(Clock::time_point deadline);
  Result<bool> PollStartupRetirement();
  // Retry the same retained custody with a fresh bounded close-call deadline.
  Status ResumeStartupRetirement(Clock::time_point deadline);
  WorkerGroupState state() const noexcept { return state_; }
  bool has_startup_custody() const noexcept { return broker_ != nullptr; }
  std::uint32_t spawned_mask() const noexcept;
  std::uint32_t reaped_mask() const noexcept;
 private:
  std::array<WorkerSpawn, 8> workers_;
  std::array<std::unique_ptr<RankChannels>, 8> channels_;
  std::optional<RankProcessWatch> processes_;
  std::array<RankRequestChannel*, 8> requests_{};
  std::array<RankReceiptChannel*, 8> receipts_{};
  std::array<RankLifecycleChannel*, 8> commands_{}, notices_{};
  std::array<WorkerCgroup*, 8> cgroups_{};
  NcclBootstrapBroker* broker_ = nullptr;
  std::uint32_t world_ = 0, connected_ = 0;
  Clock::time_point deadline_{};
  Clock::time_point sequence_deadline_{};
  SamplingIdentity identity_{};
  SamplingParameters sampling_{};
  std::uint32_t prompt_tokens_ = 0, maximum_positions_ = 0, retirement_ms_ = 0;
  WorkerGroupState state_ = WorkerGroupState::kEmpty;
  std::uint32_t packed_ready_mask_ = 0;
  bool packed_handed_off_ = false;
};
}  // namespace pih::deepseek_v41
