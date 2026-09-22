#pragma once
#include "worker_memory.h"
#include "worker_communicator.h"
#include "rank_worker_loop.h"
#include "lifecycle_channel.h"

namespace pih::deepseek_v41 {
enum class WorkerRuntimeState {
  kEmpty, kConnecting, kLoading, kPublishingReady, kRunning, kAwaitingRetirement,
  kFinalizing, kReleasingMemory, kReleasingHandles, kPublishingReleased, kReleased, kFailed
};
// Single rank, one sequence, exclusive thread. All dependencies are borrowed
// except the owned sequence, loop and resource ledgers. No destructor GPU cleanup.
class WorkerRuntime final {
 public:
  using Clock = WorkerCommunicator::Clock;
  WorkerRuntime(const FlashConfig& config, const BackboneWeightFiles& files,
      const InferenceMemoryPlan& plan, const pih_nvidia_cuda_api_v1& device_api,
      const pih_nvidia_cuda_resources_api_v1& resources, const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
      EngramHashState& hashes, RankRequestChannel& requests, RankReceiptChannel& receipts,
      RankLifecycleChannel& commands, RankLifecycleChannel& notices)
      : config_(config), plan_(plan), hashes_(hashes), requests_(requests), receipts_(receipts),
        commands_(commands), notices_(notices),
        handles_(device_api, resources, async, device),
        memory_(files, plan, memory, async, handles_, device), sequence_(config) {}
  WorkerRuntime(const WorkerRuntime&) = delete;
  WorkerRuntime& operator=(const WorkerRuntime&) = delete;
  Status Start(std::span<const std::byte> bootstrap_id, std::uint32_t sm_major,
      std::uint32_t sm_minor, WorkerMemoryBudget budget, SamplingIdentity identity,
      SamplingParameters sampling, Clock::time_point startup_deadline, Clock::time_point sequence_deadline,
      std::chrono::milliseconds retirement_timeout);
  Result<WorkerRuntimeState> Poll();
  // Stops the local execution state only. Supervisor must retire the epoch and
  // processes; it must never treat this as successful CUDA/NCCL cleanup.
  void Fail() noexcept;
  Status AbortCommunicator();
  WorkerRuntimeState state() const noexcept { return state_; }
  const WorkerHandles& handles() const noexcept { return handles_; }
  const WorkerMemory& memory() const noexcept { return memory_; }
  const WorkerCommunicator& communicator() const noexcept { return communicator_; }
 private:
  Result<WorkerRuntimeState> PollImpl();
  Status BeginRetirement(Clock::time_point deadline);
  FlashConfig config_;
  const InferenceMemoryPlan& plan_;
  EngramHashState& hashes_;
  RankRequestChannel& requests_;
  RankReceiptChannel& receipts_;
  RankLifecycleChannel& commands_;
  RankLifecycleChannel& notices_;
  WorkerHandles handles_;
  WorkerMemory memory_;
  WorkerCommunicator communicator_;
  BlockSequence sequence_;
  std::unique_ptr<RankWorkerLoop> loop_;
  WorkerMemoryBudget budget_{};
  SamplingIdentity identity_{};
  SamplingParameters sampling_{};
  Clock::time_point deadline_{}, sequence_deadline_{};
  std::chrono::milliseconds retirement_timeout_{};
  WorkerRuntimeState state_ = WorkerRuntimeState::kEmpty;
};
}  // namespace pih::deepseek_v41
