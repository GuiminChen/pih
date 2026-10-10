#pragma once
#include "packed_worker_memory.h"
#include "worker_communicator.h"
#include "packed_rank_worker_loop.h"
#include "lifecycle_channel.h"

namespace pih::deepseek_v41 {
enum class PackedWorkerRuntimeState {
  kEmpty, kConnecting, kLoading, kPublishingReady, kRunning, kAwaitingRetirement,
  kFinalizing, kReleasingMemory, kReleasingHandles, kPublishingReleased, kReleased, kFailed
};
// One persistent rank owns multiple sequence slots on its exclusive execution thread. Dependencies are borrowed
// except the owned sequence, loop and resource ledgers. No destructor GPU cleanup.
class PackedWorkerRuntime final {
 public:
  using Clock = WorkerCommunicator::Clock;
  PackedWorkerRuntime(const FlashConfig& config, const BackboneWeightFiles& files,
      const PackedInferenceMemoryPlan& plan, const pih_nvidia_cuda_api_v1& device_api,
      const pih_nvidia_cuda_resources_api_v1& resources, const pih_nvidia_cuda_memory_api_v1& memory,
      const pih_nvidia_cuda_async_api_v1& async, std::int32_t device,
      EngramHashState& hashes, RankRequestChannel& requests, RankReceiptChannel& receipts,
      RankLifecycleChannel& commands, RankLifecycleChannel& notices)
      : config_(config), plan_(plan), hashes_(hashes), requests_(requests), receipts_(receipts),
        commands_(commands), notices_(notices),
        handles_(device_api, resources, async, device),
        memory_(files, plan, memory, async, handles_, device) {}
  PackedWorkerRuntime(const PackedWorkerRuntime&) = delete;
  PackedWorkerRuntime& operator=(const PackedWorkerRuntime&) = delete;
  Status Start(std::span<const std::byte> bootstrap_id, std::uint32_t sm_major,
      std::uint32_t sm_minor, PackedWorkerMemoryBudget budget, SamplingIdentity identity,
      SamplingParameters sampling, Clock::time_point startup_deadline, Clock::time_point sequence_deadline,
      std::chrono::milliseconds retirement_timeout);
  Result<PackedWorkerRuntimeState> Poll();
  // Stops the local execution state only. Supervisor must retire the epoch and
  // processes; it must never treat this as successful CUDA/NCCL cleanup.
  void Fail() noexcept;
  Status AbortCommunicator();
  PackedWorkerRuntimeState state() const noexcept { return state_; }
  const WorkerHandles& handles() const noexcept { return handles_; }
  const PackedWorkerMemory& memory() const noexcept { return memory_; }
  const WorkerCommunicator& communicator() const noexcept { return communicator_; }
 private:
  Result<PackedWorkerRuntimeState> PollImpl();
  Status BeginRetirement(Clock::time_point deadline);
  FlashConfig config_;
  const PackedInferenceMemoryPlan& plan_;
  EngramHashState& hashes_;
  RankRequestChannel& requests_;
  RankReceiptChannel& receipts_;
  RankLifecycleChannel& commands_;
  RankLifecycleChannel& notices_;
  WorkerHandles handles_;
  PackedWorkerMemory memory_;
  WorkerCommunicator communicator_;
  std::unique_ptr<PackedRankWorkerLoop> loop_;
  PackedWorkerMemoryBudget budget_{};
  SamplingIdentity identity_{};
  SamplingParameters sampling_{};
  Clock::time_point deadline_{}, sequence_deadline_{};
  std::chrono::milliseconds retirement_timeout_{};
  PackedWorkerRuntimeState state_ = PackedWorkerRuntimeState::kEmpty;
};
}  // namespace pih::deepseek_v41
