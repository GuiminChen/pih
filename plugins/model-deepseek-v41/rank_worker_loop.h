#pragma once
#include "inference_memory_owner.h"
#include "request_channel.h"

namespace pih::deepseek_v41 {
struct RankWorkerResources final {
  BlockSequence& sequence;
  EngramHashState& hashes;
  InferenceMemoryOwner& memory;
  const BackboneWeightUpload& weights;
  ExpertWorkspaceOwner& prefill_workspace;
  ExpertWorkspaceOwner& decode_workspace;
  RankRequestChannel& requests;
  RankReceiptChannel& receipts;
  std::uintptr_t communicator = 0, completion_event = 0;
};
enum class RankWorkerState { kReceiving, kExecuting, kSending, kSendingTerminal, kComplete, kFailed };
// One admitted sequence, exclusive borrowed resources, no background thread.
class RankWorkerLoop final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<RankWorkerLoop>> Create(const FlashConfig& config, const RankWorkerResources& resources,
      SamplingIdentity identity, SamplingParameters sampling, std::uint32_t prompt_tokens, Clock::time_point deadline);
  RankWorkerLoop(const RankWorkerLoop&) = delete;
  RankWorkerLoop& operator=(const RankWorkerLoop&) = delete;
  ~RankWorkerLoop();
  Result<RankWorkerState> Poll();
  void Fail() noexcept;
 private:
  RankWorkerLoop(const FlashConfig& config, const RankWorkerResources& resources) : config_(config), resources_(resources) {}
  Result<RankWorkerState> PollImpl();
  FlashConfig config_;
  RankWorkerResources resources_;
  SamplingIdentity identity_{};
  SamplingParameters sampling_{};
  Clock::time_point deadline_{};
  std::uint64_t last_plan_ = 0, ordinal_ = 0;
  std::uint32_t prompt_ = 0;
  RankWorkerState state_ = RankWorkerState::kReceiving;
};
}  // namespace pih::deepseek_v41
