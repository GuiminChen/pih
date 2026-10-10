#pragma once
#include "packed_worker_memory.h"
#include "request_channel.h"
#include "rank_worker_loop.h"
#include "packed_request_ledger.h"

namespace pih::deepseek_v41 {
struct PackedRankWorkerResources final {
  PackedInferenceMemoryOwner& memory;
  const BackboneWeightUpload& weights;
  ExpertWorkspaceOwner& prefill_workspace;
  ExpertWorkspaceOwner& decode_workspace;
  RankRequestChannel& requests;
  RankReceiptChannel& receipts;
  std::uintptr_t communicator{}, completion_event{};
};
class PackedRankWorkerLoop final {
 public:
  using Clock = RankWorkerLoop::Clock;
  static Result<std::unique_ptr<PackedRankWorkerLoop>> Create(const FlashConfig& config,
      const EngramHashState& template_hashes, const PackedRankWorkerResources& resources,
      std::uint64_t epoch, std::uint32_t maximum_sequences, std::uint32_t maximum_tokens, Clock::time_point deadline);
  Result<RankWorkerState> Poll();
  void Fail() noexcept;
  bool idle() const noexcept;
  static bool CanRetire(RankWorkerState state, const PackedRequestLedger& ledger) noexcept {
    return state == RankWorkerState::kReceiving && ledger.idle();
  }
  static bool NeedsPrefillWorkspace(const PackedRequest& plan) noexcept {
    return std::any_of(plan.spans().begin(), plan.spans().end(), [](const auto& member) {
      return !member.finish && (member.position == 0 || member.token_count > 1);
    });
  }
 private:
  PackedRankWorkerLoop(const FlashConfig& config, const PackedRankWorkerResources& resources)
      : config_(config), resources_(resources) {}
  Result<RankWorkerState> PollImpl();
  struct Slot final {
    SamplingIdentity identity{};
    SamplingParameters parameters{};
    std::uint64_t ordinal{}, last_plan{};
  };
  FlashConfig config_;
  PackedRankWorkerResources resources_;
  std::vector<Slot> slots_;
  std::optional<PackedRequestLedger> ledger_;
  std::vector<std::unique_ptr<BlockSequence>> sequences_;
  std::vector<EngramHashState> hashes_;
  std::vector<BlockSequence*> sequence_views_;
  std::vector<EngramHashState*> hash_views_;
  std::optional<PackedRequest> received_, active_;
  std::vector<RankStepReceipt> receipts_;
  std::uint64_t epoch_{}, last_plan_{};
  std::uint32_t maximum_tokens_{}, receipt_member_{};
  Clock::time_point deadline_{};
  RankWorkerState state_ = RankWorkerState::kReceiving;
};
}
