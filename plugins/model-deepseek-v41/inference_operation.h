#pragma once
#include "backbone_operation.h"
#include "boundary_plan.h"
#include "embedding_operation.h"
#include "sampling_operation.h"
#include "rank_commit.h"

namespace pih::deepseek_v41 {
enum class InferenceState { kEmbedding, kBackbone, kHead, kSampling, kComplete, kFailed };
// One rank-local ordinary text step. Last rank samples; all ranks obtain logits.
// A candidate is an observation, never a controller commit or an output token.
class InferenceOperation final {
 public:
  using Clock = std::chrono::steady_clock;
  InferenceOperation(const InferenceOperation&) = delete;
  InferenceOperation& operator=(const InferenceOperation&) = delete;
  ~InferenceOperation();
  static Result<std::unique_ptr<InferenceOperation>> Start(const FlashConfig& config,
      BlockSequence& sequence, EngramHashState& hashes, std::span<const std::uint32_t> tokens,
      const BoundaryPlan& boundary, EngramDeviceRegion device, EngramDeviceRegion host,
      std::uintptr_t stream, const BackboneResources& resources,
      const SamplingParameters& parameters, const SamplingIdentity& identity, Clock::time_point deadline);
  Result<InferenceState> Advance();
  Result<EngramDeviceRegion> Logits() const;
  Result<SamplingObservation> Candidate() const;
  InferenceState state() const noexcept { return state_; }
 private:
  friend class InferenceMemoryOwner;
  Result<RankStepReceipt> Receipt() const;
  InferenceOperation(const FlashConfig& config, BlockSequence& sequence, const BackboneResources& resources,
      const BoundaryViews& views, const SamplingIdentity& identity, Clock::time_point deadline);
  Result<InferenceState> AdvanceImpl();
  FlashConfig config_;
  BlockSequence& sequence_;
  BackboneResources resources_;
  BoundaryViews views_;
  SamplingIdentity identity_;
  Clock::time_point deadline_;
  std::uint32_t step_end_ = 0;
  std::optional<EmbeddingOperation> embedding_;
  std::unique_ptr<BackboneOperation> backbone_;
  std::optional<HeadOperation> head_;
  std::optional<SamplingOperation> sampling_;
  InferenceState state_ = InferenceState::kFailed;
};
}  // namespace pih::deepseek_v41
