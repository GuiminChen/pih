#pragma once
#include "inference_operation.h"
#include "packed_backbone_operation.h"
#include "packed_boundary_plan.h"

namespace pih::deepseek_v41 {
// One native rank-local plan: shared embedding, 40 shared FFN executions and
// shared vocabulary projection, with private attention/KV/Engram sequence state.
class PackedInferenceOperation final {
 public:
  using Clock = InferenceOperation::Clock;
  static Result<std::unique_ptr<PackedInferenceOperation>> Start(const FlashConfig& config,
      PackedRequest plan, std::span<BlockSequence* const> sequences, std::span<EngramHashState* const> hashes,
      const PackedBoundaryPlan& boundary, EngramDeviceRegion device, EngramDeviceRegion host,
      std::uintptr_t stream, const PackedBackboneResources& backbone, Clock::time_point deadline);
  ~PackedInferenceOperation();
  Result<InferenceState> Advance();
  Result<RankStepReceipt> Receipt(std::uint32_t member) const;
  Result<EngramDeviceRegion> Logits(std::uint32_t member) const;
 private:
  PackedInferenceOperation(const FlashConfig& config, PackedRequest plan, const PackedBackboneResources& resources,
      PackedBoundaryViews views, Clock::time_point deadline)
      : config_(config), plan_(std::move(plan)), resources_(resources), views_(views), deadline_(deadline) {}
  Result<InferenceState> AdvanceImpl();
  Status StartSampling();
  void Fail() noexcept;
  FlashConfig config_;
  PackedRequest plan_;
  PackedBackboneResources resources_;
  PackedBoundaryViews views_;
  Clock::time_point deadline_;
  std::vector<BlockSequence*> slots_;
  std::vector<BackboneResources> member_resources_;
  std::vector<SamplingObservation> observations_;
  std::optional<EmbeddingOperation> embedding_;
  std::unique_ptr<PackedBackboneOperation> backbone_;
  std::optional<HeadOperation> head_;
  std::optional<SamplingOperation> sampling_;
  std::uint32_t sampling_member_{};
  InferenceState state_ = InferenceState::kFailed;
};
}
