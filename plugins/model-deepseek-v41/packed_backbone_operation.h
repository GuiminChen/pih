#pragma once
#include "backbone_operation.h"
#include "packed_request.h"

namespace pih::deepseek_v41 {
struct PackedBackboneResources final {
  // Canonical plan order, each entry owns distinct persistent caches and source
  // scratch. Weights, communicator, completion event and expert workspace share
  // a single execution owner across the rank.
  std::span<const BackboneResources> members;
  const FfnPlan& ffn;
  std::array<EngramDeviceRegion, 2> ffn_arenas;
  EngramDeviceRegion attention_residual, attention_pre;
};
class PackedBackboneOperation final {
 public:
  using Clock = BackboneOperation::Clock;
  static Result<std::unique_ptr<PackedBackboneOperation>> Start(const FlashConfig& config,
      const PackedRequest& plan, std::span<BlockSequence* const> sequences,
      const PackedBackboneResources& resources, Clock::time_point deadline);
  ~PackedBackboneOperation();
  Result<BackboneState> Advance();
  Result<SequenceStepOutput> Output(std::uint32_t member) const;
 private:
  PackedBackboneOperation(const FlashConfig& config, PackedRequest plan,
      const PackedBackboneResources& resources, Clock::time_point deadline)
      : config_(config), plan_(std::move(plan)), resources_(resources), deadline_(deadline) {}
  Status StartLayer();
  Status StartSources();
  Result<BackboneState> AdvanceImpl();
  void Fail() noexcept;
  FlashConfig config_;
  PackedRequest plan_;
  PackedBackboneResources resources_;
  Clock::time_point deadline_;
  std::vector<BlockSequence*> sequences_;
  std::vector<BlockOperation> sources_;
  std::optional<PreparedBlockOperation> block_;
  FfnViews ffn_views_{};
  std::uint64_t reservation_{};
  std::uint32_t layer_{}, source_member_{};
  BackboneState state_ = BackboneState::kFailed;
};
}
