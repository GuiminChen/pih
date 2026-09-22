#pragma once
#include "inference_operation.h"

namespace pih::deepseek_v41 {
struct InferenceMemoryViews final {
  BackboneResources backbone;
  EngramDeviceRegion boundary_device, boundary_host;
};
// One rank/sequence layout. Weights, routed-expert workspace and CUDA/NCCL
// handles have separate ownership and are not included in these budgets.
class InferenceMemoryPlan final {
 public:
  static Result<InferenceMemoryPlan> Create(const FlashConfig& config, std::uint32_t tokens,
      std::uint32_t maximum_positions, std::uint32_t world, std::uint32_t rank,
      std::uint64_t device_budget, std::uint64_t host_budget);
  std::uint64_t device_bytes() const noexcept { return bytes_; }
  std::uint64_t host_bytes() const noexcept { return boundary_->host_bytes(); }
  const BoundaryPlan& boundary() const noexcept { return *boundary_; }
  Result<InferenceMemoryViews> Bind(EngramDeviceRegion device, EngramDeviceRegion host,
      const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
      std::uintptr_t communicator, std::uintptr_t completion_event) const;
 private:
  InferenceMemoryPlan() = default;
  std::optional<BoundaryPlan> boundary_;
  std::optional<StepPhasesPlan> phases_;
  std::optional<EngramPlan> engram_;
  std::optional<SequenceCachePlan> cache_;
  std::optional<FfnPlan> ffn_;
  std::optional<AttentionPreparePlan> attention_;
  std::optional<AttentionOutputPlan> output_;
  std::optional<CompressorPlan> compressor_;
  std::optional<IndexerPlan> indexer_;
  std::array<CacheSegment, 17> segments_{};
  std::uint64_t bytes_ = 0;
};
}  // namespace pih::deepseek_v41
