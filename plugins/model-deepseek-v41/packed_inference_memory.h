#pragma once
#include "inference_memory.h"
#include "packed_inference_operation.h"

namespace pih::deepseek_v41 {
struct PackedInferenceMemoryViews final {
  std::vector<InferenceMemoryViews> slots;
  std::array<EngramDeviceRegion, 2> ffn_arenas;
  EngramDeviceRegion attention_residual, attention_pre, boundary_device, boundary_host;
};
class PackedInferenceMemoryPlan final {
 public:
  static Result<PackedInferenceMemoryPlan> Create(const FlashConfig& config, std::uint32_t maximum_tokens,
      std::uint32_t maximum_sequences, std::uint32_t maximum_positions, std::uint32_t world, std::uint32_t rank,
      std::uint64_t device_budget, std::uint64_t host_budget);
  std::uint64_t device_bytes() const noexcept { return bytes_; }
  std::uint64_t host_bytes() const noexcept { return host_bytes_; }
  std::uint32_t maximum_sequences() const noexcept { return sequences_; }
  std::uint32_t maximum_tokens() const noexcept { return tokens_; }
  const PackedBoundaryPlan& boundary() const noexcept { return *boundary_; }
  const FfnPlan& ffn() const noexcept { return *ffn_; }
  Result<PackedInferenceMemoryViews> Bind(EngramDeviceRegion device, EngramDeviceRegion host,
      const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace,
      std::uintptr_t communicator, std::uintptr_t event) const;
 private:
  std::optional<InferenceMemoryPlan> ordinary_;
  std::optional<PackedBoundaryPlan> boundary_;
  std::optional<FfnPlan> ffn_;
  std::array<CacheSegment, 5> shared_{};
  std::uint64_t bytes_{}, host_bytes_{}, boundary_host_offset_{};
  std::uint32_t sequences_{}, tokens_{};
};
}
