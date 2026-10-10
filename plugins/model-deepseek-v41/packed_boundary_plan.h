#pragma once
#include "boundary_plan.h"

namespace pih::deepseek_v41 {
struct PackedBoundaryViews final {
  BoundaryViews ordinary;
  EngramDeviceRegion gathered_logits, logits, host_head_rows, device_head_rows;
};
class PackedBoundaryPlan final {
 public:
  static Result<PackedBoundaryPlan> Create(const FlashConfig& config, std::uint32_t maximum_tokens,
      std::uint32_t maximum_sequences, std::uint32_t world, std::uint32_t rank,
      std::uint64_t device_budget, std::uint64_t host_budget);
  std::uint64_t device_bytes() const noexcept { return device_bytes_; }
  std::uint64_t host_bytes() const noexcept { return host_bytes_; }
  std::uint32_t token_capacity() const noexcept { return ordinary_.token_capacity(); }
  std::uint32_t world_size() const noexcept { return ordinary_.world_size(); }
  std::uint32_t rank() const noexcept { return ordinary_.rank(); }
  const Sha256Digest& config_sha256() const noexcept { return ordinary_.config_sha256(); }
  Result<PackedBoundaryViews> Bind(EngramDeviceRegion device, EngramDeviceRegion host,
      std::uint32_t tokens, std::uint32_t sequences, std::uintptr_t stream) const;
 private:
  explicit PackedBoundaryPlan(BoundaryPlan ordinary) : ordinary_(std::move(ordinary)) {}
  BoundaryPlan ordinary_;
  std::uint64_t local_offset_{}, gathered_offset_{}, logits_offset_{}, rows_offset_{}, host_rows_offset_{};
  std::uint64_t device_bytes_{}, host_bytes_{};
  std::uint32_t sequences_{}, world_{};
};
}
