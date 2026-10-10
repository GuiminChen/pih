#include "packed_boundary_plan.h"

namespace pih::deepseek_v41 {
namespace { std::uint64_t Align(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; } }
Result<PackedBoundaryPlan> PackedBoundaryPlan::Create(const FlashConfig& config, std::uint32_t maximum_tokens,
    std::uint32_t maximum_sequences, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget, std::uint64_t host_budget) {
  if (!maximum_sequences || maximum_sequences > 32 || maximum_sequences > maximum_tokens ||
      (world != 2 && world != 4 && world != 8) || rank >= world)
    return Status::InvalidArgument("Packed boundary batch/rank geometry invalid");
  const auto rows_bytes = Align(maximum_sequences * 4ULL);
  const auto local_bytes = Align(maximum_sequences * (129280ULL / world) * 4);
  const auto logits_bytes = Align(maximum_sequences * 129280ULL * 4);
  const auto extra_device = rows_bytes + local_bytes + 2 * logits_bytes;
  if (device_budget < extra_device || host_budget < rows_bytes)
    return Status::ResourceExhausted("Packed boundary extra vocabulary storage exceeds budget");
  auto ordinary = BoundaryPlan::Create(config, maximum_tokens, world, rank, device_budget - extra_device, host_budget - rows_bytes);
  if (!ordinary.ok()) return ordinary.status();
  PackedBoundaryPlan plan(std::move(*ordinary)); plan.sequences_ = maximum_sequences; plan.world_ = world;
  plan.local_offset_ = Align(plan.ordinary_.device_bytes());
  plan.gathered_offset_ = plan.local_offset_ + local_bytes;
  plan.logits_offset_ = plan.gathered_offset_ + logits_bytes;
  plan.rows_offset_ = plan.logits_offset_ + logits_bytes;
  plan.device_bytes_ = plan.rows_offset_ + rows_bytes;
  plan.host_rows_offset_ = Align(plan.ordinary_.host_bytes()); plan.host_bytes_ = plan.host_rows_offset_ + rows_bytes;
  if (plan.device_bytes_ > device_budget || plan.host_bytes_ > host_budget)
    return Status::ResourceExhausted("Packed boundary aligned reservation exceeds budget");
  return plan;
}
Result<PackedBoundaryViews> PackedBoundaryPlan::Bind(EngramDeviceRegion device, EngramDeviceRegion host,
    std::uint32_t tokens, std::uint32_t sequences, std::uintptr_t stream) const {
  if (!sequences || sequences > sequences_ || sequences > tokens || device.bytes != device_bytes_ || host.bytes != host_bytes_ ||
      !device.address || device.address % 256 || !host.address || host.address % 256 ||
      device.bytes > UINTPTR_MAX - device.address || host.bytes > UINTPTR_MAX - host.address)
    return Status::InvalidArgument("Packed boundary binding extent or capacity invalid");
  auto ordinary = ordinary_.Bind({device.address, ordinary_.device_bytes()}, {host.address, ordinary_.host_bytes()}, tokens, stream);
  if (!ordinary.ok()) return ordinary.status();
  PackedBoundaryViews views; views.ordinary = *ordinary;
  views.gathered_logits = {device.address + gathered_offset_, sequences * 129280ULL * 4};
  views.logits = {device.address + logits_offset_, sequences * 129280ULL * 4};
  views.device_head_rows = {device.address + rows_offset_, sequences * 4ULL};
  views.host_head_rows = {host.address + host_rows_offset_, sequences * 4ULL};
  auto& projection = views.ordinary.head.projection;
  projection.output = {device.address + local_offset_, sequences * (129280ULL / world_) * 4};
  projection.row_indices = views.device_head_rows; projection.sequences = sequences; projection.input_rows = tokens;
  projection.input = views.ordinary.head.input.norm.output;
  return views;
}
}
