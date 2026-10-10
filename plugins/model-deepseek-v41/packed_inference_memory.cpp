#include "packed_inference_memory.h"

namespace pih::deepseek_v41 {
namespace { std::uint64_t Align(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; } }
Result<PackedInferenceMemoryPlan> PackedInferenceMemoryPlan::Create(const FlashConfig& config, std::uint32_t tokens,
    std::uint32_t sequences, std::uint32_t positions, std::uint32_t world, std::uint32_t rank,
    std::uint64_t device_budget, std::uint64_t host_budget) {
  if (!sequences || sequences > 32 || tokens < sequences || tokens > 4096)
    return Status::InvalidArgument("Packed inference memory batch bounds invalid");
  PackedInferenceMemoryPlan plan; plan.sequences_ = sequences; plan.tokens_ = tokens;
  auto ordinary = InferenceMemoryPlan::Create(config, tokens, positions, world, rank, device_budget / sequences, host_budget / sequences);
  if (!ordinary.ok()) return ordinary.status(); plan.ordinary_.emplace(std::move(*ordinary));
  plan.bytes_ = Align(plan.ordinary_->device_bytes()) * sequences;
  plan.host_bytes_ = Align(plan.ordinary_->host_bytes()) * sequences;
  if (plan.bytes_ > device_budget || plan.host_bytes_ > host_budget)
    return Status::ResourceExhausted("Packed aligned member allocations exceed memory budget");
  auto boundary = PackedBoundaryPlan::Create(config, tokens, sequences, world, rank,
      device_budget - plan.bytes_, host_budget - plan.host_bytes_);
  if (!boundary.ok()) return boundary.status(); plan.boundary_.emplace(std::move(*boundary));
  auto ffn = FfnPlan::Create(config, tokens, world, rank, device_budget - plan.bytes_);
  if (!ffn.ok()) return ffn.status(); plan.ffn_.emplace(std::move(*ffn));
  const std::array<std::uint64_t, 5> shared_sizes{plan.boundary_->device_bytes(), plan.ffn_->bytes(), plan.ffn_->bytes(),
      tokens * 4ULL * 5120 * 2, tokens * 4ULL * 4};
  for (unsigned i = 0; i < shared_sizes.size(); ++i) {
    const auto bytes = Align(shared_sizes[i]);
    if (bytes > device_budget - plan.bytes_) return Status::ResourceExhausted("Packed combined device memory exceeds budget");
    plan.shared_[i] = {plan.bytes_, shared_sizes[i]}; plan.bytes_ += bytes;
  }
  plan.boundary_host_offset_ = plan.host_bytes_;
  if (plan.boundary_->host_bytes() > host_budget - plan.host_bytes_)
    return Status::ResourceExhausted("Packed combined host memory exceeds budget");
  plan.host_bytes_ += Align(plan.boundary_->host_bytes());
  if (plan.host_bytes_ > host_budget) return Status::ResourceExhausted("Packed aligned host memory exceeds budget");
  return plan;
}
Result<PackedInferenceMemoryViews> PackedInferenceMemoryPlan::Bind(EngramDeviceRegion device, EngramDeviceRegion host,
    const BackboneWeightUpload& weights, ExpertWorkspaceOwner& workspace, std::uintptr_t communicator, std::uintptr_t event) const {
  if (!device.address || device.address % 256 || device.bytes != bytes_ || !host.address || host.address % 256 ||
      host.bytes != host_bytes_ || device.bytes > UINTPTR_MAX - device.address || host.bytes > UINTPTR_MAX - host.address)
    return Status::InvalidArgument("Packed inference combined allocation extent invalid");
  PackedInferenceMemoryViews views; views.slots.reserve(sequences_);
  for (unsigned slot = 0; slot < sequences_; ++slot) {
    auto member = ordinary_->Bind({device.address + slot * Align(ordinary_->device_bytes()), ordinary_->device_bytes()},
        {host.address + slot * Align(ordinary_->host_bytes()), ordinary_->host_bytes()}, weights, workspace, communicator, event);
    if (!member.ok()) return member.status(); views.slots.push_back(std::move(*member));
  }
  const auto region = [&](unsigned index) { return EngramDeviceRegion{device.address + shared_[index].offset, shared_[index].bytes}; };
  views.boundary_device = region(0); views.ffn_arenas = {region(1), region(2)};
  views.attention_residual = region(3); views.attention_pre = region(4);
  views.boundary_host = {host.address + boundary_host_offset_, boundary_->host_bytes()};
  return views;
}
}
