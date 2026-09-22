#include "sequence_cache.h"
#include <limits>

namespace pih::deepseek_v41 {
Result<SequenceCachePlan> SequenceCachePlan::Create(const FlashConfig& config,
    std::uint32_t maximum, std::uint64_t budget) {
  if (config.config_sha256() == Sha256Digest{} || !maximum || maximum > FlashConfig::kMaximumPositions)
    return Status::InvalidArgument("Sequence cache requires admitted config and bounded context capacity");
  SequenceCachePlan plan; plan.config_sha256_ = config.config_sha256(); plan.maximum_positions_ = maximum;
  auto reserve = [&](CacheSegment& segment, std::uint64_t bytes) -> Status {
    if (!bytes) return Status::Ok();
    const auto aligned = (bytes + 255) & ~std::uint64_t{255};
    if (aligned > budget - plan.bytes_) return Status::ResourceExhausted("Sequence cache exceeds supplied budget");
    segment = {plan.bytes_, bytes}; plan.bytes_ += aligned; return Status::Ok();
  };
  for (std::uint32_t layer = 0; layer < 40; ++layer) {
    auto& layout = plan.layers_[layer];
    const auto& role = config.attention_sharing()[layer];
    auto status = reserve(layout.window, 128ULL * 512 * 2); if (!status.ok()) return status;
    if (!role.owns_kv) continue;
    layout.compressed_capacity = maximum / role.compression_ratio;
    status = reserve(layout.compressed, std::uint64_t(layout.compressed_capacity) * 512 * 2);
    if (!status.ok()) return status;
    status = reserve(layout.index_keys, std::uint64_t(layout.compressed_capacity) * 128 * 2);
    if (!status.ok()) return status;
    if (role.compression_ratio == 2) {
      status = reserve(layout.pool_values, 2 * 512 * 4); if (!status.ok()) return status;
      status = reserve(layout.pool_scores, 2 * 512 * 4); if (!status.ok()) return status;
    }
  }
  return plan;
}
Result<LayerCacheRegions> SequenceCachePlan::Bind(EngramDeviceRegion arena, std::uint32_t layer) const {
  if (layer >= 40 || !arena.address || arena.address % 256 || arena.bytes != bytes_ ||
      arena.bytes > std::numeric_limits<std::uintptr_t>::max() - arena.address)
    return Status::InvalidArgument("Sequence cache arena extent/alignment or layer invalid");
  const auto& layout = layers_[layer];
  const auto region = [&](CacheSegment s) -> EngramDeviceRegion {
    return s.bytes ? EngramDeviceRegion{arena.address + s.offset, s.bytes} : EngramDeviceRegion{};
  };
  return LayerCacheRegions{region(layout.window), region(layout.compressed), region(layout.index_keys),
      region(layout.pool_values), region(layout.pool_scores), layout.compressed_capacity};
}
Status SequenceCachePlan::ValidateAllocation(const pih_cuda_allocation_v1& a, std::int32_t device) const {
  if (device < 0 || a.struct_size != sizeof(a) || a.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      a.memory_kind != PIH_CUDA_MEMORY_DEVICE_V1 || a.device_ordinal != device || !a.generation ||
      a.alignment < 256 || (a.alignment & (a.alignment - 1)) || !a.address || a.address % a.alignment)
    return Status::FailedPrecondition("Sequence cache allocation identity/device/alignment invalid");
  auto bound = Bind({a.address, a.bytes}, 0); if (!bound.ok()) return bound.status();
  return Status::Ok();
}
Status SequenceCachePlan::Allocate(const pih_nvidia_cuda_memory_api_v1& memory, std::int32_t device,
    pih_cuda_allocation_v1& output) const {
  if (memory.struct_size != sizeof(memory) || memory.contract_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      !memory.context || !memory.allocate_device || !memory.deallocate_device || device < 0 ||
      output.struct_size != sizeof(output) || output.abi_version != PIH_NVIDIA_CUDA_MEMORY_ABI_VERSION_V1 ||
      output.address || output.bytes || output.alignment || output.generation || output.memory_kind || output.device_ordinal)
    return Status::InvalidArgument("Sequence cache memory capability or empty output ledger invalid");
  const auto status = memory.allocate_device(memory.context, device, bytes_, 256, &output);
  if (!pih_status_is_valid_v1(&status)) return Status::Internal("Cache allocator returned malformed status; quarantine ledger");
  if (!pih_status_is_ok_v1(&status)) {
    if (output.address || output.bytes || output.generation)
      return Status::Internal("Cache allocation failed with ambiguous ownership; quarantine ledger");
    switch (status.code) {
      case PIH_STATUS_RESOURCE_EXHAUSTED_V1: return Status::ResourceExhausted("Cache allocator exhausted memory");
      case PIH_STATUS_UNAVAILABLE_V1: return Status::Unavailable("Cache allocator unavailable");
      case PIH_STATUS_DEADLINE_EXCEEDED_V1: return Status::DeadlineExceeded("Cache allocator deadline exceeded");
      default: return Status::FailedPrecondition("Cache allocator rejected allocation");
    }
  }
  return ValidateAllocation(output, device);
}
}  // namespace pih::deepseek_v41
