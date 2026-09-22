#include "pih/model/qwen3_bf16_pinned_arenas.h"

namespace pih {

Result<QwenBf16PinnedHostArenas> QwenBf16PinnedHostArenas::Allocate(
    const QwenBf16EngineResourcePlan& plan,
    RegisteredPinnedAllocator& allocator) {
  auto staging = Buffer::Allocate(allocator, plan.step_staging_bytes(), 256);
  if (!staging.ok()) return staging.status();
  auto result = Buffer::Allocate(allocator, plan.pinned_result_bytes(), 256);
  if (!result.ok()) return result.status();
  for (const auto* buffer : {&*staging, &*result}) {
    if (buffer->device().type() != DeviceType::kCpu ||
        buffer->device().index() != 0 || buffer->generation() == 0 ||
        buffer->data() == nullptr ||
        reinterpret_cast<std::uintptr_t>(buffer->data()) % 256 != 0) {
      return Status::FailedPrecondition(
          "Qwen pinned allocator returned an invalid allocation");
    }
  }
  return QwenBf16PinnedHostArenas(
      std::move(*staging), std::move(*result), -1);
}

Result<QwenBf16PinnedHostArenas>
QwenBf16PinnedHostArenas::AllocateVerified(
    const QwenBf16EngineResourcePlan& plan,
    RegisteredPinnedAllocator& allocator, PinnedPlacementVerifier& verifier,
    std::int32_t numa_node) {
  if (numa_node < 0) {
    return Status::InvalidArgument("Qwen pinned NUMA node is invalid");
  }
  auto arenas = Allocate(plan, allocator);
  if (!arenas.ok()) return arenas.status();
  Status status = verifier.verify(
      arenas->staging_.data(), arenas->staging_.size_bytes(), numa_node);
  if (!status.ok()) return status;
  status = verifier.verify(
      arenas->result_.data(), arenas->result_.size_bytes(), numa_node);
  if (!status.ok()) return status;
  arenas->verified_numa_node_ = numa_node;
  return arenas;
}

Result<CudaCopyEndpoint> QwenBf16PinnedHostArenas::endpoint(
    const Buffer& buffer, std::uint64_t owner_id, std::uint32_t rank,
    std::int32_t numa_node) {
  if (owner_id == 0 || rank == UINT32_MAX || numa_node < 0) {
    return Status::InvalidArgument("Qwen pinned endpoint identity is invalid");
  }
  return CudaCopyEndpoint{
      reinterpret_cast<std::uintptr_t>(buffer.data()), buffer.size_bytes(), 0,
      owner_id, buffer.generation(),
      CudaCopyMemoryType::kRegisteredPinnedHost, rank, numa_node};
}

Result<CudaCopyEndpoint> QwenBf16PinnedHostArenas::staging_endpoint(
    std::uint64_t owner_id, std::uint32_t rank) const {
  if (verified_numa_node_ < 0) {
    return Status::FailedPrecondition(
        "Qwen pinned staging placement is not verified");
  }
  return endpoint(staging_, owner_id, rank, verified_numa_node_);
}

Result<CudaCopyEndpoint> QwenBf16PinnedHostArenas::result_endpoint(
    std::uint64_t owner_id, std::uint32_t rank) const {
  if (verified_numa_node_ < 0) {
    return Status::FailedPrecondition(
        "Qwen pinned result placement is not verified");
  }
  return endpoint(result_, owner_id, rank, verified_numa_node_);
}

}  // namespace pih
