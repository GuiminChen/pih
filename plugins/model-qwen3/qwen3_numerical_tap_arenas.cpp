#include "pih/model/qwen3_numerical_tap_arenas.h"

#include <cstdint>

namespace pih {

Result<QwenNumericalTapArenas> QwenNumericalTapArenas::AllocateVerified(
    const QwenNumericalTapPlan& plan, Allocator& device_allocator,
    RegisteredPinnedAllocator& pinned_allocator,
    PinnedPlacementVerifier& placement_verifier, std::int32_t owning_rank,
    std::int32_t numa_node) {
  if (owning_rank < 0 || numa_node < 0) {
    return Status::InvalidArgument(
        "Qwen numerical tap arena placement is invalid");
  }
  auto device = Buffer::Allocate(device_allocator, plan.arena_bytes(),
                                 QwenNumericalTapPlan::kAlignment);
  if (!device.ok()) return device.status();
  auto pinned = Buffer::Allocate(pinned_allocator, plan.arena_bytes(),
                                 QwenNumericalTapPlan::kAlignment);
  if (!pinned.ok()) return pinned.status();
  const bool device_valid =
      device->data() != nullptr && device->size_bytes() == plan.arena_bytes() &&
      device->generation() != 0 &&
      device->device().type() == DeviceType::kCuda &&
      device->device().index() == owning_rank &&
      reinterpret_cast<std::uintptr_t>(device->data()) %
              QwenNumericalTapPlan::kAlignment ==
          0;
  const bool pinned_valid =
      pinned->data() != nullptr && pinned->size_bytes() == plan.arena_bytes() &&
      pinned->generation() != 0 &&
      pinned->device().type() == DeviceType::kCpu &&
      pinned->device().index() == 0 &&
      reinterpret_cast<std::uintptr_t>(pinned->data()) %
              QwenNumericalTapPlan::kAlignment ==
          0;
  if (!device_valid || !pinned_valid) {
    return Status::FailedPrecondition(
        "Qwen numerical tap allocator returned an invalid arena");
  }
  const Status placed = placement_verifier.verify(
      pinned->data(), pinned->size_bytes(), numa_node);
  if (!placed.ok()) return placed;
  std::vector<QwenNumericalTapCapture> captures(plan.captures().begin(),
                                                plan.captures().end());
  return QwenNumericalTapArenas(std::move(*device), std::move(*pinned),
                                std::move(captures), owning_rank, numa_node);
}

Result<CudaCopyEndpoint> QwenNumericalTapArenas::endpoint(
    const Buffer& buffer, CudaCopyMemoryType memory_type,
    std::int32_t device_or_numa, std::size_t capture_index,
    std::uint64_t owner_id, std::uint32_t rank) const {
  if (capture_index >= captures_.size() || owner_id == 0 ||
      rank != static_cast<std::uint32_t>(owning_rank_)) {
    return Status::InvalidArgument(
        "Qwen numerical tap endpoint identity is invalid");
  }
  return CudaCopyEndpoint{
      reinterpret_cast<std::uintptr_t>(buffer.data()), buffer.size_bytes(),
      captures_[capture_index].offset_bytes, owner_id, buffer.generation(),
      memory_type, rank, device_or_numa};
}

Result<CudaCopyEndpoint> QwenNumericalTapArenas::device_endpoint(
    std::size_t capture_index, std::uint64_t owner_id,
    std::uint32_t rank) const {
  return endpoint(device_, CudaCopyMemoryType::kDevice, owning_rank_,
                  capture_index, owner_id, rank);
}

Result<CudaCopyEndpoint> QwenNumericalTapArenas::pinned_endpoint(
    std::size_t capture_index, std::uint64_t owner_id,
    std::uint32_t rank) const {
  return endpoint(pinned_, CudaCopyMemoryType::kRegisteredPinnedHost,
                  numa_node_, capture_index, owner_id, rank);
}

Result<std::span<const std::byte>> QwenNumericalTapArenas::pinned_capture(
    std::size_t capture_index) const {
  if (capture_index >= captures_.size()) {
    return Status::InvalidArgument(
        "Qwen numerical tap capture index is invalid");
  }
  const auto& capture = captures_[capture_index];
  const auto* begin = static_cast<const std::byte*>(pinned_.data()) +
                      capture.offset_bytes;
  return std::span<const std::byte>(begin,
                                    static_cast<std::size_t>(capture.size_bytes));
}

}  // namespace pih
