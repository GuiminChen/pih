#include "pih/backend/cuda/deepseek_expert_host_staging.h"

namespace pih {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
  return (value + alignment - 1U) & ~(alignment - 1U);
}

}  // namespace

Result<DeepSeekExpertHostStagingOwner>
DeepSeekExpertHostStagingOwner::Allocate(
    std::uint32_t capacity, RegisteredPinnedAllocator& allocator) {
  if (capacity == 0 ||
      capacity > DeepSeekExpertComputeArenaLayout::kMaximumTokens) {
    return Status::InvalidArgument("DeepSeek host staging capacity is invalid");
  }
  const auto weights_bytes = static_cast<std::uint64_t>(capacity) * sizeof(float);
  const auto indices_offset = align_up(weights_bytes, kAlignment);
  const auto indices_bytes =
      static_cast<std::uint64_t>(capacity) * sizeof(std::uint32_t);
  const auto error_offset = align_up(indices_offset + indices_bytes, kAlignment);
  const auto required_bytes = align_up(error_offset + sizeof(std::uint32_t),
                                       kAlignment);
  auto buffer = Buffer::Allocate(allocator, required_bytes, kAlignment);
  if (!buffer.ok()) return buffer.status();
  if (buffer->data() == nullptr || buffer->generation() == 0 ||
      buffer->device().type() != DeviceType::kCpu ||
      buffer->device().index() != 0 ||
      reinterpret_cast<std::uintptr_t>(buffer->data()) % kAlignment != 0) {
    return Status::FailedPrecondition(
        "DeepSeek pinned allocator returned an invalid allocation");
  }
  return DeepSeekExpertHostStagingOwner(
      std::move(*buffer), capacity, indices_offset, error_offset);
}

DeepSeekExpertHostStaging DeepSeekExpertHostStagingOwner::staging() noexcept {
  auto* base = static_cast<std::byte*>(buffer_.data());
  return {
      reinterpret_cast<float*>(base),
      reinterpret_cast<std::uint32_t*>(base + indices_offset_),
      reinterpret_cast<std::uint32_t*>(base + error_offset_), capacity_};
}

}  // namespace pih
