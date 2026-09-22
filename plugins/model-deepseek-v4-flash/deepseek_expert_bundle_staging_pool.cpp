#include "pih/model/deepseek_expert_bundle_staging_pool.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekExpertBundleStagingPool>
DeepSeekExpertBundleStagingPool::Allocate(
    std::uint32_t extent_count, RegisteredPinnedAllocator& allocator,
    std::uint32_t transfer_reservation_window) {
  if (transfer_reservation_window == 0 ||
      extent_count < transfer_reservation_window ||
      extent_count > kMaximumExtentCount) {
    return Status::InvalidArgument(
        "DeepSeek expert bundle staging extent count is invalid");
  }
  auto bytes = checked_mul_u64(extent_count,
                               DeepSeekExpertBundleLayout::kBundleBytes);
  if (!bytes.ok() || *bytes > kMaximumBackingBytes) {
    return bytes.ok()
               ? Status::ResourceExhausted(
                     "DeepSeek expert bundle staging exceeds host envelope")
               : bytes.status();
  }
  auto buffer = Buffer::Allocate(
      allocator, *bytes, DeepSeekExpertBundleLayout::kAlignment);
  if (!buffer.ok()) return buffer.status();
  if (buffer->data() == nullptr || buffer->generation() == 0 ||
      buffer->device() != Device::Cpu() ||
      reinterpret_cast<std::uintptr_t>(buffer->data()) %
              DeepSeekExpertBundleLayout::kAlignment !=
          0) {
    return Status::FailedPrecondition(
        "DeepSeek expert bundle staging registration is invalid");
  }
  std::vector<DeepSeekPinnedExpertExtent> extents;
  extents.reserve(extent_count);
  const auto base = reinterpret_cast<std::uintptr_t>(buffer->data());
  for (std::uint32_t index = 0; index < extent_count; ++index) {
    extents.push_back(
        {base + static_cast<std::uint64_t>(index) *
                    DeepSeekExpertBundleLayout::kBundleBytes,
         DeepSeekExpertBundleLayout::kBundleBytes, buffer->generation()});
  }
  return DeepSeekExpertBundleStagingPool(std::move(*buffer),
                                         std::move(extents));
}

}  // namespace pih
