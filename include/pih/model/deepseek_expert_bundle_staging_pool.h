#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_expert_transfer_driver.h"

namespace pih {

class DeepSeekExpertBundleStagingPool final {
 public:
  static constexpr std::uint64_t kMaximumBackingBytes = 8ULL * 1024 * 1024 * 1024;
  static constexpr std::uint32_t kMaximumExtentCount =
      static_cast<std::uint32_t>(kMaximumBackingBytes /
                                 DeepSeekExpertBundleLayout::kBundleBytes);

  static Result<DeepSeekExpertBundleStagingPool> Allocate(
      std::uint32_t extent_count, RegisteredPinnedAllocator& allocator,
      std::uint32_t transfer_reservation_window =
          DeepSeekExpertPager::kTransferReservationWindow);

  DeepSeekExpertBundleStagingPool(
      const DeepSeekExpertBundleStagingPool&) = delete;
  DeepSeekExpertBundleStagingPool& operator=(
      const DeepSeekExpertBundleStagingPool&) = delete;
  DeepSeekExpertBundleStagingPool(
      DeepSeekExpertBundleStagingPool&&) noexcept = default;
  DeepSeekExpertBundleStagingPool& operator=(
      DeepSeekExpertBundleStagingPool&&) noexcept = default;

  [[nodiscard]] std::span<const DeepSeekPinnedExpertExtent> extents()
      const noexcept { return extents_; }
  [[nodiscard]] std::uint32_t extent_count() const noexcept {
    return static_cast<std::uint32_t>(extents_.size());
  }
  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return buffer_.size_bytes();
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return buffer_.generation();
  }

 private:
  DeepSeekExpertBundleStagingPool(
      Buffer buffer, std::vector<DeepSeekPinnedExpertExtent> extents)
      : buffer_(std::move(buffer)), extents_(std::move(extents)) {}

  Buffer buffer_;
  std::vector<DeepSeekPinnedExpertExtent> extents_;
};

}  // namespace pih
