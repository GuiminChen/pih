#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/sha256.h"

namespace pih {

class DeepSeekNcclBootstrapLease final {
 public:
  static constexpr std::size_t kUniqueIdBytes = 128;
  static Result<DeepSeekNcclBootstrapLease> Create(
      RegisteredPinnedAllocator& allocator, std::span<const std::byte> raw_id,
      std::uint64_t engine_epoch, std::uint32_t edge_id,
      std::uint64_t lease_id);

  DeepSeekNcclBootstrapLease(const DeepSeekNcclBootstrapLease&) = delete;
  DeepSeekNcclBootstrapLease& operator=(const DeepSeekNcclBootstrapLease&) = delete;
  DeepSeekNcclBootstrapLease(DeepSeekNcclBootstrapLease&& other) noexcept;
  DeepSeekNcclBootstrapLease& operator=(DeepSeekNcclBootstrapLease&& other) noexcept;
  ~DeepSeekNcclBootstrapLease();

  Result<std::span<const std::byte>> borrow(std::uint64_t engine_epoch,
                                            std::uint32_t edge_id,
                                            std::uint64_t lease_id) const;
  Status zeroize() noexcept;
  [[nodiscard]] const Sha256Digest& commitment() const noexcept {
    return commitment_;
  }
  [[nodiscard]] bool zeroized() const noexcept { return zeroized_; }

 private:
  DeepSeekNcclBootstrapLease(RegisteredPinnedAllocator& allocator,
                             Allocation allocation, std::uint64_t engine_epoch,
                             std::uint32_t edge_id, std::uint64_t lease_id,
                             Sha256Digest commitment) noexcept;
  void release() noexcept;

  RegisteredPinnedAllocator* allocator_ = nullptr;
  Allocation allocation_{};
  std::uint64_t engine_epoch_ = 0;
  std::uint32_t edge_id_ = 0;
  std::uint64_t lease_id_ = 0;
  Sha256Digest commitment_{};
  bool zeroized_ = false;
};

}  // namespace pih
