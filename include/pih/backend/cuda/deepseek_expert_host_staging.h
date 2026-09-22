#pragma once

#include "pih/backend/cuda/deepseek_expert_cuda_backend.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"

namespace pih {

class DeepSeekExpertHostStagingOwner final {
 public:
  static constexpr std::uint64_t kAlignment = 256;

  static Result<DeepSeekExpertHostStagingOwner> Allocate(
      std::uint32_t capacity, RegisteredPinnedAllocator& allocator);

  DeepSeekExpertHostStagingOwner(const DeepSeekExpertHostStagingOwner&) = delete;
  DeepSeekExpertHostStagingOwner& operator=(const DeepSeekExpertHostStagingOwner&) = delete;
  DeepSeekExpertHostStagingOwner(DeepSeekExpertHostStagingOwner&&) noexcept = default;
  DeepSeekExpertHostStagingOwner& operator=(DeepSeekExpertHostStagingOwner&&) noexcept = default;

  [[nodiscard]] DeepSeekExpertHostStaging staging() noexcept;
  [[nodiscard]] std::uint64_t bytes() const noexcept { return buffer_.size_bytes(); }
  [[nodiscard]] std::uint64_t generation() const noexcept { return buffer_.generation(); }

 private:
  DeepSeekExpertHostStagingOwner(Buffer buffer, std::uint32_t capacity,
                                 std::uint64_t indices_offset,
                                 std::uint64_t error_offset)
      : buffer_(std::move(buffer)), capacity_(capacity),
        indices_offset_(indices_offset), error_offset_(error_offset) {}

  Buffer buffer_;
  std::uint32_t capacity_ = 0;
  std::uint64_t indices_offset_ = 0;
  std::uint64_t error_offset_ = 0;
};

}  // namespace pih
