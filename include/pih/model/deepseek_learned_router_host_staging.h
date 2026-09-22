#pragma once

#include <span>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_learned_router_coordinator.h"

namespace pih {

class DeepSeekLearnedRouterHostStaging final {
 public:
  static constexpr std::uint64_t kAlignment = 256;

  static Result<DeepSeekLearnedRouterHostStaging> Allocate(
      std::uint32_t maximum_tokens, RegisteredPinnedAllocator& allocator);

  DeepSeekLearnedRouterHostStaging(
      const DeepSeekLearnedRouterHostStaging&) = delete;
  DeepSeekLearnedRouterHostStaging& operator=(
      const DeepSeekLearnedRouterHostStaging&) = delete;
  DeepSeekLearnedRouterHostStaging(
      DeepSeekLearnedRouterHostStaging&&) noexcept = default;
  DeepSeekLearnedRouterHostStaging& operator=(
      DeepSeekLearnedRouterHostStaging&&) noexcept = default;

  Result<std::span<float>> scores(std::uint32_t token_count) noexcept;
  [[nodiscard]] std::uint32_t* error_flag() noexcept;
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return buffer_.generation();
  }

 private:
  DeepSeekLearnedRouterHostStaging(
      Buffer buffer, std::uint32_t maximum_tokens,
      std::uint64_t error_offset) noexcept
      : buffer_(std::move(buffer)), maximum_tokens_(maximum_tokens),
        error_offset_(error_offset) {}

  Buffer buffer_;
  std::uint32_t maximum_tokens_ = 0;
  std::uint64_t error_offset_ = 0;
};

}  // namespace pih
