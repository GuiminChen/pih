#pragma once

#include <span>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_bf16_engine_resource_plan.h"

namespace pih {

class QwenBf16PinnedHostArenas final {
 public:
  static Result<QwenBf16PinnedHostArenas> Allocate(
      const QwenBf16EngineResourcePlan& plan,
      RegisteredPinnedAllocator& allocator);
  static Result<QwenBf16PinnedHostArenas> AllocateVerified(
      const QwenBf16EngineResourcePlan& plan,
      RegisteredPinnedAllocator& allocator,
      PinnedPlacementVerifier& verifier, std::int32_t numa_node);

  QwenBf16PinnedHostArenas(const QwenBf16PinnedHostArenas&) = delete;
  QwenBf16PinnedHostArenas& operator=(
      const QwenBf16PinnedHostArenas&) = delete;
  QwenBf16PinnedHostArenas(QwenBf16PinnedHostArenas&&) noexcept = default;
  QwenBf16PinnedHostArenas& operator=(
      QwenBf16PinnedHostArenas&&) noexcept = default;

  [[nodiscard]] std::span<std::byte> staging() noexcept {
    return {static_cast<std::byte*>(staging_.data()),
            static_cast<std::size_t>(staging_.size_bytes())};
  }
  [[nodiscard]] std::span<std::byte> result() noexcept {
    return {static_cast<std::byte*>(result_.data()),
            static_cast<std::size_t>(result_.size_bytes())};
  }
  Result<CudaCopyEndpoint> staging_endpoint(
      std::uint64_t owner_id, std::uint32_t rank) const;
  Result<CudaCopyEndpoint> result_endpoint(
      std::uint64_t owner_id, std::uint32_t rank) const;
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return staging_.size_bytes() + result_.size_bytes();
  }

 private:
  QwenBf16PinnedHostArenas(Buffer staging, Buffer result,
                           std::int32_t verified_numa_node)
      : staging_(std::move(staging)), result_(std::move(result)),
        verified_numa_node_(verified_numa_node) {}
  static Result<CudaCopyEndpoint> endpoint(
      const Buffer& buffer, std::uint64_t owner_id, std::uint32_t rank,
      std::int32_t numa_node);

  Buffer staging_;
  Buffer result_;
  std::int32_t verified_numa_node_ = -1;
};

}  // namespace pih
