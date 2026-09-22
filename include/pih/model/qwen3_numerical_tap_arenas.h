#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_numerical_tap_plan.h"

namespace pih {

class QwenNumericalTapArenas final {
 public:
  static Result<QwenNumericalTapArenas> AllocateVerified(
      const QwenNumericalTapPlan& plan, Allocator& device_allocator,
      RegisteredPinnedAllocator& pinned_allocator,
      PinnedPlacementVerifier& placement_verifier, std::int32_t owning_rank,
      std::int32_t numa_node);

  QwenNumericalTapArenas(const QwenNumericalTapArenas&) = delete;
  QwenNumericalTapArenas& operator=(const QwenNumericalTapArenas&) = delete;
  QwenNumericalTapArenas(QwenNumericalTapArenas&&) noexcept = default;
  QwenNumericalTapArenas& operator=(QwenNumericalTapArenas&&) noexcept = default;

  Result<CudaCopyEndpoint> device_endpoint(std::size_t capture_index,
                                           std::uint64_t owner_id,
                                           std::uint32_t rank) const;
  Result<CudaCopyEndpoint> pinned_endpoint(std::size_t capture_index,
                                           std::uint64_t owner_id,
                                           std::uint32_t rank) const;
  Result<std::span<const std::byte>> pinned_capture(
      std::size_t capture_index) const;

  [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
    return device_.size_bytes();
  }

 private:
  QwenNumericalTapArenas(Buffer device, Buffer pinned,
                         std::vector<QwenNumericalTapCapture> captures,
                         std::int32_t owning_rank, std::int32_t numa_node)
      : device_(std::move(device)), pinned_(std::move(pinned)),
        captures_(std::move(captures)), owning_rank_(owning_rank),
        numa_node_(numa_node) {}
  Result<CudaCopyEndpoint> endpoint(const Buffer& buffer,
                                    CudaCopyMemoryType memory_type,
                                    std::int32_t device_or_numa,
                                    std::size_t capture_index,
                                    std::uint64_t owner_id,
                                    std::uint32_t rank) const;

  Buffer device_;
  Buffer pinned_;
  std::vector<QwenNumericalTapCapture> captures_;
  std::int32_t owning_rank_;
  std::int32_t numa_node_;
};

}  // namespace pih
