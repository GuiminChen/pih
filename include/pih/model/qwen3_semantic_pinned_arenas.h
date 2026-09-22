#pragma once

#include <cstdint>
#include <span>

#include "pih/backend/cuda/pinned_placement_verifier.h"
#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/backend/cuda/typed_copy_plan.h"
#include "pih/core/buffer.h"
#include "pih/model/qwen3_kv_semantic_observation_plan.h"
#include "pih/model/qwen3_semantic_observation_transfer.h"

namespace pih {

class QwenSemanticPinnedArenas final {
 public:
  static constexpr std::uint64_t kAlignment = 256;

  static Result<QwenSemanticPinnedArenas> AllocateVerified(
      const QwenKvSemanticObservationPlan& kv_plan,
      RegisteredPinnedAllocator& allocator,
      PinnedPlacementVerifier& verifier, std::int32_t owning_rank,
      std::int32_t numa_node);

  QwenSemanticPinnedArenas(const QwenSemanticPinnedArenas&) = delete;
  QwenSemanticPinnedArenas& operator=(const QwenSemanticPinnedArenas&) = delete;
  QwenSemanticPinnedArenas(QwenSemanticPinnedArenas&&) noexcept = default;
  QwenSemanticPinnedArenas& operator=(QwenSemanticPinnedArenas&&) noexcept =
      default;

  Result<CudaCopyEndpoint> logits_endpoint(std::uint64_t owner_id,
                                           std::uint32_t rank) const;
  Result<CudaCopyEndpoint> kv_endpoint(std::uint64_t owner_id,
                                       std::uint32_t rank) const;
  [[nodiscard]] std::span<const std::byte> logits() const noexcept;
  [[nodiscard]] std::span<const std::byte> kv() const noexcept;
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return logits_.size_bytes() + kv_.size_bytes();
  }

 private:
  QwenSemanticPinnedArenas(Buffer logits, Buffer kv,
                           std::int32_t owning_rank,
                           std::int32_t numa_node)
      : logits_(std::move(logits)), kv_(std::move(kv)),
        owning_rank_(owning_rank), numa_node_(numa_node) {}
  Result<CudaCopyEndpoint> endpoint(const Buffer& buffer,
                                    std::uint64_t owner_id,
                                    std::uint32_t rank) const;

  Buffer logits_;
  Buffer kv_;
  std::int32_t owning_rank_;
  std::int32_t numa_node_;
};

}  // namespace pih
