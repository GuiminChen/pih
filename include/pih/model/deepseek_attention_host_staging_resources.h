#pragma once

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/buffer.h"
#include "pih/model/deepseek_attention_runtime_resources.h"

namespace pih {

class DeepSeekAttentionHostStagingResources final {
 public:
  static constexpr std::uint32_t kMaximumScoreTile = 4096;
  static constexpr std::uint32_t kMaximumSparseRow = 8320;

  static Result<DeepSeekAttentionHostStagingResources> Allocate(
      RegisteredPinnedAllocator& allocator, std::uint32_t maximum_queries,
      std::uint32_t maximum_ratio4_logical_pages);

  DeepSeekAttentionHostStagingResources(
      const DeepSeekAttentionHostStagingResources&) = delete;
  DeepSeekAttentionHostStagingResources& operator=(
      const DeepSeekAttentionHostStagingResources&) = delete;
  DeepSeekAttentionHostStagingResources(
      DeepSeekAttentionHostStagingResources&&) noexcept = default;
  DeepSeekAttentionHostStagingResources& operator=(
      DeepSeekAttentionHostStagingResources&&) noexcept = default;

  [[nodiscard]] DeepSeekIndexSelectionHostStaging index() noexcept;
  [[nodiscard]] DeepSeekSparseAttentionHostStaging sparse() noexcept;
  [[nodiscard]] std::uint32_t* compressor_error() noexcept;
  [[nodiscard]] std::uint32_t* page_error() noexcept;
  [[nodiscard]] std::uint32_t* indexer_projection_error() noexcept;
  [[nodiscard]] std::uint32_t maximum_queries() const noexcept {
    return maximum_queries_;
  }

 private:
  DeepSeekAttentionHostStagingResources(
      Buffer scores, Buffer indices, Buffer page_slots, Buffer errors,
      std::uint32_t maximum_queries,
      std::uint32_t maximum_ratio4_logical_pages) noexcept
      : scores_(std::move(scores)), indices_(std::move(indices)),
        page_slots_(std::move(page_slots)), errors_(std::move(errors)),
        maximum_queries_(maximum_queries),
        maximum_ratio4_logical_pages_(maximum_ratio4_logical_pages) {}

  std::uint32_t* error(std::uint32_t ordinal) noexcept;
  Buffer scores_;
  Buffer indices_;
  Buffer page_slots_;
  Buffer errors_;
  std::uint32_t maximum_queries_ = 0;
  std::uint32_t maximum_ratio4_logical_pages_ = 0;
};

}  // namespace pih
