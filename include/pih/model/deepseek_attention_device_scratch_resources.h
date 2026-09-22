#pragma once

#include "pih/core/buffer.h"
#include "pih/model/deepseek_index_selection_driver.h"

namespace pih {

class DeepSeekAttentionDeviceScratchResources final {
 public:
  static constexpr std::uint32_t kMaximumScoreTile = 4096;
  static constexpr std::uint32_t kMaximumSparseRow = 8320;

  static Result<DeepSeekAttentionDeviceScratchResources> Allocate(
      Allocator& allocator, std::uint32_t maximum_queries,
      std::uint32_t maximum_ratio4_logical_pages,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekAttentionDeviceScratchResources(
      const DeepSeekAttentionDeviceScratchResources&) = delete;
  DeepSeekAttentionDeviceScratchResources& operator=(
      const DeepSeekAttentionDeviceScratchResources&) = delete;
  DeepSeekAttentionDeviceScratchResources(
      DeepSeekAttentionDeviceScratchResources&&) noexcept = default;
  DeepSeekAttentionDeviceScratchResources& operator=(
      DeepSeekAttentionDeviceScratchResources&&) noexcept = default;

  [[nodiscard]] DeepSeekIndexSelectionArena index_arena() const noexcept;
  [[nodiscard]] std::uintptr_t indexer_query_bf16() const noexcept;
  [[nodiscard]] std::uintptr_t indexer_qr_e4m3() const noexcept {
    return reinterpret_cast<std::uintptr_t>(indexer_qr_.data());
  }
  [[nodiscard]] std::uintptr_t indexer_qr_scale_bits() const noexcept {
    return reinterpret_cast<std::uintptr_t>(indexer_qr_scales_.data());
  }
  [[nodiscard]] std::uintptr_t indexer_head_weight_f32() const noexcept;
  [[nodiscard]] std::uintptr_t sparse_indices_i32() const noexcept;
  [[nodiscard]] std::uintptr_t sparse_page_slots_u32() const noexcept;
  [[nodiscard]] std::uintptr_t compressor_error_u32() const noexcept;
  [[nodiscard]] std::uintptr_t page_error_u32() const noexcept;
  [[nodiscard]] std::uintptr_t index_error_u32() const noexcept;
  [[nodiscard]] std::uintptr_t sparse_error_u32() const noexcept;
  [[nodiscard]] std::uintptr_t indexer_projection_error_u32() const noexcept;
  [[nodiscard]] std::uint32_t maximum_queries() const noexcept {
    return maximum_queries_;
  }

 private:
  DeepSeekAttentionDeviceScratchResources(
      Buffer scores, Buffer indices, Buffer indexer_query,
      Buffer indexer_head_weight, Buffer page_slots, Buffer errors,
      Buffer indexer_qr, Buffer indexer_qr_scales,
      std::uint32_t maximum_queries, std::uint64_t context_identity,
      std::int32_t device_ordinal) noexcept
      : scores_(std::move(scores)), indices_(std::move(indices)),
        indexer_query_(std::move(indexer_query)),
        indexer_qr_(std::move(indexer_qr)), indexer_qr_scales_(std::move(indexer_qr_scales)),
        indexer_head_weight_(std::move(indexer_head_weight)),
        page_slots_(std::move(page_slots)),
        errors_(std::move(errors)), maximum_queries_(maximum_queries),
        context_identity_(context_identity), device_ordinal_(device_ordinal) {}

  std::uintptr_t error(std::uint32_t ordinal) const noexcept;
  Buffer scores_;
  Buffer indices_;
  Buffer indexer_query_;
  Buffer indexer_qr_;
  Buffer indexer_qr_scales_;
  Buffer indexer_head_weight_;
  Buffer page_slots_;
  Buffer errors_;
  std::uint32_t maximum_queries_ = 0;
  std::uint64_t context_identity_ = 0;
  std::int32_t device_ordinal_ = -1;
};

}  // namespace pih
