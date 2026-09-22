#pragma once

#include <vector>

#include "pih/model/deepseek_attention_physical_layout.h"
#include "pih/model/deepseek_attention_page_arena.h"
#include "pih/model/deepseek_attention_sequence_device_resources.h"
#include "pih/model/deepseek_v4_config.h"

namespace pih {

class DeepSeekRankAttentionStatePool final {
 public:
  static Result<DeepSeekRankAttentionStatePool> Allocate(
      Allocator& allocator, DeepSeekStagePlan stage,
      std::uint32_t maximum_sequences,
      std::uint32_t reserved_tokens_per_sequence,
      std::uintptr_t completion_event,
      DeepSeekFixedStateBankOperations& fixed_operations,
      std::uint64_t context_identity, std::int32_t device_ordinal);

  DeepSeekRankAttentionStatePool(
      const DeepSeekRankAttentionStatePool&) = delete;
  DeepSeekRankAttentionStatePool& operator=(
      const DeepSeekRankAttentionStatePool&) = delete;
  DeepSeekRankAttentionStatePool(
      DeepSeekRankAttentionStatePool&&) noexcept = default;
  DeepSeekRankAttentionStatePool& operator=(
      DeepSeekRankAttentionStatePool&&) noexcept = default;

  [[nodiscard]] DeepSeekAttentionSequenceDeviceResources& sequence(
      std::uint32_t slot) { return sequences_.at(slot); }
  [[nodiscard]] DeepSeekRatio4PagePool& ratio4_pool() noexcept {
    return ratio4_pool_;
  }
  [[nodiscard]] DeepSeekRatio128PagePool& ratio128_pool() noexcept {
    return ratio128_pool_;
  }
  [[nodiscard]] DeepSeekAttentionPageArena page_arena() const noexcept {
    return page_arena_;
  }
  [[nodiscard]] const DeepSeekFixedStateLayout& fixed_layout() const noexcept {
    return physical_layout_.fixed_layout();
  }
  [[nodiscard]] const DeepSeekAttentionPhysicalLayout& physical_layout()
      const noexcept {
    return physical_layout_;
  }
  [[nodiscard]] std::uint32_t sequence_capacity() const noexcept {
    return static_cast<std::uint32_t>(sequences_.size());
  }
  [[nodiscard]] std::uint32_t reserved_tokens_per_sequence() const noexcept {
    return reserved_tokens_per_sequence_;
  }
  [[nodiscard]] std::uint32_t ratio4_page_pairs_per_sequence() const noexcept {
    return ratio4_page_pairs_per_sequence_;
  }
  [[nodiscard]] std::uint32_t ratio128_pages_per_sequence() const noexcept {
    return ratio128_pages_per_sequence_;
  }

 private:
  DeepSeekRankAttentionStatePool(
      DeepSeekAttentionPhysicalLayout physical_layout, Buffer page_storage,
      DeepSeekRatio4PagePool ratio4_pool,
      DeepSeekRatio128PagePool ratio128_pool,
      DeepSeekAttentionPageArena page_arena,
      std::vector<DeepSeekAttentionSequenceDeviceResources> sequences,
      std::uint32_t reserved_tokens_per_sequence,
      std::uint32_t ratio4_page_pairs_per_sequence,
      std::uint32_t ratio128_pages_per_sequence) noexcept
      : physical_layout_(std::move(physical_layout)),
        page_storage_(std::move(page_storage)),
        ratio4_pool_(std::move(ratio4_pool)),
        ratio128_pool_(std::move(ratio128_pool)), page_arena_(page_arena),
        sequences_(std::move(sequences)),
        reserved_tokens_per_sequence_(reserved_tokens_per_sequence),
        ratio4_page_pairs_per_sequence_(ratio4_page_pairs_per_sequence),
        ratio128_pages_per_sequence_(ratio128_pages_per_sequence) {}

  DeepSeekAttentionPhysicalLayout physical_layout_;
  Buffer page_storage_;
  DeepSeekRatio4PagePool ratio4_pool_;
  DeepSeekRatio128PagePool ratio128_pool_;
  DeepSeekAttentionPageArena page_arena_;
  std::vector<DeepSeekAttentionSequenceDeviceResources> sequences_;
  std::uint32_t reserved_tokens_per_sequence_ = 0;
  std::uint32_t ratio4_page_pairs_per_sequence_ = 0;
  std::uint32_t ratio128_pages_per_sequence_ = 0;
};

}  // namespace pih
