#pragma once

#include <cstdint>

#include "pih/model/deepseek_attention_page_pool.h"
#include "pih/model/deepseek_expert_compute_arena.h"

namespace pih {

class DeepSeekAttentionPageArena final {
 public:
  static Result<DeepSeekAttentionPageArena> Create(
      DeepSeekExpertArenaSpan ratio4_main,
      DeepSeekExpertArenaSpan ratio4_index,
      DeepSeekExpertArenaSpan ratio128_main,
      std::uint32_t ratio4_page_pairs,
      std::uint32_t ratio128_pages);

  Result<DeepSeekExpertArenaSpan> Resolve(DeepSeekBlockHandle handle) const;
  [[nodiscard]] std::uintptr_t ratio4_index_base() const noexcept {
    return ratio4_index_.address;
  }
  [[nodiscard]] std::uintptr_t ratio4_main_base() const noexcept {
    return ratio4_main_.address;
  }
  [[nodiscard]] std::uintptr_t ratio128_main_base() const noexcept {
    return ratio128_main_.address;
  }

 private:
  DeepSeekExpertArenaSpan ratio4_main_;
  DeepSeekExpertArenaSpan ratio4_index_;
  DeepSeekExpertArenaSpan ratio128_main_;
  std::uint32_t ratio4_page_pairs_ = 0;
  std::uint32_t ratio128_pages_ = 0;
};

}  // namespace pih
