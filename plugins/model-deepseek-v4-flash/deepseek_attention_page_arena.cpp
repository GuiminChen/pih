#include "pih/model/deepseek_attention_page_arena.h"

#include <limits>

namespace pih {
namespace {

constexpr std::uint64_t kRatio4MainPageBytes = 65536;
constexpr std::uint64_t kRatio4IndexPageBytes = 16384;
constexpr std::uint64_t kRatio128MainPageBytes = 65536;
constexpr std::uintptr_t kPageAlignment = 256;

bool valid_span(DeepSeekExpertArenaSpan span, std::uint32_t pages,
                std::uint64_t page_bytes) {
  if (span.address == 0 || pages == 0 ||
      span.address % kPageAlignment != 0 ||
      pages > std::numeric_limits<std::uint64_t>::max() / page_bytes) {
    return false;
  }
  const auto expected = static_cast<std::uint64_t>(pages) * page_bytes;
  return span.bytes == expected &&
         span.address <= std::numeric_limits<std::uintptr_t>::max() - expected;
}

}  // namespace

Result<DeepSeekAttentionPageArena> DeepSeekAttentionPageArena::Create(
    DeepSeekExpertArenaSpan ratio4_main,
    DeepSeekExpertArenaSpan ratio4_index,
    DeepSeekExpertArenaSpan ratio128_main,
    std::uint32_t ratio4_page_pairs,
    std::uint32_t ratio128_pages) {
  if (!valid_span(ratio4_main, ratio4_page_pairs, kRatio4MainPageBytes) ||
      !valid_span(ratio4_index, ratio4_page_pairs, kRatio4IndexPageBytes) ||
      !valid_span(ratio128_main, ratio128_pages, kRatio128MainPageBytes)) {
    return Status::InvalidArgument(
        "DeepSeek attention page arena geometry is invalid");
  }
  DeepSeekAttentionPageArena arena;
  arena.ratio4_main_ = ratio4_main;
  arena.ratio4_index_ = ratio4_index;
  arena.ratio128_main_ = ratio128_main;
  arena.ratio4_page_pairs_ = ratio4_page_pairs;
  arena.ratio128_pages_ = ratio128_pages;
  return arena;
}

Result<DeepSeekExpertArenaSpan> DeepSeekAttentionPageArena::Resolve(
    DeepSeekBlockHandle handle) const {
  DeepSeekExpertArenaSpan arena;
  std::uint32_t pages = 0;
  std::uint64_t page_bytes = 0;
  switch (handle.pool_kind()) {
    case DeepSeekStatePoolKind::kRatio4MainBf16:
      arena = ratio4_main_;
      pages = ratio4_page_pairs_;
      page_bytes = kRatio4MainPageBytes;
      break;
    case DeepSeekStatePoolKind::kRatio4IndexBf16:
      arena = ratio4_index_;
      pages = ratio4_page_pairs_;
      page_bytes = kRatio4IndexPageBytes;
      break;
    case DeepSeekStatePoolKind::kRatio128MainBf16:
      arena = ratio128_main_;
      pages = ratio128_pages_;
      page_bytes = kRatio128MainPageBytes;
      break;
    default:
      return Status::InvalidArgument(
          "DeepSeek page handle has no compressed arena");
  }
  if (handle.generation() == 0 || handle.slot() >= pages) {
    return Status::InvalidArgument(
        "DeepSeek page handle is outside its compressed arena");
  }
  return DeepSeekExpertArenaSpan{
      arena.address + static_cast<std::uint64_t>(handle.slot()) * page_bytes,
      page_bytes};
}

}  // namespace pih
