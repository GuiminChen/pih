#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_execution_arena.h"

namespace pih {

class QwenBf16StepResultLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::uint64_t kTotalBytes = 512;
  static constexpr std::int64_t kVocabularySize = 151936;

  static constexpr QwenBf16ArenaSpan sampled_token() noexcept {
    return {0, sizeof(std::int64_t)};
  }
  static constexpr QwenBf16ArenaSpan device_error() noexcept {
    return {kAlignment, sizeof(std::uint32_t)};
  }

  static Status initialize(std::span<std::byte> backing);
  static Result<std::int64_t> parse(std::span<const std::byte> backing,
                                    bool publication_authorized);
};

}  // namespace pih
