#pragma once

#include <cstdint>

#include "pih/core/result.h"
#include "pih/model/qwen3_bf16_linear_shape.h"

namespace pih {

enum class QwenBf16LinearShapeRole : std::uint8_t {
  kPrefill,
  kTailPrefill,
  kDecode,
};

class QwenBf16LinearShapeSelector final {
 public:
  static Result<QwenBf16LinearShapeSelector> Create(
      std::uint64_t prefill_tokens, std::uint64_t decode_tokens);
  static Result<QwenBf16LinearShapeSelector> Create(
      std::uint64_t prefill_tokens, std::uint64_t tail_prefill_tokens,
      std::uint64_t decode_tokens);
  Result<QwenBf16LinearShapeRole> select(std::uint64_t rows) const;

 private:
  QwenBf16LinearShapeSelector(std::uint64_t prefill_tokens,
                              std::uint64_t tail_prefill_tokens,
                              std::uint64_t decode_tokens)
      : prefill_tokens_(prefill_tokens),
        tail_prefill_tokens_(tail_prefill_tokens),
        decode_tokens_(decode_tokens) {}
  std::uint64_t prefill_tokens_;
  std::uint64_t tail_prefill_tokens_;
  std::uint64_t decode_tokens_;
};

}  // namespace pih
