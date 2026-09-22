#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

class QwenBf16MlpArenaLayout final {
 public:
  static constexpr std::uint64_t kAlignment = 256;
  static constexpr std::uint64_t kIntermediateFeatures = 3072;
  static constexpr std::uint64_t kElementBytes = 2;

  static Result<QwenBf16MlpArenaLayout> Create(std::uint64_t tokens);

  [[nodiscard]] std::uint64_t tokens() const noexcept { return tokens_; }
  [[nodiscard]] std::uint64_t gate_offset_bytes() const noexcept { return 0; }
  [[nodiscard]] std::uint64_t up_offset_bytes() const noexcept {
    return up_offset_bytes_;
  }
  [[nodiscard]] std::uint64_t intermediate_bytes() const noexcept {
    return intermediate_bytes_;
  }
  [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
    return arena_bytes_;
  }

 private:
  QwenBf16MlpArenaLayout(std::uint64_t tokens,
                         std::uint64_t intermediate_bytes,
                         std::uint64_t up_offset_bytes,
                         std::uint64_t arena_bytes)
      : tokens_(tokens),
        intermediate_bytes_(intermediate_bytes),
        up_offset_bytes_(up_offset_bytes),
        arena_bytes_(arena_bytes) {}

  std::uint64_t tokens_ = 0;
  std::uint64_t intermediate_bytes_ = 0;
  std::uint64_t up_offset_bytes_ = 0;
  std::uint64_t arena_bytes_ = 0;
};

}  // namespace pih
