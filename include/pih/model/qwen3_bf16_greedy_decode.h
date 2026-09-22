#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

enum class QwenBf16GreedyDecodeState : std::uint8_t {
  kReady,
  kRunning,
  kCompleted,
  kPoisoned,
};

class QwenBf16GreedyStepDriver {
 public:
  virtual ~QwenBf16GreedyStepDriver() = default;
  virtual Result<std::int64_t> execute(std::span<const std::int64_t> tokens,
                                       std::uint64_t first_position) = 0;
};

class QwenBf16GreedyDecode final {
 public:
  static constexpr std::uint64_t kMaximumPositions = 40960;

  static Result<QwenBf16GreedyDecode> Create(
      std::span<const std::int64_t> prompt, std::uint64_t maximum_new_tokens,
      std::int64_t eos_token_id);

  Status run_next(QwenBf16GreedyStepDriver& driver);

  [[nodiscard]] QwenBf16GreedyDecodeState state() const noexcept {
    return state_;
  }
  [[nodiscard]] std::span<const std::int64_t> generated() const noexcept {
    return generated_;
  }
  [[nodiscard]] std::uint64_t next_position() const noexcept {
    return prompt_.size() + generated_.size();
  }

 private:
  QwenBf16GreedyDecode(std::vector<std::int64_t> prompt,
                       std::uint64_t maximum_new_tokens,
                       std::int64_t eos_token_id)
      : prompt_(std::move(prompt)),
        maximum_new_tokens_(maximum_new_tokens),
        eos_token_id_(eos_token_id) {
    generated_.reserve(maximum_new_tokens_);
  }

  std::vector<std::int64_t> prompt_;
  std::vector<std::int64_t> generated_;
  std::uint64_t maximum_new_tokens_;
  std::int64_t eos_token_id_;
  QwenBf16GreedyDecodeState state_ = QwenBf16GreedyDecodeState::kReady;
};

}  // namespace pih
