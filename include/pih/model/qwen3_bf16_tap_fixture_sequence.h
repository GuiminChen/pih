#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/qwen3_bf16_tap_suite_plan.h"

namespace pih {

struct QwenBf16TapFixtureInput final {
  QwenBf16TapFixtureKind kind;
  std::uint32_t first_position;
  std::uint32_t capture_position;
  std::span<const std::int64_t> tokens;
};

class QwenBf16TapFixtureSequence final {
 public:
  static constexpr std::size_t kRequiredTokenCount = 4098;
  static constexpr std::int64_t kVocabularySize = 151936;

  static Result<QwenBf16TapFixtureSequence> Create(
      std::span<const std::int64_t> tokens,
      const QwenBf16TapSuitePlan& suite);

  [[nodiscard]] std::size_t size() const noexcept {
    return QwenBf16TapSuitePlan::kFixtureCount;
  }
  [[nodiscard]] QwenBf16TapFixtureInput operator[](
      std::size_t index) const noexcept;
  [[nodiscard]] std::span<const std::int64_t> tokens() const noexcept {
    return tokens_;
  }
  Result<Sha256Digest> semantic_digest() const;

 private:
  QwenBf16TapFixtureSequence(std::vector<std::int64_t> tokens,
                            Sha256Digest suite_digest)
      : tokens_(std::move(tokens)), suite_digest_(suite_digest) {}

  std::vector<std::int64_t> tokens_;
  Sha256Digest suite_digest_;
};

}  // namespace pih
