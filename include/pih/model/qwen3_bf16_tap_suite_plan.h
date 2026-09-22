#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pih/model/qwen3_numerical_tap_plan.h"

namespace pih {

enum class QwenBf16TapFixtureKind : std::uint8_t {
  kPositionZeroActivations = 0,
  kKvPosition2,
  kKvPosition17,
  kKvPosition129,
  kKvPosition4097,
};

struct QwenBf16TapFixturePlan final {
  QwenBf16TapFixtureKind kind;
  std::uint32_t first_position;
  QwenNumericalTapPlan taps;
};

class QwenBf16TapSuitePlan final {
 public:
  static constexpr std::size_t kFixtureCount = 5;
  static constexpr std::size_t kCaptureCount = 369;

  static Result<QwenBf16TapSuitePlan> Create();

  [[nodiscard]] std::size_t size() const noexcept { return fixtures_.size(); }
  [[nodiscard]] const QwenBf16TapFixturePlan& operator[](
      std::size_t index) const noexcept {
    return fixtures_[index];
  }
  [[nodiscard]] auto begin() const noexcept { return fixtures_.begin(); }
  [[nodiscard]] auto end() const noexcept { return fixtures_.end(); }
  Result<Sha256Digest> semantic_digest() const;

 private:
  explicit QwenBf16TapSuitePlan(
      std::vector<QwenBf16TapFixturePlan> fixtures)
      : fixtures_(std::move(fixtures)) {}

  std::vector<QwenBf16TapFixturePlan> fixtures_;
};

Status validate_qwen_bf16_tap_suite_coverage(
    const QwenBf16TapSuitePlan& suite);

}  // namespace pih
