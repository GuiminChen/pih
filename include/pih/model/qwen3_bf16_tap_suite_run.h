#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include "pih/model/qwen3_bf16_tap_suite_plan.h"
#include "pih/model/qwen3_numerical_tap_run.h"

namespace pih {

enum class QwenBf16TapSuiteRunState : std::uint8_t {
  kCollecting = 0,
  kSealed,
  kPoisoned,
};

struct QwenBf16TapSuiteRunReceipt final {
  std::uint64_t suite_generation;
  std::size_t fixture_count;
  std::size_t capture_count;
  Sha256Digest suite_plan_digest;
  Sha256Digest writer_root;
};

class QwenBf16TapSuiteRun final {
 public:
  static Result<QwenBf16TapSuiteRun> Create(
      const QwenBf16TapSuitePlan& suite,
      std::uint64_t suite_generation,
      std::uint64_t first_fixture_generation);

  Status record(std::size_t fixture_index,
                const QwenNumericalTapRunReceipt& receipt);
  Result<QwenBf16TapSuiteRunReceipt> seal();

  [[nodiscard]] QwenBf16TapSuiteRunState state() const noexcept {
    return state_;
  }

 private:
  QwenBf16TapSuiteRun(
      std::uint64_t suite_generation,
      std::uint64_t first_fixture_generation,
      Sha256Digest suite_plan_digest,
      std::array<Sha256Digest, QwenBf16TapSuitePlan::kFixtureCount>
          fixture_plan_digests,
      std::array<std::size_t, QwenBf16TapSuitePlan::kFixtureCount>
          fixture_capture_counts)
      : suite_generation_(suite_generation),
        first_fixture_generation_(first_fixture_generation),
        suite_plan_digest_(suite_plan_digest),
        fixture_plan_digests_(fixture_plan_digests),
        fixture_capture_counts_(fixture_capture_counts) {}
  Status poison(const char* message);

  std::uint64_t suite_generation_;
  std::uint64_t first_fixture_generation_;
  Sha256Digest suite_plan_digest_;
  std::array<Sha256Digest, QwenBf16TapSuitePlan::kFixtureCount>
      fixture_plan_digests_{};
  std::array<std::size_t, QwenBf16TapSuitePlan::kFixtureCount>
      fixture_capture_counts_{};
  std::array<std::optional<QwenNumericalTapRunReceipt>,
             QwenBf16TapSuitePlan::kFixtureCount>
      receipts_{};
  std::size_t recorded_count_ = 0;
  QwenBf16TapSuiteRunState state_ =
      QwenBf16TapSuiteRunState::kCollecting;
};

}  // namespace pih
