#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_command_buffer.h"
#include "pih/model/qwen3_bf16_tap_binding.h"
#include "pih/model/qwen3_bf16_tap_suite_plan.h"

namespace pih {
namespace {

QwenBf16CommandBuffer commands() {
  const Qwen3Config config{1024, 3072, 28, 16, 8, 128, 151936, 40960,
                           1'000'000.0, 0.000001, 151643, 151645};
  auto schedule = QwenBf16ExecutionSchedule::Create(config).value();
  auto weights = QwenBf16WeightBindingPlan::Create(schedule).value();
  return QwenBf16CommandBuffer::Create(schedule, weights).value();
}

TEST(QwenBf16TapSuitePlanTest, FreezesFiveExecutableFixturePartitions) {
  auto suite = QwenBf16TapSuitePlan::Create();

  ASSERT_TRUE(suite.ok()) << suite.status().message();
  ASSERT_EQ(suite->size(), 5);
  EXPECT_EQ((*suite)[0].first_position, 0);
  EXPECT_EQ((*suite)[0].taps.captures().size(), 145);
  constexpr std::array positions{2U, 17U, 129U, 4097U};
  std::size_t total = (*suite)[0].taps.captures().size();
  const auto frozen_commands = commands();
  for (std::size_t fixture_index = 1; fixture_index < suite->size();
       ++fixture_index) {
    const auto& fixture = (*suite)[fixture_index];
    EXPECT_EQ(fixture.first_position, positions[fixture_index - 1]);
    ASSERT_EQ(fixture.taps.captures().size(), 56);
    for (const auto& capture : fixture.taps.captures()) {
      EXPECT_EQ(capture.request.position, fixture.first_position);
      EXPECT_TRUE(capture.request.point == QwenNumericalTapPoint::kKvKey ||
                  capture.request.point == QwenNumericalTapPoint::kKvValue);
    }
    auto bindings =
        QwenBf16TapBindingPlan::Create(fixture.taps, frozen_commands);
    EXPECT_TRUE(bindings.ok()) << bindings.status().message();
    total += fixture.taps.captures().size();
  }
  EXPECT_EQ(total, QwenBf16TapSuitePlan::kCaptureCount);
  EXPECT_TRUE(validate_qwen_bf16_tap_suite_coverage(*suite).ok());
  EXPECT_TRUE(suite->semantic_digest().ok());
}

TEST(QwenBf16TapSuitePlanTest, ActivationFixtureIsSingleTokenCompatible) {
  auto suite = QwenBf16TapSuitePlan::Create().value();
  for (const auto& capture : suite[0].taps.captures()) {
    EXPECT_EQ(capture.request.position, 0);
    EXPECT_EQ(capture.request.rows, 1);
    EXPECT_NE(capture.request.point, QwenNumericalTapPoint::kKvKey);
    EXPECT_NE(capture.request.point, QwenNumericalTapPoint::kKvValue);
  }
  EXPECT_EQ(suite[0].taps.captures().back().request.point,
            QwenNumericalTapPoint::kLogits);
}

}  // namespace
}  // namespace pih
