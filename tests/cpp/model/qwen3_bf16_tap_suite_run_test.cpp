#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_bf16_tap_suite_run.h"

namespace pih {
namespace {

QwenNumericalTapRunReceipt fixture_receipt(
    const QwenBf16TapSuitePlan& suite, std::size_t index,
    std::uint64_t first_generation) {
  auto digest = suite[index].taps.semantic_digest().value();
  auto writer = sha256(std::as_bytes(std::span(&index, 1))).value();
  return {first_generation + index, suite[index].taps.captures().size(),
          digest, writer};
}

TEST(QwenBf16TapSuiteRunTest, SealsAllFiveFixtureRootsInCanonicalOrder) {
  auto suite = QwenBf16TapSuitePlan::Create().value();
  auto run = QwenBf16TapSuiteRun::Create(suite, 7, 100).value();
  for (std::size_t index = 0; index < suite.size(); ++index) {
    ASSERT_TRUE(run.record(index, fixture_receipt(suite, index, 100)).ok());
  }

  auto receipt = run.seal();

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->suite_generation, 7);
  EXPECT_EQ(receipt->fixture_count, 5);
  EXPECT_EQ(receipt->capture_count, 369);
  EXPECT_EQ(run.state(), QwenBf16TapSuiteRunState::kSealed);
}

TEST(QwenBf16TapSuiteRunTest, MissingDuplicateOrCrossGenerationPoisons) {
  auto suite = QwenBf16TapSuitePlan::Create().value();
  auto missing = QwenBf16TapSuiteRun::Create(suite, 7, 100).value();
  ASSERT_TRUE(missing.record(0, fixture_receipt(suite, 0, 100)).ok());
  EXPECT_FALSE(missing.seal().ok());
  EXPECT_EQ(missing.state(), QwenBf16TapSuiteRunState::kPoisoned);

  auto duplicate = QwenBf16TapSuiteRun::Create(suite, 7, 100).value();
  const auto first = fixture_receipt(suite, 0, 100);
  ASSERT_TRUE(duplicate.record(0, first).ok());
  EXPECT_FALSE(duplicate.record(0, first).ok());
  EXPECT_EQ(duplicate.state(), QwenBf16TapSuiteRunState::kPoisoned);

  auto cross_generation =
      QwenBf16TapSuiteRun::Create(suite, 7, 100).value();
  auto wrong = fixture_receipt(suite, 1, 100);
  --wrong.run_generation;
  EXPECT_FALSE(cross_generation.record(1, wrong).ok());
  EXPECT_EQ(cross_generation.state(), QwenBf16TapSuiteRunState::kPoisoned);
}

}  // namespace
}  // namespace pih
