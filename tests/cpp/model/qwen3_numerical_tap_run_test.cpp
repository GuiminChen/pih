#include <array>
#include <cstddef>

#include <gtest/gtest.h>

#include "pih/model/qwen3_numerical_tap_run.h"

namespace pih {
namespace {

QwenNumericalTapPlan small_plan(bool reverse = false) {
  std::array requests{
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLayerHidden, 0, 0, 1},
      QwenNumericalTapRequest{QwenNumericalTapPoint::kLogits, 28, 0, 1},
  };
  if (reverse) std::swap(requests[0], requests[1]);
  return QwenNumericalTapPlan::Create(requests, 1ULL << 20).value();
}

QwenNumericalTapReceipt receipt(std::size_t index, std::uint64_t generation,
                                std::uint64_t bytes, std::byte fill) {
  Sha256Digest digest;
  digest.bytes.fill(fill);
  return {index, generation, bytes, digest};
}

TEST(QwenNumericalTapRunTest, SealsCompleteOrderedZeroDropWriterRoot) {
  const auto plan = small_plan();
  auto run = QwenNumericalTapRun::Create(plan, 7);
  ASSERT_TRUE(run.ok());
  ASSERT_TRUE(run->record(receipt(1, 7, plan.captures()[1].size_bytes,
                                  std::byte{2}))
                  .ok());
  ASSERT_TRUE(run->record(receipt(0, 7, plan.captures()[0].size_bytes,
                                  std::byte{1}))
                  .ok());

  auto sealed = run->seal();

  ASSERT_TRUE(sealed.ok()) << sealed.status().message();
  EXPECT_EQ(sealed->run_generation, 7);
  EXPECT_EQ(sealed->capture_count, 2);
  EXPECT_EQ(sealed->plan_digest, plan.semantic_digest().value());
  EXPECT_NE(sealed->writer_root, Sha256Digest{});
  EXPECT_EQ(run->state(), QwenNumericalTapRunState::kSealed);
  EXPECT_FALSE(run->seal().ok());
}

TEST(QwenNumericalTapRunTest, MissingDuplicateOrDriftInvalidatesWholeRun) {
  const auto plan = small_plan();
  auto missing = QwenNumericalTapRun::Create(plan, 7).value();
  ASSERT_TRUE(missing.record(
                         receipt(0, 7, plan.captures()[0].size_bytes,
                                 std::byte{1}))
                  .ok());
  EXPECT_FALSE(missing.seal().ok());
  EXPECT_EQ(missing.state(), QwenNumericalTapRunState::kPoisoned);

  auto duplicate = QwenNumericalTapRun::Create(plan, 7).value();
  const auto first =
      receipt(0, 7, plan.captures()[0].size_bytes, std::byte{1});
  ASSERT_TRUE(duplicate.record(first).ok());
  EXPECT_FALSE(duplicate.record(first).ok());
  EXPECT_EQ(duplicate.state(), QwenNumericalTapRunState::kPoisoned);

  auto drift = QwenNumericalTapRun::Create(plan, 7).value();
  EXPECT_FALSE(drift.record(receipt(0, 8, plan.captures()[0].size_bytes,
                                    std::byte{1}))
                   .ok());
  EXPECT_EQ(drift.state(), QwenNumericalTapRunState::kPoisoned);
}

TEST(QwenNumericalTapRunTest, PlanAndReceiptOrderAreCryptographicallyBound) {
  const auto first = small_plan();
  const auto second = small_plan(true);
  ASSERT_NE(first.semantic_digest().value(), second.semantic_digest().value());

  auto run_a = QwenNumericalTapRun::Create(first, 7).value();
  ASSERT_TRUE(run_a.record(receipt(0, 7, first.captures()[0].size_bytes,
                                   std::byte{1}))
                  .ok());
  ASSERT_TRUE(run_a.record(receipt(1, 7, first.captures()[1].size_bytes,
                                   std::byte{2}))
                  .ok());
  auto root_a = run_a.seal().value().writer_root;

  auto run_b = QwenNumericalTapRun::Create(first, 7).value();
  ASSERT_TRUE(run_b.record(receipt(0, 7, first.captures()[0].size_bytes,
                                   std::byte{2}))
                  .ok());
  ASSERT_TRUE(run_b.record(receipt(1, 7, first.captures()[1].size_bytes,
                                   std::byte{1}))
                  .ok());
  EXPECT_NE(root_a, run_b.seal().value().writer_root);
}

}  // namespace
}  // namespace pih
