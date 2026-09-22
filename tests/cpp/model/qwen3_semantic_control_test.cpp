#include <cstddef>
#include <string>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_control.h"

namespace pih {
namespace {

QwenNumericalRunIdentity identity() {
  const std::string json =
      "{\"schema\":\"pih.qwen_bf16_numerical_run.v1\","
      "\"model_sha256\":\"" + std::string(64, 'a') +
      "\",\"fixture_sha256\":\"" + std::string(64, 'b') +
      "\",\"tolerance_sha256\":\"" + std::string(64, 'c') +
      "\",\"kernel_bundle_sha256\":\"" + std::string(64, 'd') +
      "\",\"build_sha256\":\"" + std::string(64, 'e') +
      "\",\"environment_sha256\":\"" + std::string(64, 'f') +
      "\",\"target_gpu\":\"RTX_4090_D\",\"target_sm\":\"sm_89\","
      "\"run_generation\":7}";
  return QwenNumericalRunIdentity::Parse(json).value();
}

Sha256Digest digest(std::byte value) {
  Sha256Digest result;
  result.bytes.fill(value);
  return result;
}

QwenSemanticOutcome outcome(std::uint64_t device_peak,
                            std::uint64_t pinned_peak) {
  return {digest(std::byte{1}), digest(std::byte{2}), digest(std::byte{3}),
          digest(std::byte{4}), digest(std::byte{5}), device_peak, pinned_peak};
}

QwenBf16TapSuiteRunReceipt tap_receipt(std::uint64_t generation = 7) {
  return {generation, 5, 369, digest(std::byte{6}), digest(std::byte{7})};
}

TEST(QwenSemanticControlTest, BindsEqualSemanticsAndSeparateCapacityPeaks) {
  const auto instrumented = outcome(12'000, 3'000);
  const auto control = outcome(10'000, 500);

  auto receipt = QwenSemanticControlReceipt::Create(
      identity(), tap_receipt(), instrumented, control);

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(receipt->instrumented_device_peak_bytes(), 12'000);
  EXPECT_EQ(receipt->control_device_peak_bytes(), 10'000);
  EXPECT_EQ(receipt->instrumented_pinned_peak_bytes(), 3'000);
  EXPECT_EQ(receipt->control_pinned_peak_bytes(), 500);
  EXPECT_EQ(receipt->tap_suite().capture_count, 369);
  EXPECT_EQ(receipt->dispatch_root(), instrumented.dispatch_root);
  EXPECT_EQ(receipt->final_logits_root(), instrumented.final_logits_root);
  EXPECT_EQ(receipt->greedy_trajectory_root(),
            instrumented.greedy_trajectory_root);
  EXPECT_EQ(receipt->kv_state_root(), instrumented.kv_state_root);
  EXPECT_EQ(receipt->accepted_token_ledger_root(),
            instrumented.accepted_token_ledger_root);
  EXPECT_NE(receipt->semantic_digest(), Sha256Digest{});
}

TEST(QwenSemanticControlTest, RejectsEverySemanticDriftAndRunSplicing) {
  auto control = outcome(10'000, 500);
  control.final_logits_root = digest(std::byte{9});
  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), tap_receipt(), outcome(12'000, 3'000), control)
                   .ok());

  control = outcome(10'000, 500);
  control.dispatch_root = digest(std::byte{9});
  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), tap_receipt(), outcome(12'000, 3'000), control)
                   .ok());

  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), tap_receipt(8), outcome(12'000, 3'000),
                   outcome(10'000, 500))
                   .ok());
}

TEST(QwenSemanticControlTest, RejectsMissingTapOrCapacityEvidence) {
  auto empty_tap = tap_receipt();
  empty_tap.fixture_count = 0;
  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), empty_tap, outcome(12'000, 3'000),
                   outcome(10'000, 500))
                   .ok());

  auto incomplete_suite = tap_receipt();
  incomplete_suite.fixture_count = 4;
  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), incomplete_suite, outcome(12'000, 3'000),
                   outcome(10'000, 500))
                   .ok());

  auto no_capacity = outcome(0, 0);
  EXPECT_FALSE(QwenSemanticControlReceipt::Create(
                   identity(), tap_receipt(), no_capacity,
                   outcome(10'000, 500))
                   .ok());
}

}  // namespace
}  // namespace pih
