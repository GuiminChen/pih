#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "pih/model/qwen3_semantic_paired_executor.h"

namespace pih {
namespace {

Sha256Digest digest(std::byte value) {
  Sha256Digest result;
  result.bytes.fill(value);
  return result;
}

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

Result<QwenSemanticOutcome> outcome(std::byte value, std::uint64_t device,
                                    std::uint64_t pinned) {
  QwenSemanticOutcomeRecorder recorder;
  const std::array payload{value, value};
  Status status = recorder.record_dispatch(payload);
  if (status.ok()) status = recorder.record_final_logits(payload);
  if (status.ok()) status = recorder.record_greedy_trajectory(payload);
  if (status.ok()) status = recorder.record_kv_state(payload);
  if (status.ok()) status = recorder.record_accepted_token_ledger(payload);
  if (status.ok()) status = recorder.observe_capacity(device, pinned);
  if (!status.ok()) return status;
  return recorder.seal();
}

class FakeDriver final : public QwenSemanticPairedExecutionDriver {
 public:
  Result<QwenSemanticInstrumentedRun> run_instrumented(
      std::span<const std::int64_t> tokens) override {
    calls.push_back({'i', tokens.data(), tokens.size()});
    if (!instrumented_status.ok()) return instrumented_status;
    auto semantic = outcome(instrumented_value, 120, 30);
    if (!semantic.ok()) return semantic.status();
    return QwenSemanticInstrumentedRun{
        {7, 5, 369, digest(std::byte{8}), digest(std::byte{9})},
        std::move(*semantic)};
  }

  Result<QwenSemanticOutcome> run_control(
      std::span<const std::int64_t> tokens) override {
    calls.push_back({'c', tokens.data(), tokens.size()});
    if (!control_status.ok()) return control_status;
    return outcome(control_value, 100, 10);
  }

  struct Call { char kind; const std::int64_t* data; std::size_t size; };
  std::vector<Call> calls;
  Status instrumented_status = Status::Ok();
  Status control_status = Status::Ok();
  std::byte instrumented_value{1};
  std::byte control_value{1};
};

TEST(QwenSemanticPairedExecutorTest, RunsSameFixtureAndSealsEqualSemantics) {
  std::vector<std::int64_t> tokens(4098, 17);
  FakeDriver driver;
  QwenSemanticPairedExecutor executor;

  auto receipt = executor.run(identity(), tokens, driver);

  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  ASSERT_EQ(driver.calls.size(), 2);
  EXPECT_EQ(driver.calls[0].kind, 'i');
  EXPECT_EQ(driver.calls[1].kind, 'c');
  EXPECT_EQ(driver.calls[0].data, tokens.data());
  EXPECT_EQ(driver.calls[1].data, tokens.data());
  EXPECT_EQ(driver.calls[0].size, tokens.size());
  EXPECT_EQ(driver.calls[1].size, tokens.size());
  EXPECT_EQ(executor.state(), QwenSemanticPairedExecutorState::kCompleted);
}

TEST(QwenSemanticPairedExecutorTest, DriftOrDriverFailurePoisonsWithoutRetry) {
  std::vector<std::int64_t> tokens(4098, 17);
  FakeDriver drifted;
  drifted.control_value = std::byte{2};
  QwenSemanticPairedExecutor executor;
  EXPECT_FALSE(executor.run(identity(), tokens, drifted).ok());
  EXPECT_EQ(executor.state(), QwenSemanticPairedExecutorState::kPoisoned);
  EXPECT_FALSE(executor.run(identity(), tokens, drifted).ok());

  FakeDriver failed;
  failed.instrumented_status = Status::Internal("injected");
  QwenSemanticPairedExecutor failed_executor;
  EXPECT_FALSE(failed_executor.run(identity(), tokens, failed).ok());
  ASSERT_EQ(failed.calls.size(), 1);
  EXPECT_EQ(failed.calls.front().kind, 'i');
}

TEST(QwenSemanticPairedExecutorTest, RejectsInvalidFixtureBeforeDriverAction) {
  FakeDriver driver;
  QwenSemanticPairedExecutor executor;
  std::vector<std::int64_t> short_tokens(4097, 1);
  auto result = executor.run(identity(), short_tokens, driver);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_TRUE(driver.calls.empty());
  EXPECT_EQ(executor.state(), QwenSemanticPairedExecutorState::kReady);
}

}  // namespace
}  // namespace pih
