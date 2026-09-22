#include "pih/model/deepseek_dspark_head_executor.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {
class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
class Operations final : public DeepSeekDsparkHeadOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status hc_head(DeepSeekHcHeadLaunch) override {
    calls.push_back("hc"); return Status::Ok();
  }
  Status rms_norm(DeepSeekRmsNormLaunch) override {
    calls.push_back("rms"); return Status::Ok();
  }
  Status lm_head(DeepSeekLmHeadLaunch) override {
    calls.push_back("lm"); return Status::Ok();
  }
  Status markov(DeepSeekDsparkMarkovLaunch) override {
    calls.push_back("markov"); return Status::Ok();
  }
  Status argmax(DeepSeekArgmaxLaunch) override {
    calls.push_back("argmax"); return Status::Ok();
  }
  Status confidence(DeepSeekDsparkConfidenceLaunch) override {
    calls.push_back("confidence"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  std::vector<std::string> calls;
};
struct Fixture final {
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 101, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, banks, ratio4, ratio128).value();
  Operations operations;
  std::uint32_t host_error = 7;
  DeepSeekDsparkHeadExecutor executor =
      DeepSeekDsparkHeadExecutor::Create(operations, &host_error).value();
};
DeepSeekDsparkHeadSubmission submission() {
  DeepSeekDsparkHeadSubmission value;
  constexpr std::uintptr_t error = 90, stream = 11;
  constexpr std::uintptr_t raw = 0x1000000, biased = 0x2000000;
  constexpr std::uintptr_t embeds = 0x3000000, tokens = 0x4000000;
  value.hc = {0x5000000, 12, 13, 14, 0x6000000, error, stream,
              5, 4096, 4, 1.0e-6F, 1.0e-6F};
  value.rms = {0x6000000, 15, 0x7000000, error, stream,
               5, 4096, 1.0e-6F};
  value.lm = {0x7000000, 16, raw, error, stream, 5, 129280, 4096};
  for (std::size_t index = 0; index < value.kBlockSize; ++index) {
    value.markov[index] = {
        tokens + index * sizeof(std::uint32_t), 2, 3,
        raw + index * 129280ULL * sizeof(float),
        embeds + index * 256ULL * 2ULL,
        biased + index * 129280ULL * sizeof(float), error, stream,
        1, 129280, 256};
    value.argmax[index] = {
        value.markov[index].biased_logits_f32,
        tokens + (index + 1) * sizeof(std::uint32_t), error, stream, 129280};
  }
  value.confidence = {5, embeds, 6, 7, error, stream, 5, 4096, 256};
  return value;
}

TEST(DeepSeekDsparkHeadExecutorTest, QueuesFiveCausalDraftStepsThenConfidence) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch(submission(), fixture.transaction).ok());
  std::vector<std::string> expected{"zero", "hc", "rms", "lm"};
  for (int i = 0; i < 5; ++i) {
    expected.push_back("markov"); expected.push_back("argmax");
  }
  expected.push_back("confidence"); expected.push_back("d2h");
  EXPECT_EQ(fixture.operations.calls, expected);
  EXPECT_EQ(fixture.host_error, 0U);
}
TEST(DeepSeekDsparkHeadExecutorTest, RejectsBrokenTokenChainBeforeSubmission) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto invalid = submission();
  invalid.markov[3].token_ids_u32 += 4;
  EXPECT_FALSE(fixture.executor.launch(invalid, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}
TEST(DeepSeekDsparkHeadExecutorTest, ReusesEarlierTransactionErrorChannel) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto claimed = fixture.transaction.claim_external_error_channel(
      &fixture.host_error, 90);
  ASSERT_TRUE(claimed.ok());
  ASSERT_TRUE(fixture.executor.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls.front(), "hc");
}
}}  // namespace pih::<anonymous>
