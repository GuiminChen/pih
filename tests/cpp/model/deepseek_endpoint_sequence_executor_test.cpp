#include "pih/model/deepseek_endpoint_sequence_executor.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Operations final : public DeepSeekEndpointSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t device, std::uintptr_t) override {
    calls.push_back("zero"); devices.push_back(device); return Status::Ok();
  }
  Status embedding(DeepSeekEmbeddingLaunch value) override {
    calls.push_back("embedding"); embedding_launch = value; return Status::Ok();
  }
  Status hc_head(DeepSeekHcHeadLaunch value) override {
    calls.push_back("hc_head"); hc_launch = value; return Status::Ok();
  }
  Status rms_norm(DeepSeekRmsNormLaunch value) override {
    calls.push_back("rms"); rms_launch = value; return Status::Ok();
  }
  Status lm_head(DeepSeekLmHeadLaunch value) override {
    calls.push_back("lm_head"); lm_launch = value; return Status::Ok();
  }
  Status sample(DeepSeekArgmaxLaunch value) override {
    calls.push_back("sample"); sample_launch = value; return Status::Ok();
  }
  Status stochastic_sample(DeepSeekStochasticSampleLaunch value) override {
    calls.push_back("stochastic_sample"); stochastic_launch = value;
    return Status::Ok();
  }
  Status copy_token_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 42; calls.push_back("token_d2h"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  Status copy_logprob_d2h_async(float* host, std::uintptr_t,
                                std::uintptr_t) override {
    *host = -0.25F; calls.push_back("logprob_d2h"); return Status::Ok();
  }
  Status copy_rng_d2h_async(std::uint32_t* host, std::uintptr_t,
                            std::uintptr_t) override {
    *host = 0x12345678U; calls.push_back("rng_d2h"); return Status::Ok();
  }
  Status copy_top_ids_d2h_async(std::uint32_t* host, std::uintptr_t,
                                std::uint32_t count,
                                std::uintptr_t) override {
    for (std::uint32_t i = 0; i < count; ++i) host[i] = 100 + i;
    calls.push_back("top_ids_d2h"); return Status::Ok();
  }
  Status copy_top_logprobs_d2h_async(float* host, std::uintptr_t,
                                     std::uint32_t count,
                                     std::uintptr_t) override {
    for (std::uint32_t i = 0; i < count; ++i) host[i] = -0.1F * (i + 1);
    calls.push_back("top_logprobs_d2h"); return Status::Ok();
  }
  std::vector<std::string> calls;
  std::vector<std::uintptr_t> devices;
  DeepSeekEmbeddingLaunch embedding_launch;
  DeepSeekHcHeadLaunch hc_launch;
  DeepSeekRmsNormLaunch rms_launch;
  DeepSeekLmHeadLaunch lm_launch;
  DeepSeekArgmaxLaunch sample_launch;
  DeepSeekStochasticSampleLaunch stochastic_launch;
};

struct Fixture final {
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 51, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          5, 0, 0, banks, ratio4, ratio128).value();
  Operations operations;
  std::uint32_t host_error = 9;
  std::uint32_t host_sampled = 0;
  float host_logprob = 0.0F;
  std::uint32_t host_rng = 0;
  std::uint32_t host_top_ids[20]{};
  float host_top_logprobs[20]{};
  DeepSeekEndpointSequenceExecutor executor =
      DeepSeekEndpointSequenceExecutor::Create(
          operations, &host_error, &host_sampled, &host_logprob,
          &host_rng, host_top_ids, host_top_logprobs).value();
};

DeepSeekEmbeddingLaunch embedding() {
  return {1, 2, 3, 10, 11, 2, 129280, 4096, 4};
}

DeepSeekHeadSequenceSubmission head() {
  return {{20, 21, 22, 23, 24, 10, 11, 1, 4096, 4, 1.0e-6F, 1.0e-6F},
          {24, 25, 26, 10, 11, 1, 4096, 1.0e-6F},
          {26, 27, 28, 10, 11, 1, 129280, 4096},
          {28, 29, 10, 11, 129280, 30}};
}

DeepSeekHeadSequenceSubmission stochastic_head() {
  auto value = head();
  value.sample = {};
  value.stochastic_sample = DeepSeekStochasticSampleLaunch{
      28, 29, 30, 31, 32, 33, 10, 11,
      129280, 129280, 0.8F, 0.9F, 64, 7, 0, 34, 35, 2};
  return value;
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     EmbeddingClaimsAndCopiesItsSequencePrivateErrorChannel) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch_embedding(embedding(),
                                                 fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "embedding", "d2h"}));
  EXPECT_EQ(fixture.host_error, 0U);
  EXPECT_EQ(fixture.operations.embedding_launch.error_flag_u32, 10U);
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     HeadQueuesOfficialReduceNormAndLogitsOrder) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch_head(head(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "hc_head", "rms", "lm_head",
                                      "sample", "token_d2h", "logprob_d2h",
                                      "d2h"}));
  EXPECT_EQ(fixture.host_sampled, 42U);
  EXPECT_FLOAT_EQ(fixture.host_logprob, -0.25F);
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     GreedyHeadCopiesRankedLogprobsBeforeError) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto greedy = head();
  greedy.sample.top_logprobs_ids_u32 = 34;
  greedy.sample.top_logprobs_f32 = 35;
  greedy.sample.top_logprobs_count = 2;
  ASSERT_TRUE(fixture.executor.launch_head(greedy, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "hc_head", "rms", "lm_head",
                                      "sample", "token_d2h", "logprob_d2h",
                                      "top_ids_d2h", "top_logprobs_d2h",
                                      "d2h"}));
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     EmbeddingThenHeadReuseTheSameChannelWithoutClearingErrors) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch_embedding(embedding(),
                                                 fixture.transaction).ok());
  ASSERT_TRUE(fixture.executor.launch_head(head(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.devices.size(), 1U);
  EXPECT_EQ(fixture.operations.devices[0], 10U);
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     StochasticHeadCopiesAtomicSamplingReceiptBeforeError) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch_head(stochastic_head(),
                                            fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "hc_head", "rms", "lm_head",
                                      "stochastic_sample", "token_d2h",
                                      "logprob_d2h", "top_ids_d2h",
                                      "top_logprobs_d2h", "rng_d2h", "d2h"}));
  EXPECT_EQ(fixture.host_sampled, 42U);
  EXPECT_FLOAT_EQ(fixture.host_logprob, -0.25F);
  EXPECT_EQ(fixture.host_rng, 0x12345678U);
  EXPECT_EQ(fixture.host_top_ids[0], 100U);
  EXPECT_EQ(fixture.host_top_ids[1], 101U);
  EXPECT_FLOAT_EQ(fixture.host_top_logprobs[0], -0.1F);
  EXPECT_FLOAT_EQ(fixture.host_top_logprobs[1], -0.2F);
}

TEST(DeepSeekEndpointSequenceExecutorTest,
     RejectsMismatchedHeadPointerFlowBeforeClaimingTheChannel) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto invalid = head();
  invalid.rms.input_bf16 = 99;
  EXPECT_FALSE(fixture.executor.launch_head(invalid, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

}}  // namespace pih::<anonymous>
