#include "pih/model/deepseek_endpoint_stage_backend.h"

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
class Operations final : public DeepSeekEndpointSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status embedding(DeepSeekEmbeddingLaunch) override {
    calls.push_back("embedding"); return Status::Ok();
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
  Status sample(DeepSeekArgmaxLaunch) override { return Status::Ok(); }
  Status copy_token_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 7; return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  std::vector<std::string> calls;
};
struct Sequence final {
  explicit Sequence(std::uint32_t id, std::uintptr_t error)
      : transaction(DeepSeekAttentionSequenceTransaction::Create(
            id, 0, 0, banks, ratio4, ratio128).value()),
        executor(DeepSeekEndpointSequenceExecutor::Create(
                     operations, &host_error, &host_sampled).value()) {
    embedding.error_flag_u32 = error;
    head.hc.error_flag_u32 = error;
    head.rms.error_flag = error;
    head.lm.error_flag_u32 = error;
    head.sample.error_flag_u32 = error;
  }
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 61, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction;
  Operations operations;
  std::uint32_t host_error = 0;
  std::uint32_t host_sampled = 0;
  DeepSeekEndpointSequenceExecutor executor;
  DeepSeekEmbeddingLaunch embedding{1, 2, 3, 0, 11, 1, 129280, 4096, 4};
  DeepSeekHeadSequenceSubmission head{
      {20, 21, 22, 23, 24, 0, 11, 1, 4096, 4, 1.0e-6F, 1.0e-6F},
      {24, 25, 26, 0, 11, 1, 4096, 1.0e-6F},
      {26, 27, 28, 0, 11, 1, 129280, 4096},
      {28, 29, 0, 11, 129280}};
};
class Provider final : public DeepSeekEndpointStageWorkProvider {
 public:
  Result<std::span<const DeepSeekEndpointStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return std::span<const DeepSeekEndpointStageSequenceWork>(work);
  }
  std::vector<DeepSeekEndpointStageSequenceWork> work;
};
class Fallback final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand&, const DeepSeekPipelinePlanDescriptor&) override {
    ++launches; return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return DeepSeekStageComputeStatus::kSuccess;
  }
  std::uint32_t launches = 0;
};
DeepSeekPipelinePlanDescriptor plan() {
  return {4, 8, DeepSeekPlanPhase::kDecode, 2, 2};
}

TEST(DeepSeekEndpointStageBackendTest, LaunchesEmbeddingForEveryPackedSequence) {
  Sequence first(1, 101), second(2, 102);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.embedding, first.head},
                   {&second.executor, &second.transaction, second.embedding, second.head}};
  Fallback fallback;
  auto backend = DeepSeekEndpointStageOperatorBackend::Create(fallback, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kEmbedding, 0}, plan()).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "embedding", "d2h"}));
  EXPECT_EQ(second.operations.calls, first.operations.calls);
  EXPECT_EQ(fallback.launches, 0U);
}

TEST(DeepSeekEndpointStageBackendTest, LaunchesOfficialHeadChainPerSequence) {
  Sequence first(1, 101), second(2, 102);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.embedding, first.head},
                   {&second.executor, &second.transaction, second.embedding, second.head}};
  Fallback fallback;
  auto backend = DeepSeekEndpointStageOperatorBackend::Create(fallback, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kHead, 0}, plan()).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "hc", "rms", "lm", "d2h"}));
}

TEST(DeepSeekEndpointStageBackendTest, RejectsDuplicatePackedTransactionBeforeLaunch) {
  Sequence first(1, 101);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.embedding, first.head},
                   {&first.executor, &first.transaction, first.embedding, first.head}};
  Fallback fallback;
  auto backend = DeepSeekEndpointStageOperatorBackend::Create(fallback, provider).value();
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kEmbedding, 0}, plan()).ok());
  EXPECT_TRUE(first.operations.calls.empty());
}

}}  // namespace pih::<anonymous>
