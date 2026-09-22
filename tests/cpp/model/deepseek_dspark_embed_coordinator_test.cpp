#include "pih/model/deepseek_dspark_embed_coordinator.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {
class Fixed final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
class Operations final : public DeepSeekDsparkEmbedOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    calls.push_back("quant"); return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override {
    calls.push_back("main_proj"); return Status::Ok();
  }
  Status rms(DeepSeekRmsNormLaunch) override {
    calls.push_back("main_norm"); return Status::Ok();
  }
  Status draft_init(DeepSeekDsparkDraftInitLaunch) override {
    calls.push_back("draft_init"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  std::vector<std::string> calls;
};
struct Fixture final {
  Fixed fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 111, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, banks, ratio4, ratio128).value();
  Operations operations;
  std::uint32_t host_error = 4;
  DeepSeekDsparkEmbedCoordinator coordinator =
      DeepSeekDsparkEmbedCoordinator::Create(operations, &host_error).value();
};
DeepSeekDsparkEmbedSubmission submission() {
  constexpr std::uintptr_t error = 90, stream = 11;
  return {{1, 2, 3, error, stream, 2, 12288},
          {2, 3, 4, 5, 6, error, stream, 2, 4096, 12288},
          {6, 7, 8, error, stream, 2, 4096, 1.0e-6F},
          {9, 10, 11, 12, error, stream, 2, 17, 5, 129280, 4096, 4}};
}
TEST(DeepSeekDsparkEmbedCoordinatorTest,
     QueuesOfficialMainProjectionNormAndDraftInitialization) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "quant", "main_proj",
                                      "main_norm", "draft_init", "d2h"}));
}
TEST(DeepSeekDsparkEmbedCoordinatorTest,
     PrefillQueuesOnlyMainProjectionAndNormalization) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto prefill = submission();
  prefill.main_quant.token_count = 17;
  prefill.main_proj.m = 17;
  prefill.main_norm.rows = 17;
  prefill.draft_init = {};
  ASSERT_TRUE(fixture.coordinator.launch_prefill_state(
      prefill, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "quant", "main_proj",
                                      "main_norm", "d2h"}));
}
TEST(DeepSeekDsparkEmbedCoordinatorTest,
     RejectsBrokenMainProjectionFlowBeforeSubmission) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto invalid = submission();
  invalid.main_proj.k = 8192;
  EXPECT_FALSE(fixture.coordinator.launch(invalid, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}
TEST(DeepSeekDsparkEmbedCoordinatorTest,
     ReusesTransactionExternalErrorChannel) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.transaction.claim_external_error_channel(
      &fixture.host_error, 90).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls.front(), "quant");
}
}}  // namespace pih::<anonymous>
