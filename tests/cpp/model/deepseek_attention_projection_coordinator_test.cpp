#include "pih/model/deepseek_attention_projection_coordinator.h"

#include <gtest/gtest.h>

#include <algorithm>
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
class Operations final : public DeepSeekAttentionProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    calls.push_back("quant"); return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override {
    calls.push_back("gemm"); return Status::Ok();
  }
  Status rms(DeepSeekRmsNormLaunch) override {
    calls.push_back("rms"); return Status::Ok();
  }
  Status head_rms(DeepSeekHeadRmsLaunch) override {
    calls.push_back("head_rms"); return Status::Ok();
  }
  Status rotary(DeepSeekRotaryLaunch value) override {
    calls.push_back(value.inverse ? "inverse_rope" : "rope");
    return Status::Ok();
  }
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch) override {
    calls.push_back("kv_simulate"); return Status::Ok();
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
      {0x1000, 256}, {0x2000, 256}, 71, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, banks, ratio4, ratio128).value();
  Operations operations;
  std::uint32_t host_error = 4;
  DeepSeekAttentionProjectionCoordinator coordinator =
      DeepSeekAttentionProjectionCoordinator::Create(operations, &host_error)
          .value();
};

DeepSeekAttentionProjectionSubmission submission() {
  constexpr std::uintptr_t error = 90, stream = 11;
  return {
      {1, 2, 3, error, stream, 2, 4096},
      {2, 3, 10, 11, 12, error, stream, 2, 1024, 4096},
      {12, 13, 14, error, stream, 2, 1024, 1.0e-6F},
      {14, 15, 16, error, stream, 2, 1024},
      {15, 16, 17, 18, 19, error, stream, 2, 32768, 1024},
      {19, 19, error, stream, 2, 64, 512, 1.0e-6F},
      {19, 20, error, stream, 2, 64, 512, 64, false, 21, 104},
      {2, 3, 21, 22, 23, error, stream, 2, 512, 4096},
      {23, 24, 23, error, stream, 2, 512, 1.0e-6F},
      {23, 20, error, stream, 2, 1, 512, 64, false, 21, 104},
      {23, error, stream, 2, 512, 448, 64}};
}

TEST(DeepSeekAttentionProjectionCoordinatorTest,
     QueuesOfficialQueryAndKvProjectionOrder) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "quant", "gemm", "rms",
              "quant", "gemm", "head_rms", "rope", "gemm", "rms",
              "rope", "kv_simulate", "d2h"}));
  EXPECT_EQ(fixture.host_error, 0U);
}

TEST(DeepSeekAttentionProjectionCoordinatorTest,
     RejectsBrokenIntermediateFlowBeforeAnySubmission) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto invalid = submission();
  invalid.q_norm.input_bf16 = 99;
  EXPECT_FALSE(fixture.coordinator.launch(invalid, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}

TEST(DeepSeekAttentionProjectionCoordinatorTest,
     ReusesAnErrorChannelClaimedByEarlierLayerWork) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(std::count(fixture.operations.calls.begin(),
                       fixture.operations.calls.end(), "zero"), 1);
}

}}  // namespace pih::<anonymous>
