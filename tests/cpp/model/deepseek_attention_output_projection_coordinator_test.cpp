#include "pih/model/deepseek_attention_output_projection_coordinator.h"

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
class Operations final : public DeepSeekAttentionOutputProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status rotary(DeepSeekRotaryLaunch value) override {
    calls.push_back(value.inverse ? "inverse_rope" : "rope"); return Status::Ok();
  }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override {
    calls.push_back("wo_a"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch) override {
    calls.push_back("quant"); return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch) override {
    calls.push_back("wo_b"); return Status::Ok();
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
      {0x1000, 256}, {0x2000, 256}, 81, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 = DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(1, 0, 0, banks, ratio4,
                                                    ratio128).value();
  Operations operations;
  std::uint32_t host_error = 3;
  DeepSeekAttentionOutputProjectionCoordinator coordinator =
      DeepSeekAttentionOutputProjectionCoordinator::Create(
          operations, &host_error).value();
};
DeepSeekAttentionOutputProjectionSubmission submission() {
  constexpr std::uintptr_t error = 90, stream = 11;
  DeepSeekAttentionOutputProjectionSubmission value;
  value.inverse_rope = {1, 2, error, stream, 2, 64, 512, 64, true, 3, 104};
  value.wo_a = {1, 3, 4, 5, 6, 7, error, stream, 2, 8, 1024, 4096};
  value.quant = {7, 8, 9, error, stream, 2, 8192};
  value.wo_b = {8, 9, 10, 11, 12, error, stream, 2, 4096, 8192};
  return value;
}

TEST(DeepSeekAttentionOutputProjectionCoordinatorTest,
     QueuesInverseRopeAndOfficialGroupedOutputProjection) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.coordinator.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "inverse_rope", "wo_a",
                                      "quant", "wo_b", "d2h"}));
}
TEST(DeepSeekAttentionOutputProjectionCoordinatorTest,
     RejectsForwardRopeOrBrokenIntermediateFlowBeforeLaunch) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto invalid = submission();
  invalid.inverse_rope.inverse = false;
  EXPECT_FALSE(fixture.coordinator.launch(invalid, fixture.transaction).ok());
  invalid = submission();
  invalid.quant.input_bf16 = 99;
  EXPECT_FALSE(fixture.coordinator.launch(invalid, fixture.transaction).ok());
  EXPECT_TRUE(fixture.operations.calls.empty());
}
}}  // namespace pih::<anonymous>
