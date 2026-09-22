#include "pih/model/deepseek_dspark_prefill_stage_operation.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih { namespace {

class PrefillFixedOperations final : public DeepSeekFixedStateBankOperations {
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

class PrefillOperations final : public DeepSeekDsparkPrefillStageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status quant(DeepSeekFp8ActivationQuantLaunch launch) override {
    calls.push_back("quant"); quant_launch = launch; return Status::Ok();
  }
  Status gemm(DeepSeekFp8GemmLaunch launch) override {
    calls.push_back("gemm"); gemm_launch = launch; return Status::Ok();
  }
  Status rms(DeepSeekRmsNormLaunch launch) override {
    calls.push_back("rms"); rms_launch = launch; return Status::Ok();
  }
  Status rotary(DeepSeekRotaryLaunch launch) override {
    calls.push_back("rotary"); rotary_launch = launch; return Status::Ok();
  }
  Status kv_fp8_simulate(DeepSeekKvFp8SimulateLaunch launch) override {
    calls.push_back("simulate"); simulate_launch = launch; return Status::Ok();
  }
  Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) override {
    calls.push_back("store"); store_launch = launch;
    return fail_store ? Status::Unavailable("injected store failure")
                      : Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("event"); return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    if (polls_before_complete != 0) {
      --polls_before_complete;
      return DeepSeekExpertAsyncStatus::kInProgress;
    }
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status synchronize_stream(std::uintptr_t) override {
    calls.push_back("sync"); return Status::Ok();
  }
  std::vector<std::string> calls;
  DeepSeekFp8ActivationQuantLaunch quant_launch;
  DeepSeekFp8GemmLaunch gemm_launch;
  DeepSeekRmsNormLaunch rms_launch;
  DeepSeekRotaryLaunch rotary_launch;
  DeepSeekKvFp8SimulateLaunch simulate_launch;
  DeepSeekDsparkRecentStoreLaunch store_launch;
  bool fail_store = false;
  std::uint32_t polls_before_complete = 0;
};

struct PrefillFixture final {
  PrefillFixture() {
    EXPECT_TRUE(transaction.begin(11).ok());
    const auto bank = transaction.tentative_fixed_state().value();
    const auto stage = DeepSeekDsparkStageId::kMtp1;
    const auto recent = layout.ResolveDspark(stage, bank).value();
    resources = {
        stage, {0x101, 0x102, 0x103, 29}, &layout, &transaction,
        transaction.prepare_epoch(), recent, 0x201, 0x202, 0x203,
        0x204, 0x205, 0x206, 0x207, 11, 12, 5, 4096, 29};
  }
  static constexpr std::uint64_t kStateBytes =
      kDeepSeekDsparkStageCount * kDeepSeekDsparkRecentStateBytes;
  PrefillFixedOperations fixed_operations;
  DeepSeekFixedStateLayout layout =
      DeepSeekFixedStateLayout::Build({}, true).value();
  DeepSeekFixedStateBanks fixed_banks = DeepSeekFixedStateBanks::Create(
      {0x100000, kStateBytes}, {0x200000, kStateBytes}, 121,
      fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, fixed_banks, ratio4, ratio128).value();
  DeepSeekDsparkMtpPrefillResources resources;
  PrefillOperations operations;
  std::uint32_t host_error = 7;
};

TEST(DeepSeekDsparkPrefillStageOperationTest,
     SubmitsOfficialAttentionOnlyStateInitializationOrder) {
  PrefillFixture fixture;
  auto operation = DeepSeekDsparkPrefillStageOperation::Create(
      fixture.operations, &fixture.host_error).value();
  const DeepSeekPipelinePlanDescriptor plan{
      9, 17, DeepSeekPlanPhase::kPrefill, 5, 1};

  ASSERT_TRUE(operation.launch_prefill(plan, fixture.resources).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "quant", "gemm", "rms",
                                      "rotary", "simulate", "store", "d2h",
                                      "event"}));
  EXPECT_EQ(fixture.host_error, 0U);
  EXPECT_EQ(fixture.operations.quant_launch.logical_k, 4096U);
  EXPECT_EQ(fixture.operations.gemm_launch.n, 512U);
  EXPECT_EQ(fixture.operations.rms_launch.output_bf16,
            fixture.resources.kv_scratch_bf16);
  EXPECT_EQ(fixture.operations.rotary_launch.head_count, 1U);
  EXPECT_EQ(fixture.operations.store_launch.recent_ring_bf16,
            fixture.resources.recent_state.recent_bf16.address);
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekDsparkPrefillStageOperationTest,
     DoesNotAcknowledgeBeforeCompletionEvent) {
  PrefillFixture fixture;
  fixture.operations.polls_before_complete = 1;
  auto operation = DeepSeekDsparkPrefillStageOperation::Create(
      fixture.operations, &fixture.host_error).value();
  ASSERT_TRUE(operation.launch_prefill(
      {9, 17, DeepSeekPlanPhase::kPrefill, 5, 1}, fixture.resources).ok());
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekDsparkPrefillStageOperationTest,
     ReusesEmbedErrorChannelAndSynchronizesPartialFailureOnCancel) {
  PrefillFixture fixture;
  ASSERT_TRUE(fixture.transaction.claim_external_error_channel(
      &fixture.host_error, fixture.resources.error_flag_u32).value());
  auto operation = DeepSeekDsparkPrefillStageOperation::Create(
      fixture.operations, &fixture.host_error).value();
  fixture.operations.fail_store = true;
  const DeepSeekPipelinePlanDescriptor plan{
      9, 17, DeepSeekPlanPhase::kPrefill, 5, 1};

  EXPECT_FALSE(operation.launch_prefill(plan, fixture.resources).ok());
  EXPECT_EQ(fixture.operations.calls.front(), "quant");
  EXPECT_EQ(fixture.operations.calls.back(), "store");
  EXPECT_TRUE(operation.cancel().ok());
  EXPECT_EQ(fixture.operations.calls.back(), "sync");
  EXPECT_FALSE(operation.poll().ok());
}

TEST(DeepSeekDsparkPrefillStageOperationTest,
     RejectsDecodeAndMalformedRecentStoreGeometry) {
  PrefillFixture fixture;
  auto operation = DeepSeekDsparkPrefillStageOperation::Create(
      fixture.operations, &fixture.host_error).value();
  EXPECT_FALSE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, {}).ok());
  DeepSeekDsparkRecentStoreLaunch store{
      1, 2, 3, 4, 5, 6, 512, 128, 4096};
  EXPECT_TRUE(validate_deepseek_dspark_recent_store_launch(store).ok());
  store.ring_rows = 64;
  EXPECT_FALSE(validate_deepseek_dspark_recent_store_launch(store).ok());
}

}}  // namespace pih::<anonymous>
