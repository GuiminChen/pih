#include "pih/model/deepseek_dspark_decode_stage_operation.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih {
namespace {

class DecodeFixedOperations final : public DeepSeekFixedStateBankOperations {
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

class DecodeOperations final : public DeepSeekDsparkDecodeStageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status positions(DeepSeekDsparkPositionLaunch launch) override {
    calls.push_back("positions"); position_launch = launch;
    return Status::Ok();
  }
  Status mhc_pre(DeepSeekMhcPreLaunch) override {
    calls.push_back("mhc_pre"); return Status::Ok();
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
  Status rotary(DeepSeekRotaryLaunch) override {
    calls.push_back("rotary"); return Status::Ok();
  }
  Status kv_simulate(DeepSeekKvFp8SimulateLaunch) override {
    calls.push_back("simulate"); return Status::Ok();
  }
  Status recent_store(DeepSeekDsparkRecentStoreLaunch launch) override {
    calls.push_back("store"); store_launch = launch; return Status::Ok();
  }
  Status attention(DeepSeekDsparkAttentionLaunch launch) override {
    calls.push_back("attention"); attention_launch = launch;
    return fail_attention ? Status::Unavailable("injected attention failure")
                          : Status::Ok();
  }
  Status grouped_gemm(DeepSeekGroupedFp8GemmLaunch) override {
    calls.push_back("grouped_gemm"); return Status::Ok();
  }
  Status mhc_post(DeepSeekMhcPostLaunch launch) override {
    calls.push_back("mhc_post"); post_launch = launch; return Status::Ok();
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
  DeepSeekDsparkPositionLaunch position_launch;
  DeepSeekDsparkRecentStoreLaunch store_launch;
  DeepSeekDsparkAttentionLaunch attention_launch;
  DeepSeekMhcPostLaunch post_launch;
  bool fail_attention = false;
  std::uint32_t polls_before_complete = 0;
};

struct DecodeFixture final {
  DecodeFixture() {
    EXPECT_TRUE(transaction.begin(11).ok());
    const auto stage = DeepSeekDsparkStageId::kMtp1;
    weights.stage = stage;
    weights.attention = {
        0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107,
        0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 29};
    weights.mhc = {0x201, 0x202, 0x203, 0x204,
                   0x205, 0x206, 0x207, 0x208, 29};
    weights.generation = 29;
    const auto bank = transaction.tentative_fixed_state().value();
    resources.stage = stage;
    resources.weights = &weights;
    resources.state_layout = &layout;
    resources.transaction = &transaction;
    resources.prepare_epoch = transaction.prepare_epoch();
    resources.recent_state = layout.ResolveDspark(stage, bank).value();
    resources.weight_generation = 29;
    resources.attention_workspace = {
        0x301, 0x302, 0x303, 0x304, 0x305, 0x306,
        0x307, 0x308, 0x309, 0x30A, 0x30B, 0x30C,
        0x30D, 0x30E, 0x30F, 0x310, 0x311, 0x312};
    resources.mhc_workspace = {
        0x401, 0x402, 0x403, 0x404, 0x405, 0x406};
    resources.stage_input_hc_bf16 = 0x501;
    resources.attention_output_hc_bf16 = 0x502;
    resources.stage_output_hc_bf16 = 0x503;
    resources.main_normalized_bf16 = 0x504;
    resources.main_activation_e4m3 = 0x505;
    resources.main_activation_scale_ue8m0 = 0x506;
    resources.main_kv_bf16 = 0x507;
    resources.main_positions_u32 = 0x508;
    resources.draft_positions_u32 = 0x509;
    resources.rope_frequencies_f32 = 0x50A;
    resources.error_flag_u32 = 0x50B;
    resources.stream = 11;
    resources.completion_event = 12;
    resources.current_position = 127;
    resources.position_table_count = 4096;
  }

  static constexpr std::uint64_t kStateBytes =
      kDeepSeekDsparkStageCount * kDeepSeekDsparkRecentStateBytes;
  DecodeFixedOperations fixed_operations;
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
  DeepSeekDsparkCommonWeightBindings weights;
  DeepSeekDsparkMtpStageResources resources;
  DecodeOperations operations;
  std::uint32_t host_error = 7;
};

TEST(DeepSeekDsparkDecodeStageOperationTest,
     SubmitsOfficialRecentDraftAttentionDataflow) {
  DecodeFixture fixture;
  auto operation = DeepSeekDsparkDecodeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  const DeepSeekPipelinePlanDescriptor plan{
      9, 17, DeepSeekPlanPhase::kDecode, 1, 1};

  ASSERT_TRUE(operation.launch(plan, fixture.resources).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({
                "zero", "positions", "mhc_pre", "quant", "gemm",
                "rms", "rotary", "simulate", "store", "quant",
                "gemm", "rms", "quant", "gemm", "head_rms",
                "rotary", "gemm", "rms", "rotary", "simulate",
                "attention", "rotary", "grouped_gemm", "quant",
                "gemm", "mhc_post", "d2h", "event"}));
  EXPECT_EQ(fixture.host_error, 0U);
  EXPECT_EQ(fixture.operations.position_launch.current_position, 127U);
  EXPECT_EQ(fixture.operations.store_launch.recent_ring_bf16,
            fixture.resources.recent_state.recent_bf16.address);
  EXPECT_EQ(fixture.operations.attention_launch.recent_count, 128U);
  EXPECT_EQ(fixture.operations.attention_launch.draft_kv_bf16,
            fixture.resources.attention_workspace.kv_bf16);
  EXPECT_EQ(fixture.operations.post_launch.output_bf16,
            fixture.resources.attention_output_hc_bf16);
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekDsparkDecodeStageOperationTest,
     DoesNotAcknowledgeBeforeCompletionEvent) {
  DecodeFixture fixture;
  fixture.operations.polls_before_complete = 1;
  auto operation = DeepSeekDsparkDecodeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  ASSERT_TRUE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(operation.poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekDsparkDecodeStageOperationTest,
     PartialFailureRequiresSynchronousCancelAndPoisonsInstance) {
  DecodeFixture fixture;
  fixture.operations.fail_attention = true;
  auto operation = DeepSeekDsparkDecodeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  const DeepSeekPipelinePlanDescriptor plan{
      9, 17, DeepSeekPlanPhase::kDecode, 1, 1};

  EXPECT_FALSE(operation.launch(plan, fixture.resources).ok());
  EXPECT_EQ(fixture.operations.calls.back(), "attention");
  EXPECT_TRUE(operation.cancel().ok());
  EXPECT_EQ(fixture.operations.calls.back(), "sync");
  EXPECT_FALSE(operation.poll().ok());
  EXPECT_FALSE(operation.launch(plan, fixture.resources).ok());
}

TEST(DeepSeekDsparkDecodeStageOperationTest,
     RejectsPrefillStageDriftAndFuturePositionOverflow) {
  DecodeFixture fixture;
  auto operation = DeepSeekDsparkDecodeStageOperation::Create(
      DeepSeekDsparkStageId::kMtp1, fixture.operations,
      &fixture.host_error).value();
  EXPECT_FALSE(operation.launch_prefill(
      {9, 17, DeepSeekPlanPhase::kPrefill, 1, 1}, {}).ok());
  fixture.resources.current_position = 4091;
  EXPECT_FALSE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
  fixture.resources.current_position = 1;
  fixture.resources.stage = DeepSeekDsparkStageId::kMtp2;
  EXPECT_FALSE(operation.launch(
      {9, 17, DeepSeekPlanPhase::kDecode, 1, 1}, fixture.resources).ok());
}

}  // namespace
}  // namespace pih
