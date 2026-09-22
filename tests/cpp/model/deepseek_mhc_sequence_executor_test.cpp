#include "pih/model/deepseek_mhc_sequence_executor.h"

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

class Operations final : public DeepSeekMhcSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t device, std::uintptr_t) override {
    calls.push_back("zero");
    zero_devices.push_back(device);
    return Status::Ok();
  }
  Status pre(DeepSeekMhcPreLaunch value) override {
    calls.push_back("pre");
    pre_launch = value;
    return Status::Ok();
  }
  Status branch(DeepSeekMhcBranchLaunch value) override {
    calls.push_back("branch");
    branch_launch = value;
    return fail_branch ? Status::Internal("injected branch failure")
                       : Status::Ok();
  }
  Status post(DeepSeekMhcPostLaunch value) override {
    calls.push_back("post");
    post_launch = value;
    return Status::Ok();
  }
  Status target_hidden_tap(DeepSeekMhcTargetHiddenTapLaunch value) override {
    calls.push_back("tap");
    target_hidden_tap_launch = value;
    return fail_tap ? Status::Internal("injected tap failure") : Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h");
    return Status::Ok();
  }
  std::vector<std::string> calls;
  DeepSeekMhcPreLaunch pre_launch;
  DeepSeekMhcBranchLaunch branch_launch;
  DeepSeekMhcPostLaunch post_launch;
  DeepSeekMhcTargetHiddenTapLaunch target_hidden_tap_launch;
  bool fail_branch = false;
  bool fail_tap = false;
  std::vector<std::uintptr_t> zero_devices;
};

struct Fixture final {
  FixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 31, fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          9, 0, 0, banks, ratio4, ratio128).value();
  Operations operations;
  std::uint32_t host_error = 7;
  DeepSeekMhcSequenceExecutor executor =
      DeepSeekMhcSequenceExecutor::Create(operations, &host_error).value();
};

DeepSeekMhcSequenceSubmission submission() {
  return {.kind = DeepSeekMhcBranchKind::kAttention,
          .layer_id = 3,
          .residual_bf16 = 101,
          .fn_f32 = 102,
          .scale_f32 = 103,
          .base_f32 = 104,
          .norm_weight_bf16 = 111,
          .post_mix_f32 = 105,
          .residual_mix_f32 = 106,
          .layer_input_bf16 = 107,
          .branch_output_bf16 = 108,
          .output_bf16 = 109,
          .device_error_flag_u32 = 110,
          .stream = 11,
          .token_count = 2,
          .rms_epsilon = 1.0e-6F,
          .pre_epsilon = 1.0e-6F,
          .sinkhorn_epsilon = 1.0e-6F,
          .sinkhorn_iterations = 20};
}

TEST(DeepSeekMhcSequenceExecutorTest,
     OrdersPreBranchPostOnOneSequenceErrorChannel) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "pre", "branch", "post", "d2h"}));
  EXPECT_EQ(fixture.host_error, 0U);
  EXPECT_EQ(fixture.operations.pre_launch.error_flag_u32, 110U);
  EXPECT_EQ(fixture.operations.pre_launch.norm_weight_bf16, 111U);
  EXPECT_EQ(fixture.operations.branch_launch.input_bf16, 107U);
  EXPECT_EQ(fixture.operations.branch_launch.output_bf16, 108U);
  EXPECT_EQ(fixture.operations.post_launch.output_bf16, 109U);
}

TEST(DeepSeekMhcSequenceExecutorTest,
     ReusesAnAlreadyClaimedChannelWithoutClearingAccumulatedErrors) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.launch(submission(), fixture.transaction).ok());
  ASSERT_TRUE(fixture.executor.launch(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "pre", "branch", "post", "d2h",
                                      "pre", "branch", "post", "d2h"}));
}

TEST(DeepSeekMhcSequenceExecutorTest,
     RejectsCrossSequenceStreamAndPoisonsAfterPartialSubmissionFailure) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto wrong = submission();
  wrong.stream = 12;
  EXPECT_FALSE(fixture.executor.launch(wrong, fixture.transaction).ok());
  fixture.operations.fail_branch = true;
  EXPECT_FALSE(fixture.executor.launch(submission(), fixture.transaction).ok());
  fixture.operations.fail_branch = false;
  EXPECT_FALSE(fixture.executor.launch(submission(), fixture.transaction).ok());
}

TEST(DeepSeekMhcSequenceExecutorTest,
     PackedSequencesKeepIndependentHostAndDeviceErrorChannels) {
  Fixture first;
  Fixture second;
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  auto first_submission = submission();
  auto second_submission = submission();
  second_submission.device_error_flag_u32 = 210;

  ASSERT_TRUE(first.executor.launch(first_submission, first.transaction).ok());
  ASSERT_TRUE(second.executor.launch(second_submission, second.transaction).ok());

  EXPECT_EQ(first.operations.zero_devices,
            std::vector<std::uintptr_t>({110}));
  EXPECT_EQ(second.operations.zero_devices,
            std::vector<std::uintptr_t>({210}));
  EXPECT_EQ(first.operations.pre_launch.error_flag_u32, 110U);
  EXPECT_EQ(second.operations.pre_launch.error_flag_u32, 210U);
}

TEST(DeepSeekMhcSequenceExecutorTest,
     DefersPostUntilAnAsynchronousBranchHasCompleted) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  ASSERT_TRUE(fixture.executor.begin(submission(), fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "pre"}));
  ASSERT_TRUE(fixture.executor.finish(fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>({"zero", "pre", "post", "d2h"}));
}

TEST(DeepSeekMhcSequenceExecutorTest,
     CapturesTargetHiddenAfterPostAndBeforeErrorCopy) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto tapped = submission();
  tapped.kind = DeepSeekMhcBranchKind::kFeedForward;
  tapped.layer_id = 40;
  tapped.target_hidden_bf16 = 0x9000;
  tapped.target_stage_index = 0;

  ASSERT_TRUE(fixture.executor.launch(tapped, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>(
                {"zero", "pre", "branch", "post", "tap", "d2h"}));
  EXPECT_EQ(fixture.operations.target_hidden_tap_launch.residual_hc_bf16,
            tapped.output_bf16);
  EXPECT_EQ(fixture.operations.target_hidden_tap_launch.target_hidden_bf16,
            0x9000U);
  EXPECT_EQ(fixture.operations.target_hidden_tap_launch.target_stage_index,
            0U);
}

TEST(DeepSeekMhcSequenceExecutorTest,
     TargetHiddenTapFailurePoisonsBeforeErrorCopy) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(11).ok());
  auto tapped = submission();
  tapped.kind = DeepSeekMhcBranchKind::kFeedForward;
  tapped.layer_id = 42;
  tapped.target_hidden_bf16 = 0x9000;
  tapped.target_stage_index = 2;
  fixture.operations.fail_tap = true;

  EXPECT_FALSE(fixture.executor.launch(tapped, fixture.transaction).ok());
  EXPECT_EQ(fixture.operations.calls,
            std::vector<std::string>(
                {"zero", "pre", "branch", "post", "tap"}));
  fixture.operations.fail_tap = false;
  EXPECT_FALSE(fixture.executor.launch(tapped, fixture.transaction).ok());
}

}}  // namespace pih::<anonymous>
