#include "pih/model/deepseek_mhc_stage_backend.h"

#include <gtest/gtest.h>

#include <array>
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

class MhcOperations final : public DeepSeekMhcSequenceOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    calls.push_back("zero"); return Status::Ok();
  }
  Status pre(DeepSeekMhcPreLaunch) override {
    calls.push_back("pre"); return Status::Ok();
  }
  Status branch(DeepSeekMhcBranchLaunch) override {
    return Status::Internal("stage wrapper must own the branch");
  }
  Status post(DeepSeekMhcPostLaunch) override {
    calls.push_back("post"); return Status::Ok();
  }
  Status target_hidden_tap(DeepSeekMhcTargetHiddenTapLaunch) override {
    calls.push_back("tap"); return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    calls.push_back("d2h"); return Status::Ok();
  }
  std::vector<std::string> calls;
};

class Inner final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand& value,
                const DeepSeekPipelinePlanDescriptor&) override {
    command = value;
    ++launch_count;
    return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    ++poll_count;
    if (fail_poll) return DeepSeekStageComputeStatus::kError;
    return poll_count == 1 ? DeepSeekStageComputeStatus::kInProgress
                           : DeepSeekStageComputeStatus::kSuccess;
  }
  DeepSeekStageOperatorCommand command;
  std::uint32_t launch_count = 0;
  std::uint32_t poll_count = 0;
  bool fail_poll = false;
};

struct SequenceFixture final {
  explicit SequenceFixture(std::uint32_t sequence,
                           std::uintptr_t device_error)
      : transaction(DeepSeekAttentionSequenceTransaction::Create(
            sequence, 0, 0, banks, ratio4, ratio128).value()),
        executor(DeepSeekMhcSequenceExecutor::Create(operations, &host_error)
                     .value()) {
    submission.device_error_flag_u32 = device_error;
  }
  FixedOperations fixed;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x1000, 256}, {0x2000, 256}, 41, fixed).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction;
  MhcOperations operations;
  std::uint32_t host_error = 0;
  DeepSeekMhcSequenceExecutor executor;
  DeepSeekMhcSequenceSubmission submission{
      .kind = DeepSeekMhcBranchKind::kAttention,
      .layer_id = 4,
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
      .device_error_flag_u32 = 0,
      .stream = 11,
      .token_count = 1,
      .rms_epsilon = 1.0e-6F,
      .pre_epsilon = 1.0e-6F,
      .sinkhorn_epsilon = 1.0e-6F,
      .sinkhorn_iterations = 20};
};

class Provider final : public DeepSeekMhcStageWorkProvider {
 public:
  Result<std::span<const DeepSeekMhcStageSequenceWork>> resolve(
      const DeepSeekStageOperatorCommand&,
      const DeepSeekPipelinePlanDescriptor&) override {
    return std::span<const DeepSeekMhcStageSequenceWork>(work);
  }
  std::vector<DeepSeekMhcStageSequenceWork> work;
};

DeepSeekPipelinePlanDescriptor plan() {
  return {3, 7, DeepSeekPlanPhase::kDecode, 2, 2};
}

TEST(DeepSeekMhcStageBackendTest,
     WaitsForTheAsynchronousAttentionBranchBeforePostingMhc) {
  SequenceFixture first(1, 201);
  SequenceFixture second(2, 202);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.submission},
                   {&second.executor, &second.transaction, second.submission}};
  Inner inner;
  auto backend = DeepSeekMhcStageOperatorBackend::Create(inner, provider).value();

  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                             plan()).ok());
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "pre"}));
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "pre"}));
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "pre", "post", "d2h"}));
  EXPECT_EQ(second.operations.calls, first.operations.calls);
}

TEST(DeepSeekMhcStageBackendTest, RejectsPackedWorkCountBeforeLaunchingPre) {
  SequenceFixture first(1, 201);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.submission}};
  Inner inner;
  auto backend = DeepSeekMhcStageOperatorBackend::Create(inner, provider).value();
  EXPECT_FALSE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                              plan()).ok());
  EXPECT_TRUE(first.operations.calls.empty());
  EXPECT_EQ(inner.launch_count, 0U);
}

TEST(DeepSeekMhcStageBackendTest, WrapsFeedForwardAndMapsItToTheMoeBranch) {
  SequenceFixture first(1, 201);
  SequenceFixture second(2, 202);
  first.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
  second.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.submission},
                   {&second.executor, &second.transaction, second.submission}};
  Inner inner;
  auto backend = DeepSeekMhcStageOperatorBackend::Create(inner, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kMoe, 4}, plan()).ok());
  EXPECT_EQ(inner.command.kind, DeepSeekStageOperatorKind::kMoe);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kInProgress);
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "pre", "post", "d2h"}));
}

TEST(DeepSeekMhcStageBackendTest,
     BranchFailureNeverPostsResidualsAndPoisonsTheWrapper) {
  SequenceFixture first(1, 201);
  SequenceFixture second(2, 202);
  ASSERT_TRUE(first.transaction.begin(11).ok());
  ASSERT_TRUE(second.transaction.begin(11).ok());
  Provider provider;
  provider.work = {{&first.executor, &first.transaction, first.submission},
                   {&second.executor, &second.transaction, second.submission}};
  Inner inner;
  inner.fail_poll = true;
  auto backend = DeepSeekMhcStageOperatorBackend::Create(inner, provider).value();
  ASSERT_TRUE(backend.launch({DeepSeekStageOperatorKind::kAttention, 4},
                             plan()).ok());
  EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kError);
  EXPECT_EQ(first.operations.calls,
            std::vector<std::string>({"zero", "pre"}));
  EXPECT_FALSE(backend.poll().ok());
}

}}  // namespace pih::<anonymous>
