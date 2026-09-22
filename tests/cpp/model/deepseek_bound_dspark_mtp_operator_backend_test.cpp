#include "pih/model/deepseek_bound_dspark_mtp_operator_backend.h"

#include <gtest/gtest.h>

#include <array>
#include <string>
#include <vector>

#include "deepseek_dspark_weight_test_fixture.h"
#include "pih/model/deepseek_dspark_resident_subwave_executor.h"

namespace pih {
namespace {

Result<TensorView> resident_tensor(std::string_view name) {
  const bool scale = name.ends_with(".scale");
  const bool w2 = name.find(".w2.") != std::string_view::npos;
  const std::array<std::int64_t, 2> shape = scale
      ? std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 64 : 128}
      : std::array<std::int64_t, 2>{w2 ? 4096 : 2048,
                                    w2 ? 1024 : 2048};
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U +
          (std::hash<std::string_view>{}(name) & 0x0fffffffU)),
      scale ? DType::kFloat8E8M0 : DType::kInt8, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 29);
}

class FixedStateOperations final : public DeepSeekFixedStateBankOperations {
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

class Operation final : public DeepSeekDsparkMtpStageOperation {
 public:
  explicit Operation(std::string label, std::vector<std::string>& calls)
      : label_(std::move(label)), calls_(&calls) {}
  Status launch(const DeepSeekPipelinePlanDescriptor& plan,
                const DeepSeekDsparkMtpStageResources& resources) override {
    calls_->push_back(label_ + ":launch:" +
                      std::to_string(deepseek_dspark_stage_index(
                          resources.stage)));
    sequence = plan.plan_sequence;
    stage = resources.stage;
    return fail_launch
               ? Status::Unavailable(label_ + " launch failed")
               : Status::Ok();
  }
  Status launch_prefill(
      const DeepSeekPipelinePlanDescriptor& plan,
      const DeepSeekDsparkMtpPrefillResources& resources) override {
    calls_->push_back(label_ + ":prefill:" +
                      std::to_string(deepseek_dspark_stage_index(
                          resources.stage)));
    sequence = plan.plan_sequence;
    stage = resources.stage;
    return fail_launch
               ? Status::Unavailable(label_ + " prefill failed")
               : Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    calls_->push_back(label_ + ":poll");
    return fail ? DeepSeekStageComputeStatus::kError
                : DeepSeekStageComputeStatus::kSuccess;
  }
  Status cancel() override {
    calls_->push_back(label_ + ":cancel");
    ++cancels;
    return fail_cancel
               ? Status::Internal(label_ + " cancel failed")
               : Status::Ok();
  }
  std::uint64_t sequence = 0;
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  bool fail = false;
  bool fail_launch = false;
  bool fail_cancel = false;
  std::uint32_t cancels = 0;
 private:
  std::string label_;
  std::vector<std::string>* calls_;
};

class Kernel final : public DeepSeekDsparkExpertKernelDriver {
 public:
  Status launch_resident(DeepSeekDsparkStageId, std::uint16_t,
                         std::uint64_t,
                         const DeepSeekExpertBundleDeviceView&,
                         const DeepSeekExpertRoute*,
                         std::uint32_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Provider final : public DeepSeekDsparkMtpStageWorkProvider {
 public:
  Result<const DeepSeekBoundDsparkMtpStageWork*> resolve(
      DeepSeekDsparkStageId stage,
      const DeepSeekPipelinePlanDescriptor& plan) override {
    stages.push_back(stage);
    sequences.push_back(plan.plan_sequence);
    return &work.at(deepseek_dspark_stage_index(stage));
  }
  std::array<DeepSeekBoundDsparkMtpStageWork,
             kDeepSeekDsparkStageCount> work;
  std::vector<DeepSeekDsparkStageId> stages;
  std::vector<std::uint64_t> sequences;
};

struct Fixture final {
  Fixture()
      : weights(test::deepseek_dspark_test_weights()),
        experts(DeepSeekDsparkResidentExpertBindings::Resolve(
                    test::deepseek_dspark_test_stage(), resident_tensor)
                    .value()),
        attention{Operation("a0", calls), Operation("a1", calls),
                  Operation("a2", calls)},
        moe{Operation("m0", calls), Operation("m1", calls),
            Operation("m2", calls)} {
    EXPECT_TRUE(transaction.begin(11).ok());
    const auto bank = transaction.tentative_fixed_state().value();
    for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
      const auto stage = deepseek_dspark_stage_id(index).value();
      provider.work[index] = {
          {stage, &weights.common(stage), &experts,
           &state_layout, &transaction, transaction.prepare_epoch(),
           {stage, {bank.address + index * kDeepSeekDsparkRecentStateBytes,
                    kDeepSeekDsparkRecentStateBytes}},
           weights.generation()},
          {stage,
           {weights.common(stage).attention.wkv_fp8,
            weights.common(stage).attention.wkv_scale_ue8m0,
            weights.common(stage).attention.kv_norm_bf16,
            weights.generation()},
           &state_layout, &transaction, transaction.prepare_epoch(),
           {stage, {bank.address + index * kDeepSeekDsparkRecentStateBytes,
                    kDeepSeekDsparkRecentStateBytes}},
           0x300000U, 0x310000U, 0x320000U,
           0x400000U, 0x500000U, 0x600000U, 0x700000U,
           transaction.stream(), 12, 5, 4096, weights.generation()},
          &attention[index], &moe[index]};
      auto& resources = provider.work[index].resources;
      const auto base = static_cast<std::uintptr_t>(
          0x800000U + index * 0x100000U);
      resources.attention_workspace = {
          base + 0x001U, base + 0x002U, base + 0x003U,
          base + 0x004U, base + 0x005U, base + 0x006U,
          base + 0x007U, base + 0x008U, base + 0x009U,
          base + 0x00aU, base + 0x00bU, base + 0x00cU,
          base + 0x00dU, base + 0x00eU, base + 0x00fU,
          base + 0x010U, base + 0x011U, base + 0x012U};
      resources.mhc_workspace = {
          base + 0x101U, base + 0x102U, base + 0x103U,
          base + 0x104U, base + 0x105U, base + 0x106U};
      resources.stage_input_hc_bf16 = base + 0x201U;
      resources.attention_output_hc_bf16 = base + 0x202U;
      resources.stage_output_hc_bf16 = base + 0x203U;
      resources.main_normalized_bf16 = base + 0x204U;
      resources.main_activation_e4m3 = base + 0x205U;
      resources.main_activation_scale_ue8m0 = base + 0x206U;
      resources.main_kv_bf16 = base + 0x207U;
      resources.main_positions_u32 = base + 0x208U;
      resources.draft_positions_u32 = base + 0x209U;
      resources.rope_frequencies_f32 = base + 0x20aU;
      resources.error_flag_u32 = base + 0x20bU;
      resources.stream = transaction.stream();
      resources.completion_event = base + 0x20cU;
      resources.router_scores_f32 = base + 0x20dU;
      resources.router_host_scores = router_scores;
      resources.router_host_bias = router_bias;
      resources.expert_arena = expert_arena;
      resources.expert_accumulator_f32 = base + 0x20eU;
      resources.expert_kernel = &kernel;
      resources.current_position = 17;
      resources.position_table_count = 4096;
    }
  }
  std::vector<std::string> calls;
  static constexpr std::uint64_t kStateBytes =
      kDeepSeekDsparkStageCount * kDeepSeekDsparkRecentStateBytes;
  FixedStateOperations fixed_operations;
  DeepSeekFixedStateLayout state_layout =
      DeepSeekFixedStateLayout::Build({}, true).value();
  DeepSeekFixedStateBanks fixed_banks = DeepSeekFixedStateBanks::Create(
      {0x100000U, kStateBytes}, {0x200000U, kStateBytes}, 121,
      fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          1, 0, 0, fixed_banks, ratio4, ratio128).value();
  DeepSeekDsparkWeightBindings weights;
  DeepSeekDsparkResidentExpertBindings experts;
  std::array<Operation, kDeepSeekDsparkStageCount> attention;
  std::array<Operation, kDeepSeekDsparkStageCount> moe;
  Provider provider;
  std::array<float, 5 * 256> router_scores{};
  std::array<float, 256> router_bias{};
  DeepSeekExpertComputeArena expert_arena = [] {
    auto layout = DeepSeekExpertComputeArenaLayout::Create(5).value();
    return layout.bind(0x10000000U, layout.required_bytes()).value();
  }();
  Kernel kernel;
};

DeepSeekPipelinePlanDescriptor plan() {
  return {9, 17, DeepSeekPlanPhase::kDecode, 1, 1};
}

DeepSeekPipelinePlanDescriptor prefill_plan() {
  return {9, 17, DeepSeekPlanPhase::kPrefill, 5, 1};
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     DispatchesOnlyTheRequestedStageResources) {
  Fixture fixture;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = deepseek_dspark_stage_id(index).value();
    ASSERT_TRUE(backend.launch(
        {stage, DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).ok());
    EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
    ASSERT_TRUE(backend.launch(
        {stage, DeepSeekDsparkMtpOperatorKind::kMoe}, plan()).ok());
    EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
    EXPECT_EQ(fixture.attention[index].stage, stage);
    EXPECT_EQ(fixture.moe[index].stage, stage);
    EXPECT_EQ(fixture.attention[index].sequence, 17U);
    EXPECT_EQ(fixture.moe[index].sequence, 17U);
  }
  EXPECT_EQ(fixture.calls,
            (std::vector<std::string>{
                "a0:launch:0", "a0:poll", "m0:launch:0", "m0:poll",
                "a1:launch:1", "a1:poll", "m1:launch:1", "m1:poll",
                "a2:launch:2", "a2:poll", "m2:launch:2", "m2:poll"}));
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     RejectsCrossStageStateBeforeOperationLaunch) {
  Fixture fixture;
  fixture.provider.work[1].resources.recent_state.stage =
      DeepSeekDsparkStageId::kMtp0;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  EXPECT_EQ(backend.launch(
      {DeepSeekDsparkStageId::kMtp1,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     PrefillDispatchesOnlyMinimalAttentionResources) {
  Fixture fixture;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    const auto stage = deepseek_dspark_stage_id(index).value();
    ASSERT_TRUE(backend.launch(
        {stage, DeepSeekDsparkMtpOperatorKind::kAttention},
        prefill_plan()).ok());
    EXPECT_EQ(backend.poll().value(), DeepSeekStageComputeStatus::kSuccess);
  }
  EXPECT_EQ(fixture.calls,
            (std::vector<std::string>{
                "a0:prefill:0", "a0:poll", "a1:prefill:1", "a1:poll",
                "a2:prefill:2", "a2:poll"}));
  EXPECT_EQ(fixture.moe[0].sequence, 0U);
  EXPECT_EQ(fixture.moe[1].sequence, 0U);
  EXPECT_EQ(fixture.moe[2].sequence, 0U);
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     PrefillRejectsMoeAndForeignStageStateBeforeLaunch) {
  Fixture fixture;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  EXPECT_EQ(backend.launch(
      {DeepSeekDsparkStageId::kMtp0, DeepSeekDsparkMtpOperatorKind::kMoe},
      prefill_plan()).code(), StatusCode::kFailedPrecondition);
  EXPECT_TRUE(fixture.calls.empty());

  fixture.provider.work[1].prefill_resources.recent_state.stage =
      DeepSeekDsparkStageId::kMtp0;
  EXPECT_EQ(backend.launch(
      {DeepSeekDsparkStageId::kMtp1,
       DeepSeekDsparkMtpOperatorKind::kAttention},
      prefill_plan()).code(), StatusCode::kInvalidArgument);
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     RejectsMalformedSharedExpertExtentBeforeOperationLaunch) {
  Fixture fixture;
  auto forged = fixture.weights.common(DeepSeekDsparkStageId::kMtp0);
  forged.shared_expert.w2.packed.bytes = 1;
  fixture.provider.work[0].resources.weights = &forged;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  EXPECT_EQ(backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kMoe}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(fixture.calls.empty());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     RejectsMalformedDecodeGeometryAndExecutionResources) {
  Fixture batched;
  auto batched_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      batched.provider).value();
  auto batched_plan = plan();
  batched_plan.token_count = 5;
  EXPECT_EQ(batched_backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kAttention}, batched_plan).code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(batched.calls.empty());

  Fixture incomplete;
  incomplete.provider.work[0].resources.router_scores_f32 = 0;
  auto incomplete_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      incomplete.provider).value();
  EXPECT_EQ(incomplete_backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kMoe}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(incomplete.calls.empty());

  Fixture overflow;
  overflow.provider.work[0].resources.current_position = UINT32_MAX - 4U;
  auto overflow_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      overflow.provider).value();
  EXPECT_EQ(overflow_backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(overflow.calls.empty());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     RejectsForeignStateSpanAndStalePrepareEpoch) {
  Fixture foreign;
  foreign.provider.work[0].resources.recent_state.recent_bf16.address =
      0x500000U;
  auto foreign_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      foreign.provider).value();
  EXPECT_EQ(foreign_backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(foreign.calls.empty());

  Fixture aliased;
  aliased.provider.work[0].resources.recent_state.recent_bf16.address +=
      kDeepSeekDsparkRecentStateBytes;
  auto aliased_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      aliased.provider).value();
  EXPECT_EQ(aliased_backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(aliased.calls.empty());

  Fixture stale;
  ++stale.provider.work[1].resources.prepare_epoch;
  auto stale_backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      stale.provider).value();
  EXPECT_EQ(stale_backend.launch(
      {DeepSeekDsparkStageId::kMtp1,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_TRUE(stale.calls.empty());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     CancellationIsFailStopAndCancelsTheExactOperation) {
  Fixture fixture;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  ASSERT_TRUE(backend.launch(
      {DeepSeekDsparkStageId::kMtp2, DeepSeekDsparkMtpOperatorKind::kMoe},
      plan()).ok());
  EXPECT_TRUE(backend.cancel().ok());
  EXPECT_EQ(fixture.moe[2].cancels, 1U);
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_FALSE(backend.launch(
      {DeepSeekDsparkStageId::kMtp0,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).ok());
}

TEST(DeepSeekBoundDsparkMtpOperatorBackendTest,
     LaunchFailureCancelsPartialWorkAndPreservesThePrimaryError) {
  Fixture fixture;
  fixture.attention[1].fail_launch = true;
  fixture.attention[1].fail_cancel = true;
  auto backend = DeepSeekBoundDsparkMtpOperatorBackend::Create(
      fixture.provider).value();
  const auto status = backend.launch(
      {DeepSeekDsparkStageId::kMtp1,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan());
  EXPECT_EQ(status.code(), StatusCode::kUnavailable);
  EXPECT_EQ(fixture.attention[1].cancels, 1U);
  EXPECT_EQ(fixture.calls,
            (std::vector<std::string>{"a1:launch:1", "a1:cancel"}));
  EXPECT_FALSE(backend.poll().ok());
  EXPECT_FALSE(backend.launch(
      {DeepSeekDsparkStageId::kMtp1,
       DeepSeekDsparkMtpOperatorKind::kAttention}, plan()).ok());
}

}  // namespace
}  // namespace pih
