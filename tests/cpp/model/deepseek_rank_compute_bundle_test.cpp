#include "pih/model/deepseek_rank_compute_bundle.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class Transfer final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity, std::uint32_t, std::uint64_t,
               std::uint64_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class Kernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

Result<TensorView> resident_tensor(std::string_view name) {
  const bool scale = name.ends_with(".scale");
  const bool w2 = name.find(".w2.") != std::string_view::npos;
  const std::array<std::int64_t, 2> shape = scale
      ? std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 64 : 128}
      : std::array<std::int64_t, 2>{w2 ? 4096 : 2048, w2 ? 1024 : 2048};
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U +
          (std::hash<std::string_view>{}(name) & 0x0fffffffU)),
      scale ? DType::kFloat8E8M0 : DType::kInt8, shape, {},
      Device::Create(DeviceType::kCuda, 0).value(), 17);
}

TEST(DeepSeekRankComputeBundleTest,
     SingleRankOwnsEndpointDsparkAndSurvivesMove) {
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  const auto stage = plan->rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2);
  ASSERT_TRUE(pager.ok());
  Transfer transfer;
  Kernel kernel;
  auto created = DeepSeekRankComputeBundle::Create(
      stage, 8, 32, *pager, transfer, kernel);
  ASSERT_TRUE(created.ok()) << created.status().message();
  DeepSeekRankComputeBundle moved(std::move(*created));
  EXPECT_NE(moved.endpoint(), nullptr);
  EXPECT_NE(moved.dspark(), nullptr);
  EXPECT_EQ(moved.router().plan_provider(), &moved.expert_store());
  const DeepSeekPipelinePlanDescriptor drain{
      7, 1, DeepSeekPlanPhase::kDrain, 0, 1};
  EXPECT_FALSE(moved.driver().launch(drain, stage).ok());
  ASSERT_TRUE(moved.bind_plan(drain, {}).ok());
  ASSERT_TRUE(moved.driver().launch(drain, stage).ok());
  auto result = moved.driver().poll();
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(*result, DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRankComputeBundleTest,
     RejectsIncompleteDecodeWithoutPublishingPlan) {
  auto plan = DeepSeekPipelinePlan::Create(1, false).value();
  const auto stage = plan.rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, 8, 32, pager, transfer, kernel, 77).value();
  const DeepSeekPipelinePlanDescriptor decode{
      7, 1, DeepSeekPlanPhase::kDecode, 1, 1};
  EXPECT_FALSE(bundle.bind_plan(decode, {}).ok());
  EXPECT_FALSE(bundle.driver().launch(decode, stage).ok());
}

TEST(DeepSeekRankComputeBundleTest,
     DsparkEnabledPrefillRejectsIncompleteOperatorWorkNotThePhase) {
  auto plan = DeepSeekPipelinePlan::Create(1, true).value();
  const auto stage = plan.rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, 8, 32, pager, transfer, kernel, 77).value();
  const DeepSeekPipelinePlanDescriptor prefill{
      7, 1, DeepSeekPlanPhase::kPrefill, 1, 1};
  EXPECT_EQ(bundle.bind_plan(prefill, {}).code(),
            StatusCode::kInvalidArgument);
}

TEST(DeepSeekRankComputeBundleTest,
     ResidentBundleDoesNotRequirePagerOrTransfer) {
  auto pipeline = DeepSeekPipelinePlan::Create(3, false).value();
  const auto stage = pipeline.rank(1);
  auto resident = DeepSeekResidentExpertBindings::Resolve(
      stage, resident_tensor).value();
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::CreateResident(
      stage, 8, 32, kernel, resident);
  ASSERT_TRUE(bundle.ok()) << bundle.status().message();
  const DeepSeekPipelinePlanDescriptor drain{
      7, 1, DeepSeekPlanPhase::kDrain, 0, 1};
  ASSERT_TRUE(bundle->bind_plan(drain, {}).ok());
  ASSERT_TRUE(bundle->launch(drain, stage).ok());
  EXPECT_EQ(bundle->poll().value(), DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRankComputeBundleTest,
     CompletedDrainCanBeClearedAndReboundToNextPlan) {
  auto plan = DeepSeekPipelinePlan::Create(1, false).value();
  const auto stage = plan.rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, 8, 32, pager, transfer, kernel).value();
  for (std::uint64_t sequence = 1; sequence <= 2; ++sequence) {
    const DeepSeekPipelinePlanDescriptor drain{
        7, sequence, DeepSeekPlanPhase::kDrain, 0, 1};
    ASSERT_TRUE(bundle.bind_plan(drain, {}).ok());
    ASSERT_TRUE(bundle.driver().launch(drain, stage).ok());
    auto result = bundle.driver().poll();
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(*result, DeepSeekStageComputeStatus::kSuccess);
  }
}

TEST(DeepSeekRankComputeBundleTest,
     PreparedBindingCanAbortButInflightBindingCannot) {
  auto plan = DeepSeekPipelinePlan::Create(1, false).value();
  const auto stage = plan.rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2).value();
  Transfer transfer;
  Kernel kernel;
  auto bundle = DeepSeekRankComputeBundle::Create(
      stage, 8, 32, pager, transfer, kernel).value();
  const DeepSeekPipelinePlanDescriptor first{
      7, 1, DeepSeekPlanPhase::kDrain, 0, 1};
  const DeepSeekPipelinePlanDescriptor second{
      7, 2, DeepSeekPlanPhase::kDrain, 0, 1};
  ASSERT_TRUE(bundle.bind_plan(first, {}).ok());
  EXPECT_FALSE(bundle.bind_plan(second, {}).ok());
  ASSERT_TRUE(bundle.abort_bound_plan(first).ok());
  EXPECT_TRUE(bundle.can_abort_bound_plan(first));
  EXPECT_TRUE(bundle.abort_bound_plan(first).ok());
  ASSERT_TRUE(bundle.bind_plan(second, {}).ok());
  EXPECT_FALSE(bundle.can_abort_bound_plan(first));
  EXPECT_FALSE(bundle.abort_bound_plan(first).ok());
  ASSERT_TRUE(bundle.driver().launch(second, stage).ok());
  EXPECT_FALSE(bundle.abort_bound_plan(second).ok());
  ASSERT_EQ(bundle.driver().poll().value(),
            DeepSeekStageComputeStatus::kSuccess);
}

TEST(DeepSeekRankComputeBundleTest,
     PipelineRanksOwnOnlyTheirCanonicalEndpointComponents) {
  auto plan = DeepSeekPipelinePlan::Create(3, true);
  ASSERT_TRUE(plan.ok());
  Transfer transfer;
  Kernel kernel;
  for (std::uint32_t rank = 0; rank < 3; ++rank) {
    const auto stage = plan->rank(rank);
    auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2);
    ASSERT_TRUE(pager.ok());
    auto bundle = DeepSeekRankComputeBundle::Create(
        stage, 4, 16, *pager, transfer, kernel);
    ASSERT_TRUE(bundle.ok()) << bundle.status().message();
    EXPECT_EQ(bundle->endpoint() != nullptr, rank == 0 || rank == 2);
    EXPECT_EQ(bundle->dspark() != nullptr, rank == 2);
  }
}

TEST(DeepSeekRankComputeBundleTest,
     RejectsPagerForDifferentLayerOwnerAndInvalidCapacity) {
  auto plan = DeepSeekPipelinePlan::Create(2, false);
  ASSERT_TRUE(plan.ok());
  auto wrong_pager = DeepSeekExpertPager::Create(plan->rank(1).layers, 2, 2);
  ASSERT_TRUE(wrong_pager.ok());
  Transfer transfer;
  Kernel kernel;
  EXPECT_FALSE(DeepSeekRankComputeBundle::Create(
      plan->rank(0), 4, 16, *wrong_pager, transfer, kernel).ok());
  auto pager = DeepSeekExpertPager::Create(plan->rank(0).layers, 2, 2);
  ASSERT_TRUE(pager.ok());
  EXPECT_FALSE(DeepSeekRankComputeBundle::Create(
      plan->rank(0), 0, 16, *pager, transfer, kernel).ok());
  EXPECT_FALSE(DeepSeekRankComputeBundle::Create(
      plan->rank(0), 4, 0, *pager, transfer, kernel).ok());
}

}  // namespace
}  // namespace pih
