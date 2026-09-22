#include "pih/model/deepseek_rank_compute_lane_set.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class TransferDriver final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity, std::uint32_t, std::uint64_t,
               std::uint64_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(
      DeepSeekExpertIdentity, std::uint64_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class KernelDriver final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

class TransferOwner final : public DeepSeekExpertTransferLaneOwner {
 public:
  explicit TransferOwner(std::uint32_t& destroyed) : destroyed_(&destroyed) {}
  ~TransferOwner() override { ++*destroyed_; }
  DeepSeekExpertTransferDriver& transfer() noexcept override { return driver_; }
 private:
  std::uint32_t* destroyed_;
  TransferDriver driver_;
};

class KernelOwner final : public DeepSeekExpertKernelLaneOwner {
 public:
  explicit KernelOwner(std::uint32_t& destroyed) : destroyed_(&destroyed) {}
  ~KernelOwner() override { ++*destroyed_; }
  DeepSeekExpertKernelDriver& kernel() noexcept override { return driver_; }
 private:
  std::uint32_t* destroyed_;
  KernelDriver driver_;
};

class InfrastructureOwner final
    : public DeepSeekRankComputeInfrastructureOwner {
 public:
  explicit InfrastructureOwner(
      bool& alive,
      const DeepSeekResidentExpertBindings* resident = nullptr)
      : alive_(&alive), resident_(resident) {
    *alive_ = true;
  }
  ~InfrastructureOwner() override { *alive_ = false; }
  std::uintptr_t attention_compute_stream() const noexcept override {
    return 73;
  }
  DeepSeekLearnedRouterDeviceResources* learned_router_device_resources()
      noexcept override {
    return reinterpret_cast<DeepSeekLearnedRouterDeviceResources*>(0x5100);
  }
  DeepSeekLearnedRouterStagingPool* learned_router_staging_pool()
      noexcept override {
    return reinterpret_cast<DeepSeekLearnedRouterStagingPool*>(0x5200);
  }
  DeepSeekDenseMhcRuntimeResources* dense_mhc_runtime_resources()
      noexcept override {
    return reinterpret_cast<DeepSeekDenseMhcRuntimeResources*>(0x5300);
  }
  DeepSeekAttentionProjectionDeviceResources*
  attention_projection_device_resources() noexcept override {
    return reinterpret_cast<DeepSeekAttentionProjectionDeviceResources*>(
        0x5400);
  }
  DeepSeekMhcDeviceResources* mhc_device_resources() noexcept override {
    return reinterpret_cast<DeepSeekMhcDeviceResources*>(0x5500);
  }
  const DeepSeekResidentExpertBindings* resident_expert_bindings()
      const noexcept override { return resident_; }
 private:
  bool* alive_;
  const DeepSeekResidentExpertBindings* resident_;
};

class BorrowingTransferOwner final : public DeepSeekExpertTransferLaneOwner {
 public:
  BorrowingTransferOwner(bool& infrastructure_alive, bool& safe_destruction)
      : infrastructure_alive_(&infrastructure_alive),
        safe_destruction_(&safe_destruction) {}
  ~BorrowingTransferOwner() override {
    *safe_destruction_ = *infrastructure_alive_;
  }
  DeepSeekExpertTransferDriver& transfer() noexcept override { return driver_; }
 private:
  bool* infrastructure_alive_;
  bool* safe_destruction_;
  TransferDriver driver_;
};

class BorrowingKernelOwner final : public DeepSeekExpertKernelLaneOwner {
 public:
  BorrowingKernelOwner(bool& infrastructure_alive, bool& safe_destruction)
      : infrastructure_alive_(&infrastructure_alive),
        safe_destruction_(&safe_destruction) {}
  ~BorrowingKernelOwner() override {
    *safe_destruction_ = *infrastructure_alive_;
  }
  DeepSeekExpertKernelDriver& kernel() noexcept override { return driver_; }
 private:
  bool* infrastructure_alive_;
  bool* safe_destruction_;
  KernelDriver driver_;
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

TEST(DeepSeekRankComputeLaneSetTest,
     OwnsDriversAcrossMoveAndReleasesEachLaneOnce) {
  auto plan = DeepSeekPipelinePlan::Create(1, true);
  ASSERT_TRUE(plan.ok());
  const auto stage = plan->rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2);
  ASSERT_TRUE(pager.ok());
  std::uint32_t transfer_destroyed = 0;
  std::uint32_t kernel_destroyed = 0;
  {
    auto lane_set = DeepSeekRankComputeLaneSet::Create(
        stage, 8, 32, *pager,
        std::make_unique<TransferOwner>(transfer_destroyed),
        std::make_unique<KernelOwner>(kernel_destroyed));
    ASSERT_TRUE(lane_set.ok()) << lane_set.status().message();
    DeepSeekRankComputeLaneSet moved(std::move(*lane_set));
    const DeepSeekPipelinePlanDescriptor drain{
        7, 1, DeepSeekPlanPhase::kDrain, 0, 1};
    ASSERT_TRUE(moved.bundle().bind_plan(drain, {}).ok());
    ASSERT_TRUE(moved.bundle().driver().launch(drain, stage).ok());
    EXPECT_EQ(moved.bundle().driver().poll().value(),
              DeepSeekStageComputeStatus::kSuccess);
    EXPECT_EQ(transfer_destroyed, 0U);
    EXPECT_EQ(kernel_destroyed, 0U);
  }
  EXPECT_EQ(transfer_destroyed, 1U);
  EXPECT_EQ(kernel_destroyed, 1U);
}

TEST(DeepSeekRankComputeLaneSetTest, RejectsMissingLaneOwners) {
  auto plan = DeepSeekPipelinePlan::Create(1, false);
  ASSERT_TRUE(plan.ok());
  const auto stage = plan->rank(0);
  auto pager = DeepSeekExpertPager::Create(stage.layers, 2, 2);
  ASSERT_TRUE(pager.ok());
  std::uint32_t destroyed = 0;
  EXPECT_FALSE(DeepSeekRankComputeLaneSet::Create(
      stage, 4, 16, *pager, nullptr,
      std::make_unique<KernelOwner>(destroyed)).ok());
  EXPECT_EQ(destroyed, 1U);
}

TEST(DeepSeekRankComputeLaneSetTest,
     ResidentLaneOwnsOnlyKernelAndHasNoPagerIdentity) {
  auto pipeline = DeepSeekPipelinePlan::Create(3, false).value();
  const auto stage = pipeline.rank(1);
  auto resident = DeepSeekResidentExpertBindings::Resolve(
      stage, resident_tensor).value();
  std::uint32_t kernel_destroyed = 0;
  {
    auto lanes = DeepSeekRankComputeLaneSet::CreateResident(
        stage, 4, 16, resident,
        std::make_unique<KernelOwner>(kernel_destroyed));
    ASSERT_TRUE(lanes.ok()) << lanes.status().message();
    EXPECT_EQ(lanes->pager_identity(), nullptr);
    EXPECT_EQ(kernel_destroyed, 0U);
  }
  EXPECT_EQ(kernel_destroyed, 1U);
}

TEST(DeepSeekRankComputeLaneSetTest,
     OwnedInfrastructureOutlivesBothBorrowingLanes) {
  auto plan = DeepSeekPipelinePlan::Create(1, false).value();
  auto pager = DeepSeekExpertPager::Create(plan.rank(0).layers, 2, 2).value();
  bool infrastructure_alive = false;
  bool transfer_destroyed_safely = false;
  bool kernel_destroyed_safely = false;
  {
    auto lane_set = DeepSeekRankComputeLaneSet::CreateOwnedInfrastructure(
        plan.rank(0), 4, 16, pager,
        std::make_unique<InfrastructureOwner>(infrastructure_alive),
        std::make_unique<BorrowingTransferOwner>(
            infrastructure_alive, transfer_destroyed_safely),
        std::make_unique<BorrowingKernelOwner>(
            infrastructure_alive, kernel_destroyed_safely));
    ASSERT_TRUE(lane_set.ok()) << lane_set.status().message();
    EXPECT_TRUE(infrastructure_alive);
    EXPECT_EQ(lane_set->bundle().attention_compute_stream(), 73U);
    EXPECT_EQ(lane_set->learned_router_device_resources(),
              reinterpret_cast<DeepSeekLearnedRouterDeviceResources*>(
                  0x5100));
    EXPECT_EQ(lane_set->learned_router_staging_pool(),
              reinterpret_cast<DeepSeekLearnedRouterStagingPool*>(0x5200));
    EXPECT_EQ(lane_set->dense_mhc_runtime_resources(),
              reinterpret_cast<DeepSeekDenseMhcRuntimeResources*>(0x5300));
    EXPECT_EQ(lane_set->attention_projection_device_resources(),
              reinterpret_cast<DeepSeekAttentionProjectionDeviceResources*>(
                  0x5400));
    EXPECT_EQ(lane_set->mhc_device_resources(),
              reinterpret_cast<DeepSeekMhcDeviceResources*>(0x5500));
  }
  EXPECT_TRUE(transfer_destroyed_safely);
  EXPECT_TRUE(kernel_destroyed_safely);
  EXPECT_FALSE(infrastructure_alive);
}

TEST(DeepSeekRankComputeLaneSetTest,
     OwnedResidentInfrastructureOutlivesBorrowingKernel) {
  auto pipeline = DeepSeekPipelinePlan::Create(3, false).value();
  const auto stage = pipeline.rank(1);
  auto resident = DeepSeekResidentExpertBindings::Resolve(
      stage, resident_tensor).value();
  bool infrastructure_alive = false;
  bool kernel_destroyed_safely = false;
  {
    auto lanes =
        DeepSeekRankComputeLaneSet::CreateOwnedResidentInfrastructure(
            stage, 4, 16,
            std::make_unique<InfrastructureOwner>(
                infrastructure_alive, &resident),
            std::make_unique<BorrowingKernelOwner>(
                infrastructure_alive, kernel_destroyed_safely));
    ASSERT_TRUE(lanes.ok()) << lanes.status().message();
    EXPECT_EQ(lanes->pager_identity(), nullptr);
    EXPECT_EQ(lanes->bundle().attention_compute_stream(), 73U);
  }
  EXPECT_TRUE(kernel_destroyed_safely);
  EXPECT_FALSE(infrastructure_alive);
}

TEST(DeepSeekRankComputeLaneSetTest,
     RejectsDsparkStageWithoutResidentExpertBindings) {
  auto plan = DeepSeekPipelinePlan::Create(1, true).value();
  auto pager = DeepSeekExpertPager::Create(plan.rank(0).layers, 2, 2).value();
  bool infrastructure_alive = false;
  std::uint32_t transfer_destroyed = 0;
  std::uint32_t kernel_destroyed = 0;
  auto lane_set = DeepSeekRankComputeLaneSet::CreateOwnedInfrastructure(
      plan.rank(0), 4, 16, pager,
      std::make_unique<InfrastructureOwner>(infrastructure_alive),
      std::make_unique<TransferOwner>(transfer_destroyed),
      std::make_unique<KernelOwner>(kernel_destroyed));
  ASSERT_FALSE(lane_set.ok());
  EXPECT_EQ(lane_set.status().code(), StatusCode::kInvalidArgument);
  EXPECT_FALSE(infrastructure_alive);
  EXPECT_EQ(transfer_destroyed, 1U);
  EXPECT_EQ(kernel_destroyed, 1U);
}

}  // namespace
}  // namespace pih
