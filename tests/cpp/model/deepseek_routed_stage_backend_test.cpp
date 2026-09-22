#include "pih/model/deepseek_routed_stage_backend.h"

#include <gtest/gtest.h>

namespace pih { namespace {

std::vector<DeepSeekExpertRoute> routes() {
  std::vector<DeepSeekExpertRoute> value;
  for (std::uint16_t expert : {1,3,5,7,9,11}) {
    value.push_back({expert,0,static_cast<std::uint8_t>(value.size()),1.0F/6.0F});
  }
  return value;
}

class Provider final : public DeepSeekExpertPlanProvider {
 public:
  explicit Provider(const DeepSeekExpertSubwavePlan& plan):plan_(&plan){}
  Result<const DeepSeekExpertSubwavePlan*> resolve(
      std::uint32_t layer,const DeepSeekPipelinePlanDescriptor& descriptor) override {
    resolved_layer=layer; resolved_sequence=descriptor.plan_sequence; return plan_;
  }
  const DeepSeekExpertSubwavePlan* plan_; std::uint32_t resolved_layer=99;
  std::uint64_t resolved_sequence=0;
};

class Dense final : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand& command,
                const DeepSeekPipelinePlanDescriptor&) override {
    launches.push_back(command); return Status::Ok();
  }
  Result<DeepSeekStageComputeStatus> poll() override { return status; }
  std::vector<DeepSeekStageOperatorCommand> launches;
  DeepSeekStageComputeStatus status=DeepSeekStageComputeStatus::kSuccess;
};

class Transfer final : public DeepSeekExpertTransferDriver {
 public:
  Status start(DeepSeekExpertIdentity,std::uint32_t,std::uint64_t,
               std::uint64_t) override { ++starts; return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll(DeepSeekExpertIdentity,
                                         std::uint64_t) override {
    return ready ? DeepSeekExpertAsyncStatus::kSuccess
                 : DeepSeekExpertAsyncStatus::kInProgress;
  }
  bool ready=false; int starts=0;
};

class Kernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease& lease,const DeepSeekExpertRoute*,
                std::uint32_t) override { launched.push_back(lease.identity.expert); return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  Status launch_resident(
      DeepSeekExpertIdentity identity, std::uint64_t,
      const DeepSeekExpertBundleDeviceView&, const DeepSeekExpertRoute*,
      std::uint32_t) override {
    resident.push_back(identity.expert);
    return Status::Ok();
  }
  std::vector<std::uint16_t> launched;
  std::vector<std::uint16_t> resident;
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

DeepSeekPipelinePlanDescriptor descriptor() {
  return {7,13,DeepSeekPlanPhase::kDecode,1,1};
}

TEST(DeepSeekRoutedStageBackendTest, WaitsForPagingBeforeLaunchingExpertCompute) {
  auto plan=DeepSeekExpertSubwavePlan::Create(1,routes());
  auto pager=DeepSeekExpertPager::Create({4,4},2,2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  Provider provider(*plan); Dense dense; Transfer transfer; Kernel kernel;
  auto backend=DeepSeekRoutedStageOperatorBackend::Create(
      dense,provider,*pager,transfer,kernel); ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend->launch({DeepSeekStageOperatorKind::kMoe,4},descriptor()).ok());
  EXPECT_TRUE(kernel.launched.empty());
  auto waiting=backend->poll(); ASSERT_TRUE(waiting.ok());
  EXPECT_EQ(*waiting,DeepSeekStageComputeStatus::kInProgress);
  EXPECT_TRUE(kernel.launched.empty());
  transfer.ready=true;
  auto launched=backend->poll(); ASSERT_TRUE(launched.ok());
  EXPECT_EQ(*launched,DeepSeekStageComputeStatus::kInProgress);
  ASSERT_EQ(kernel.launched.size(),1U);
  EXPECT_EQ(kernel.launched.front(),1U);
  EXPECT_EQ(provider.resolved_layer,4U);
  EXPECT_EQ(provider.resolved_sequence,13U);
}

TEST(DeepSeekRoutedStageBackendTest, CompletesAllExpertsAndReturnsToDenseMode) {
  auto plan=DeepSeekExpertSubwavePlan::Create(1,routes());
  auto pager=DeepSeekExpertPager::Create({4,4},2,2);
  ASSERT_TRUE(plan.ok()); ASSERT_TRUE(pager.ok());
  Provider provider(*plan); Dense dense; Transfer transfer; transfer.ready=true; Kernel kernel;
  auto backend=DeepSeekRoutedStageOperatorBackend::Create(
      dense,provider,*pager,transfer,kernel); ASSERT_TRUE(backend.ok());
  ASSERT_TRUE(backend->launch({DeepSeekStageOperatorKind::kMoe,4},descriptor()).ok());
  DeepSeekStageComputeStatus final=DeepSeekStageComputeStatus::kInProgress;
  for (int i=0;i<20 && final!=DeepSeekStageComputeStatus::kSuccess;++i) {
    auto status=backend->poll(); ASSERT_TRUE(status.ok()); final=*status;
  }
  EXPECT_EQ(final,DeepSeekStageComputeStatus::kSuccess);
  EXPECT_EQ(kernel.launched,(std::vector<std::uint16_t>{1,3,5,7,9,11}));
  ASSERT_TRUE(backend->launch(
      {DeepSeekStageOperatorKind::kAttention,4},descriptor()).ok());
  auto dense_status=backend->poll(); ASSERT_TRUE(dense_status.ok());
  EXPECT_EQ(*dense_status,DeepSeekStageComputeStatus::kSuccess);
  ASSERT_EQ(dense.launches.size(),1U);
}

TEST(DeepSeekRoutedStageBackendTest,
     RejectsSyntheticDsparkLayerIdentity) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, routes()).value();
  auto pager = DeepSeekExpertPager::Create({42,42},2,2).value();
  Provider provider(plan); Dense dense; Transfer transfer; Kernel kernel;
  auto backend = DeepSeekRoutedStageOperatorBackend::Create(
      dense, provider, pager, transfer, kernel).value();
  EXPECT_EQ(backend.launch(
      {DeepSeekStageOperatorKind::kMoe,43}, descriptor()).code(),
            StatusCode::kInvalidArgument);
  EXPECT_EQ(transfer.starts, 0);
  EXPECT_TRUE(kernel.launched.empty());
  EXPECT_TRUE(kernel.resident.empty());
}

TEST(DeepSeekRoutedStageBackendTest,
     MainLayerUsesResidentBundlesWithoutPagerOrTransfer) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, routes()).value();
  auto resident = DeepSeekResidentExpertBindings::Resolve(
      {0,{4,4},false,false,false}, resident_tensor).value();
  Provider provider(plan); Dense dense; Kernel kernel;
  auto backend = DeepSeekRoutedStageOperatorBackend::CreateResident(
      dense, provider, kernel, resident).value();
  ASSERT_TRUE(backend.launch(
      {DeepSeekStageOperatorKind::kMoe,4}, descriptor()).ok());
  DeepSeekStageComputeStatus final = DeepSeekStageComputeStatus::kInProgress;
  for (int step = 0; step < 16 && final != DeepSeekStageComputeStatus::kSuccess;
       ++step) final = backend.poll().value();
  EXPECT_EQ(final, DeepSeekStageComputeStatus::kSuccess);
  EXPECT_TRUE(kernel.launched.empty());
  EXPECT_EQ(kernel.resident,
            (std::vector<std::uint16_t>{1,3,5,7,9,11}));
}

} }  // namespace pih
