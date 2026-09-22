#include "pih/model/deepseek_dspark_resident_subwave_executor.h"

#include <gtest/gtest.h>

namespace pih { namespace {

Result<TensorView> tensor(std::string_view name) {
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

struct DsparkExpertIdentity final {
  DeepSeekDsparkStageId stage;
  std::uint16_t expert;
  bool operator==(const DsparkExpertIdentity&) const = default;
};

class Kernel final : public DeepSeekDsparkExpertKernelDriver {
 public:
  Status launch_resident(
      DeepSeekDsparkStageId stage, std::uint16_t expert,
      std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView&, const DeepSeekExpertRoute*,
      std::uint32_t route_count) override {
    identities.push_back({stage, expert});
    EXPECT_EQ(generation, 17U);
    EXPECT_EQ(route_count, 1U);
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return fail ? DeepSeekExpertAsyncStatus::kError
                : DeepSeekExpertAsyncStatus::kSuccess;
  }
  bool fail = false;
  std::vector<DsparkExpertIdentity> identities;
};

std::vector<DeepSeekExpertRoute> routes() {
  std::vector<DeepSeekExpertRoute> result;
  for (std::uint16_t expert : {11, 3, 9, 1, 7, 5}) {
    result.push_back({expert, 0, static_cast<std::uint8_t>(result.size()),
                      1.0F / 6.0F});
  }
  return result;
}

TEST(DeepSeekDsparkResidentSubwaveExecutorTest,
     ExecutesCanonicalExpertOrderWithoutPager) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, routes()).value();
  auto bindings = DeepSeekDsparkResidentExpertBindings::Resolve(
      {0, {0, 42}, true, true, true}, tensor).value();
  auto executor = DeepSeekDsparkResidentSubwaveExecutor::Create(
      DeepSeekDsparkStageId::kMtp1, plan, bindings).value();
  Kernel kernel;
  for (int step = 0; step < 16 &&
       executor.state() != DeepSeekExpertSubwaveExecutorState::kComplete;
       ++step) ASSERT_TRUE(executor.advance(kernel).ok());
  EXPECT_EQ(executor.completed_experts(), 6U);
  EXPECT_EQ(kernel.identities,
            (std::vector<DsparkExpertIdentity>{
                {DeepSeekDsparkStageId::kMtp1,1},
                {DeepSeekDsparkStageId::kMtp1,3},
                {DeepSeekDsparkStageId::kMtp1,5},
                {DeepSeekDsparkStageId::kMtp1,7},
                {DeepSeekDsparkStageId::kMtp1,9},
                {DeepSeekDsparkStageId::kMtp1,11}}));
}

TEST(DeepSeekDsparkResidentSubwaveExecutorTest, KernelFailureIsFailStop) {
  auto plan = DeepSeekExpertSubwavePlan::Create(1, routes()).value();
  auto bindings = DeepSeekDsparkResidentExpertBindings::Resolve(
      {0, {0, 42}, true, true, true}, tensor).value();
  auto executor = DeepSeekDsparkResidentSubwaveExecutor::Create(
      DeepSeekDsparkStageId::kMtp2, plan, bindings).value();
  Kernel kernel;
  ASSERT_TRUE(executor.advance(kernel).ok());
  kernel.fail = true;
  EXPECT_FALSE(executor.advance(kernel).ok());
  EXPECT_EQ(executor.state(), DeepSeekExpertSubwaveExecutorState::kPoisoned);
}

} }  // namespace pih
