#include "pih/model/deepseek_resident_expert_bindings.h"
#include "pih/model/deepseek_resident_subwave_executor.h"

#include <gtest/gtest.h>

#include <array>
#include <functional>
#include <string_view>
#include <vector>

namespace pih { namespace {

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

class ResidentKernel final : public DeepSeekExpertKernelDriver {
 public:
  Status launch(const DeepSeekExpertLease&, const DeepSeekExpertRoute*,
                std::uint32_t) override {
    paged_launch = true;
    return Status::Internal("paged launch is forbidden");
  }
  Status launch_resident(
      DeepSeekExpertIdentity identity, std::uint64_t generation,
      const DeepSeekExpertBundleDeviceView&, const DeepSeekExpertRoute*,
      std::uint32_t route_count) override {
    identities.push_back(identity);
    EXPECT_EQ(generation, 17U);
    EXPECT_EQ(route_count, 1U);
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> poll() override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
  bool paged_launch = false;
  std::vector<DeepSeekExpertIdentity> identities;
};

TEST(DeepSeekResidentExpertBindingsTest,
     ResolvesOwnedMainLayerAndExecutesWithoutPager) {
  auto bindings = DeepSeekResidentExpertBindings::Resolve(
      {0, {4, 4}, false, false, false}, resident_tensor);
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_EQ(bindings->generation(), 17U);
  EXPECT_NE(bindings->expert(4, 0).w1.packed.address, 0U);
  EXPECT_EQ(bindings->expert(4, 255).w2.scales.bytes,
            DeepSeekExpertBundleLayout::kScaleBytesPerMatrix);
  EXPECT_FALSE(bindings->contains(3));
  EXPECT_TRUE(bindings->contains(4));

  const std::vector<DeepSeekExpertRoute> routes{
      {11, 0, 0, 0.2F}, {3, 0, 1, 0.2F}, {19, 0, 2, 0.2F},
      {7, 0, 3, 0.2F}, {23, 0, 4, 0.1F}, {5, 0, 5, 0.1F}};
  auto plan = DeepSeekExpertSubwavePlan::Create(1, routes).value();
  auto executor = DeepSeekResidentSubwaveExecutor::Create(
      4, plan, *bindings).value();
  ResidentKernel kernel;
  for (int step = 0; step < 8 &&
       executor.state() != DeepSeekExpertSubwaveExecutorState::kComplete;
       ++step) {
    ASSERT_TRUE(executor.advance(kernel).ok());
  }
  EXPECT_FALSE(kernel.paged_launch);
  EXPECT_EQ(kernel.identities,
            (std::vector<DeepSeekExpertIdentity>{
                {4, 3}, {4, 5}, {4, 7}, {4, 11}, {4, 19}, {4, 23}}));
}

} }  // namespace pih::<anonymous>
