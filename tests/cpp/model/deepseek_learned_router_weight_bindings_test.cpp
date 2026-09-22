#include "pih/model/deepseek_learned_router_weight_bindings.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

Result<TensorView> router_weight(std::string_view name,
                                 DType dtype = DType::kBFloat16,
                                 std::array<std::int64_t, 2> shape = {256, 4096}) {
  if (name != "layers.3.ffn.gate.weight" &&
      name != "layers.4.ffn.gate.weight") {
    return Status::InvalidArgument("missing test tensor");
  }
  const auto address = static_cast<std::uintptr_t>(
      name[7] == '3' ? 0x10000 : 0x210000);
  return TensorView::Create(reinterpret_cast<void*>(address), dtype, shape, {},
      Device::Create(DeviceType::kCuda, 1).value(), 17);
}

TEST(DeepSeekLearnedRouterWeightBindingsTest,
     RejectsSyntheticDsparkLayerIdentity) {
  EXPECT_FALSE(DeepSeekLearnedRouterWeightBindings::Resolve(
      {43, 43}, [](std::string_view name) { return router_weight(name); }).ok());
}

TEST(DeepSeekLearnedRouterWeightBindingsTest,
     ResolvesExactOwnedLearnedLayersInOrder) {
  auto bindings = DeepSeekLearnedRouterWeightBindings::Resolve(
      {2, 4}, [](std::string_view name) { return router_weight(name); });
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  ASSERT_EQ(bindings->size(), 2U);
  EXPECT_EQ(bindings->at(0).layer, 3U);
  EXPECT_EQ(bindings->at(1).layer, 4U);
  EXPECT_EQ(bindings->at(0).weight_bf16, 0x10000U);
  EXPECT_EQ(bindings->generation(), 17U);
}

TEST(DeepSeekLearnedRouterWeightBindingsTest,
     RejectsCheckpointDtypeShapeAndGenerationDrift) {
  EXPECT_FALSE(DeepSeekLearnedRouterWeightBindings::Resolve(
      {3, 3}, [](std::string_view name) {
        return router_weight(name, DType::kFloat8E4M3);
      }).ok());
  EXPECT_FALSE(DeepSeekLearnedRouterWeightBindings::Resolve(
      {3, 3}, [](std::string_view name) {
        return router_weight(name, DType::kBFloat16, {4096, 256});
      }).ok());
  EXPECT_FALSE(DeepSeekLearnedRouterWeightBindings::Resolve(
      {3, 4}, [](std::string_view name) {
        auto value = router_weight(name);
        if (!value.ok() || name[7] == '3') return value;
        const std::array<std::int64_t, 2> shape{256, 4096};
        return TensorView::Create(value->data(), value->dtype(), shape, {},
                                  value->device(), 18);
      }).ok());
}

}  // namespace
}  // namespace pih
