#include "pih/model/deepseek_hash_router_weight_bindings.h"

#include <gtest/gtest.h>

#include <array>

namespace pih { namespace {

Result<TensorView> hash_weight(
    std::string_view name, DType dtype = DType::kBFloat16,
    std::array<std::int64_t, 2> shape = {256, 4096},
    std::uint64_t generation = 17) {
  if (name != "layers.0.ffn.gate.weight" &&
      name != "layers.1.ffn.gate.weight" &&
      name != "layers.2.ffn.gate.weight") {
    return Status::InvalidArgument("missing hash-router test tensor");
  }
  const auto layer = static_cast<std::uintptr_t>(name[7] - '0');
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000U + layer * 0x200000U), dtype, shape, {},
      Device::Create(DeviceType::kCuda, 1).value(), generation);
}

TEST(DeepSeekHashRouterWeightBindingsTest,
     ResolvesOnlyOwnedHashLayersInCanonicalOrder) {
  auto bindings = DeepSeekHashRouterWeightBindings::Resolve(
      {1, 4}, [](std::string_view name) { return hash_weight(name); });
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  ASSERT_EQ(bindings->size(), 2U);
  EXPECT_EQ(bindings->at(0).layer, 1U);
  EXPECT_EQ(bindings->at(1).layer, 2U);
  EXPECT_EQ(bindings->generation(), 17U);

  auto learned_only = DeepSeekHashRouterWeightBindings::Resolve(
      {3, 9}, [](std::string_view name) { return hash_weight(name); });
  ASSERT_TRUE(learned_only.ok());
  EXPECT_EQ(learned_only->size(), 0U);
}

TEST(DeepSeekHashRouterWeightBindingsTest,
     RejectsDtypeShapeAndGenerationDrift) {
  EXPECT_FALSE(DeepSeekHashRouterWeightBindings::Resolve(
      {0, 0}, [](std::string_view name) {
        return hash_weight(name, DType::kFloat8E4M3);
      }).ok());
  EXPECT_FALSE(DeepSeekHashRouterWeightBindings::Resolve(
      {0, 0}, [](std::string_view name) {
        return hash_weight(name, DType::kBFloat16, {4096, 256});
      }).ok());
  EXPECT_FALSE(DeepSeekHashRouterWeightBindings::Resolve(
      {0, 1}, [](std::string_view name) {
        return hash_weight(name, DType::kBFloat16, {256, 4096},
                           name[7] == '0' ? 17 : 18);
      }).ok());
}

}}  // namespace pih::<anonymous>
