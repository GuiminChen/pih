#include "pih/model/deepseek_mhc_weight_bindings.h"

#include <gtest/gtest.h>

#include <map>

namespace pih { namespace {

Result<TensorView> tensor(std::string_view name) {
  using Entry = std::pair<DType, std::vector<std::int64_t>>;
  const std::map<std::string, Entry, std::less<>> values{
      {"layers.7.attn_norm.weight", {DType::kBFloat16, {4096}}},
      {"layers.7.hc_attn_fn", {DType::kFloat32, {24, 16384}}},
      {"layers.7.hc_attn_scale", {DType::kFloat32, {3}}},
      {"layers.7.hc_attn_base", {DType::kFloat32, {24}}},
      {"layers.7.ffn_norm.weight", {DType::kBFloat16, {4096}}},
      {"layers.7.hc_ffn_fn", {DType::kFloat32, {24, 16384}}},
      {"layers.7.hc_ffn_scale", {DType::kFloat32, {3}}},
      {"layers.7.hc_ffn_base", {DType::kFloat32, {24}}}};
  std::string canonical(name);
  if (canonical.starts_with("mtp.") && canonical.size() > 6) {
    canonical.replace(0, std::string("mtp.0.").size(), "layers.7.");
  }
  const auto found = values.find(canonical);
  if (found == values.end()) return Status::InvalidArgument("missing tensor");
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000 +
          static_cast<std::uintptr_t>(std::distance(values.begin(), found)) *
              0x10000),
      found->second.first, found->second.second, {},
      Device::Create(DeviceType::kCuda, 0).value(), 51);
}

TEST(DeepSeekMhcWeightBindingsTest, ResolvesExactMainLayerCheckpointAbi) {
  auto bindings = DeepSeekMhcWeightBindings::Resolve(7, tensor);
  ASSERT_TRUE(bindings.ok()) << bindings.status().message();
  EXPECT_NE(bindings->attention_norm_bf16, 0U);
  EXPECT_NE(bindings->attention_fn_f32, 0U);
  EXPECT_NE(bindings->feed_forward_fn_f32, 0U);
  EXPECT_EQ(bindings->generation, 51U);
}

TEST(DeepSeekMhcWeightBindingsTest,
     ResolvesThreeExplicitMtpStagesAndRejectsSyntheticLayer43) {
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    auto stage = deepseek_dspark_stage_id(index);
    ASSERT_TRUE(stage.ok());
    EXPECT_TRUE(DeepSeekMhcWeightBindings::ResolveDspark(*stage, tensor).ok());
  }
  EXPECT_FALSE(DeepSeekMhcWeightBindings::Resolve(43, tensor).ok());
  EXPECT_EQ(DeepSeekMhcWeightBindings::Resolve(44, tensor).status().code(),
            StatusCode::kInvalidArgument);
}

TEST(DeepSeekMhcWeightBindingsTest, RejectsShapeDrift) {
  auto wrong = [](std::string_view name) -> Result<TensorView> {
    auto value = tensor(name);
    if (!value.ok() || name != "layers.7.hc_ffn_fn") return value;
    return TensorView::Create(value->data(), value->dtype(),
                              std::array<std::int64_t, 2>{16384, 24}, {},
                              value->device(), value->generation());
  };
  EXPECT_FALSE(DeepSeekMhcWeightBindings::Resolve(7, wrong).ok());
}

} }  // namespace pih
