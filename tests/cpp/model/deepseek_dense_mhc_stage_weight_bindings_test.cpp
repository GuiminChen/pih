#include "pih/model/deepseek_dense_mhc_stage_weight_bindings.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace pih {
namespace {

Result<TensorView> stage_tensor(std::string_view name,
                                bool drift_last = false) {
  DType dtype = DType::kBFloat16;
  std::vector<std::int64_t> shape;
  if (name.ends_with("wq_a.weight")) {
    dtype = DType::kFloat8E4M3; shape = {1024, 4096};
  } else if (name.ends_with("wq_a.scale")) {
    dtype = DType::kFloat8E8M0; shape = {8, 32};
  } else if (name.ends_with("q_norm.weight")) {
    shape = {1024};
  } else if (name.ends_with("wq_b.weight")) {
    dtype = DType::kFloat8E4M3; shape = {32768, 1024};
  } else if (name.ends_with("wq_b.scale")) {
    dtype = DType::kFloat8E8M0; shape = {256, 8};
  } else if (name.ends_with("wkv.weight")) {
    dtype = DType::kFloat8E4M3; shape = {512, 4096};
  } else if (name.ends_with("wkv.scale")) {
    dtype = DType::kFloat8E8M0; shape = {4, 32};
  } else if (name.ends_with("kv_norm.weight")) {
    shape = {512};
  } else if (name.ends_with("attn_sink")) {
    dtype = DType::kFloat32; shape = {64};
  } else if (name.ends_with("wo_a.weight")) {
    dtype = DType::kFloat8E4M3; shape = {8192, 4096};
  } else if (name.ends_with("wo_a.scale")) {
    dtype = DType::kFloat8E8M0; shape = {64, 32};
  } else if (name.ends_with("wo_b.weight")) {
    dtype = DType::kFloat8E4M3; shape = {4096, 8192};
  } else if (name.ends_with("wo_b.scale")) {
    dtype = DType::kFloat8E8M0; shape = {32, 64};
  } else if (name.ends_with("attn_norm.weight") ||
             name.ends_with("ffn_norm.weight")) {
    shape = {4096};
  } else if (name.ends_with("hc_attn_fn") ||
             name.ends_with("hc_ffn_fn")) {
    dtype = DType::kFloat32; shape = {24, 16384};
  } else if (name.ends_with("hc_attn_scale") ||
             name.ends_with("hc_ffn_scale")) {
    dtype = DType::kFloat32; shape = {3};
  } else if (name.ends_with("hc_attn_base") ||
             name.ends_with("hc_ffn_base")) {
    dtype = DType::kFloat32; shape = {24};
  } else {
    return Status::InvalidArgument("unexpected dense mHC tensor");
  }
  const auto generation =
      drift_last && name == "layers.42.hc_ffn_base" ? 92U : 91U;
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000 + name.size() * 0x100), dtype,
      shape, {}, Device::Create(DeviceType::kCuda, 2).value(), generation);
}

TEST(DeepSeekDenseMhcStageWeightBindingsTest,
     ResolvesOnlyMainLayersInOneSealedGeneration) {
  std::vector<std::string> requested;
  auto result = DeepSeekDenseMhcStageWeightBindings::Resolve(
      {41, 42}, [&requested](std::string_view name) {
        requested.emplace_back(name);
        return stage_tensor(name);
      });
  ASSERT_TRUE(result.ok()) << result.status().message();
  ASSERT_EQ(result->layers().size(), 2U);
  EXPECT_EQ(result->layers()[0].layer, 41U);
  EXPECT_EQ(result->layers()[1].layer, 42U);
  EXPECT_EQ(result->generation(), 91U);
  EXPECT_NE(std::find(requested.begin(), requested.end(),
                      "layers.42.attn.wq_a.weight"), requested.end());
  EXPECT_NE(std::find(requested.begin(), requested.end(),
                      "layers.41.attn.wq_a.weight"), requested.end());
  EXPECT_EQ(std::ranges::count_if(requested, [](const auto& name) {
              return name.starts_with("mtp.");
            }), 0);
}

TEST(DeepSeekDenseMhcStageWeightBindingsTest,
     RejectsCoverageAndCrossTensorGenerationDrift) {
  EXPECT_FALSE(DeepSeekDenseMhcStageWeightBindings::Resolve(
                   {43, 43}, [](std::string_view name) {
                     return stage_tensor(name);
                   })
                   .ok());
  EXPECT_FALSE(DeepSeekDenseMhcStageWeightBindings::Resolve(
                   {41, 42}, [](std::string_view name) {
                     return stage_tensor(name, true);
                   })
                   .ok());
}

}  // namespace
}  // namespace pih
