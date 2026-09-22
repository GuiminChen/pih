#include "pih/model/deepseek_attention_weight_bindings.h"

#include <gtest/gtest.h>

#include <map>

namespace pih { namespace {
using Entry = std::pair<DType, std::vector<std::int64_t>>;

Result<TensorView> attention_tensor(std::string_view name) {
  const std::map<std::string, Entry, std::less<>> values{
      {"layers.9.attn.wq_a.weight", {DType::kFloat8E4M3, {1024, 4096}}},
      {"layers.9.attn.wq_a.scale", {DType::kFloat8E8M0, {8, 32}}},
      {"layers.9.attn.q_norm.weight", {DType::kBFloat16, {1024}}},
      {"layers.9.attn.wq_b.weight", {DType::kFloat8E4M3, {32768, 1024}}},
      {"layers.9.attn.wq_b.scale", {DType::kFloat8E8M0, {256, 8}}},
      {"layers.9.attn.wkv.weight", {DType::kFloat8E4M3, {512, 4096}}},
      {"layers.9.attn.wkv.scale", {DType::kFloat8E8M0, {4, 32}}},
      {"layers.9.attn.kv_norm.weight", {DType::kBFloat16, {512}}},
      {"layers.9.attn.attn_sink", {DType::kFloat32, {64}}},
      {"layers.9.attn.wo_a.weight", {DType::kFloat8E4M3, {8192, 4096}}},
      {"layers.9.attn.wo_a.scale", {DType::kFloat8E8M0, {64, 32}}},
      {"layers.9.attn.wo_b.weight", {DType::kFloat8E4M3, {4096, 8192}}},
      {"layers.9.attn.wo_b.scale", {DType::kFloat8E8M0, {32, 64}}}};
  std::string canonical(name);
  if (canonical.starts_with("mtp.") && canonical.size() > 11) {
    canonical.replace(0, std::string("mtp.0.attn.").size(),
                      "layers.9.attn.");
  }
  const auto found = values.find(canonical);
  if (found == values.end()) return Status::InvalidArgument("missing tensor");
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000 +
          static_cast<std::uintptr_t>(std::distance(values.begin(), found)) *
              0x10000),
      found->second.first, found->second.second, {},
      Device::Create(DeviceType::kCuda, 0).value(), 61);
}

TEST(DeepSeekAttentionWeightBindingsTest,
     ResolvesCheckpointNativeFp8ProjectionAbi) {
  auto value = DeepSeekAttentionWeightBindings::Resolve(9, attention_tensor);
  ASSERT_TRUE(value.ok()) << value.status().message();
  EXPECT_NE(value->wq_a_fp8, 0U);
  EXPECT_NE(value->wo_a_fp8, 0U);
  EXPECT_NE(value->wo_a_scale_ue8m0, 0U);
  EXPECT_NE(value->wo_b_fp8, 0U);
  EXPECT_EQ(value->generation, 61U);
}

TEST(DeepSeekAttentionWeightBindingsTest,
     ResolvesThreeExplicitMtpStagesAndRejectsSyntheticLayer43) {
  for (std::uint32_t index = 0; index < kDeepSeekDsparkStageCount; ++index) {
    auto stage = deepseek_dspark_stage_id(index);
    ASSERT_TRUE(stage.ok());
    EXPECT_TRUE(DeepSeekAttentionWeightBindings::ResolveDspark(
                    *stage, attention_tensor).ok());
  }
  EXPECT_FALSE(DeepSeekAttentionWeightBindings::Resolve(43, attention_tensor)
                   .ok());
  EXPECT_EQ(DeepSeekAttentionWeightBindings::Resolve(44, attention_tensor)
                .status().code(),
            StatusCode::kInvalidArgument);
}

TEST(DeepSeekAttentionWeightBindingsTest, RejectsScaleDtypeDrift) {
  auto drift = [](std::string_view name) -> Result<TensorView> {
    auto value = attention_tensor(name);
    if (!value.ok() || name != "layers.9.attn.wo_a.scale") return value;
    return TensorView::Create(value->data(), DType::kFloat32,
                              std::array<std::int64_t, 2>{64, 32}, {},
                              value->device(), value->generation());
  };
  EXPECT_FALSE(DeepSeekAttentionWeightBindings::Resolve(9, drift).ok());
}

} }  // namespace pih
