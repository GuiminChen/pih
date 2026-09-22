#include "pih/model/deepseek_compressor_weight_bindings.h"

#include <gtest/gtest.h>

#include <map>

namespace pih { namespace {
using Entry = std::pair<DType, std::vector<std::int64_t>>;

Result<TensorView> compressor_tensor(std::string_view name) {
  const std::map<std::string, Entry, std::less<>> values{
      {"layers.7.attn.compressor.wkv.weight", {DType::kBFloat16, {1024, 4096}}},
      {"layers.7.attn.compressor.wgate.weight", {DType::kBFloat16, {1024, 4096}}},
      {"layers.7.attn.compressor.ape", {DType::kFloat32, {4, 1024}}},
      {"layers.7.attn.compressor.norm.weight", {DType::kBFloat16, {512}}},
      {"layers.7.attn.indexer.wq_b.weight",
       {DType::kFloat8E4M3, {8192, 1024}}},
      {"layers.7.attn.indexer.wq_b.scale",
       {DType::kFloat8E8M0, {64, 8}}},
      {"layers.7.attn.indexer.weights_proj.weight",
       {DType::kBFloat16, {64, 4096}}},
      {"layers.7.attn.indexer.compressor.wkv.weight", {DType::kBFloat16, {256, 4096}}},
      {"layers.7.attn.indexer.compressor.wgate.weight", {DType::kBFloat16, {256, 4096}}},
      {"layers.7.attn.indexer.compressor.ape",
       {DType::kFloat32, {4, 256}}},
      {"layers.7.attn.indexer.compressor.norm.weight",
       {DType::kBFloat16, {128}}},
      {"layers.11.attn.compressor.wkv.weight", {DType::kBFloat16, {512, 4096}}},
      {"layers.11.attn.compressor.wgate.weight", {DType::kBFloat16, {512, 4096}}},
      {"layers.11.attn.compressor.ape", {DType::kFloat32, {128, 512}}},
      {"layers.11.attn.compressor.norm.weight", {DType::kBFloat16, {512}}}};
  const auto found = values.find(name);
  if (found == values.end()) return Status::InvalidArgument("missing tensor");
  return TensorView::Create(
      reinterpret_cast<void*>(0x10000 +
          static_cast<std::uintptr_t>(std::distance(values.begin(), found)) *
              0x10000),
      found->second.first, found->second.second, {},
      Device::Create(DeviceType::kCuda, 0).value(), 73);
}

TEST(DeepSeekCompressorWeightBindingsTest,
     ResolvesRatio4MainAndIndexerCheckpointAbi) {
  auto value = DeepSeekCompressorWeightBindings::Resolve(
      7, DeepSeekCompressedAttentionKind::kRatio4, compressor_tensor);
  ASSERT_TRUE(value.ok()) << value.status().message();
  EXPECT_NE(value->main_wkv_bf16, 0U);
  EXPECT_NE(value->main_ape_f32, 0U);
  EXPECT_NE(value->indexer_wq_b_e4m3, 0U);
  EXPECT_NE(value->indexer_wq_b_scale_bits, 0U);
  EXPECT_NE(value->indexer_weights_proj_bf16, 0U);
  EXPECT_NE(value->indexer_wkv_bf16, 0U);
  EXPECT_EQ(value->generation, 73U);
}

TEST(DeepSeekCompressorWeightBindingsTest,
     ResolvesRatio128WithoutInventingIndexerWeights) {
  auto value = DeepSeekCompressorWeightBindings::Resolve(
      11, DeepSeekCompressedAttentionKind::kRatio128, compressor_tensor);
  ASSERT_TRUE(value.ok()) << value.status().message();
  EXPECT_NE(value->main_wkv_bf16, 0U);
  EXPECT_EQ(value->indexer_wq_b_e4m3, 0U);
  EXPECT_EQ(value->indexer_weights_proj_bf16, 0U);
}

TEST(DeepSeekCompressorWeightBindingsTest,
     RejectsShapeDtypeGenerationAndLayerDrift) {
  auto drift = [](std::string_view name) -> Result<TensorView> {
    auto value = compressor_tensor(name);
    if (!value.ok()) return value;
    if (name == "layers.7.attn.compressor.ape") {
      return TensorView::Create(value->data(), DType::kBFloat16,
          std::array<std::int64_t, 2>{4, 1024}, {}, value->device(), 73);
    }
    return value;
  };
  EXPECT_FALSE(DeepSeekCompressorWeightBindings::Resolve(
      7, DeepSeekCompressedAttentionKind::kRatio4, drift).ok());
  EXPECT_EQ(DeepSeekCompressorWeightBindings::Resolve(
      43, DeepSeekCompressedAttentionKind::kRatio4, compressor_tensor).status().code(),
      StatusCode::kInvalidArgument);
}

} }  // namespace pih
