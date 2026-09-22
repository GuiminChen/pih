#include <array>
#include <cstdint>

#include <gtest/gtest.h>

#include "pih/model/qwen3_int4_weight_resource_set.h"

namespace pih {
namespace {

TensorView pool(std::uintptr_t address, std::uint64_t bytes,
                std::uint64_t generation, std::int32_t device = 0) {
  const std::array<std::int64_t,1> shape{static_cast<std::int64_t>(bytes)};
  return TensorView::Create(reinterpret_cast<void*>(address),DType::kUInt8,
      shape,{},Device::Create(DeviceType::kCuda,device).value(),generation)
      .value();
}

TensorView cpu_pool(std::uintptr_t address, std::uint64_t bytes,
                    std::uint64_t generation) {
  const std::array<std::int64_t,1> shape{static_cast<std::int64_t>(bytes)};
  return TensorView::Create(reinterpret_cast<void*>(address),DType::kUInt8,
      shape,{},Device::Create(DeviceType::kCpu,0).value(),generation).value();
}

TEST(QwenInt4WeightResourceSetTest, MapsCompactPoolAndTiedAliasExactly) {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  auto resources=QwenInt4WeightResourceSet::Create(
      layout,ledger,pool(0x100000000ULL,layout.logical_payload_bytes(),17),3);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->physical_payload_count(),506U);
  EXPECT_EQ(resources->logical_record_count(),507U);
  EXPECT_EQ(resources->resident_bytes(),538'378'240U);
  auto embedding=resources->view("model.embed_tokens.weight");
  auto head=resources->view("lm_head.weight");
  ASSERT_TRUE(embedding.ok()); ASSERT_TRUE(head.ok());
  EXPECT_EQ(embedding->data(),reinterpret_cast<void*>(0x100000000ULL));
  EXPECT_EQ(head->data(),embedding->data());
  EXPECT_EQ(embedding->dtype(),DType::kBFloat16);
  EXPECT_EQ(embedding->dim(0),151936U); EXPECT_EQ(embedding->dim(1),1024U);
  auto first_norm=resources->view("model.layers.0.input_layernorm.weight");
  ASSERT_TRUE(first_norm.ok());
  EXPECT_EQ(first_norm->data(),reinterpret_cast<void*>(
      0x100000000ULL+311'164'928ULL));
  EXPECT_EQ(first_norm->dtype(),DType::kBFloat16);
}

TEST(QwenInt4WeightResourceSetTest, BindsLinearValuesAndScalesBySourceName) {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  auto resources=QwenInt4WeightResourceSet::Create(
      layout,ledger,pool(0x100000000ULL,layout.logical_payload_bytes(),17),0)
      .value();
  const auto& expected=ledger.records().front();
  auto linear=resources.linear(expected.source_name);
  ASSERT_TRUE(linear.ok()) << linear.status().message();
  EXPECT_EQ(linear->family,expected.family);
  EXPECT_EQ(linear->packed.dtype(),DType::kUInt8);
  EXPECT_EQ(linear->packed.dim(0),expected.rows);
  EXPECT_EQ(linear->packed.dim(1),expected.columns/2);
  EXPECT_EQ(linear->scales.dtype(),DType::kFloat16);
  EXPECT_EQ(linear->scales.dim(0),expected.rows);
  EXPECT_EQ(linear->scales.dim(1),expected.groups_per_row);
  EXPECT_EQ(linear->packed.generation(),17U);
}

TEST(QwenInt4WeightResourceSetTest, RejectsPoolExtentDeviceAndGenerationDrift) {
  auto layout=QwenInt4ArtifactLayout::CreateOfficialPureW4().value();
  auto ledger=QwenInt4LinearShapeLedger::CreateOfficial().value();
  EXPECT_FALSE(QwenInt4WeightResourceSet::Create(
      layout,ledger,pool(0x100000000ULL,layout.logical_payload_bytes()-1,17),0)
      .ok());
  EXPECT_FALSE(QwenInt4WeightResourceSet::Create(
      layout,ledger,pool(0x100000000ULL,layout.logical_payload_bytes(),0),0)
      .ok());
  EXPECT_FALSE(QwenInt4WeightResourceSet::Create(
      layout,ledger,cpu_pool(0x100000000ULL,layout.logical_payload_bytes(),17),0)
      .ok());
}

}  // namespace
}  // namespace pih
