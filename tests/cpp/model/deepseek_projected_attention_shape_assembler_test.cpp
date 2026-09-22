#include "pih/model/deepseek_projected_attention_shape_assembler.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class DeviceAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, 17,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
};

DeepSeekCompressorWeightBindings weights() {
  DeepSeekCompressorWeightBindings result;
  result.main_wkv_bf16 = 0x100000;
  result.main_wgate_bf16 = 0x180000;
  result.main_ape_f32 = 0x200000;
  result.main_norm_bf16 = 0x300000;
  result.indexer_wq_b_e4m3 = 0x400000;
  result.indexer_wq_b_scale_bits = 0x480000;
  result.indexer_weights_proj_bf16 = 0x500000;
  result.indexer_wkv_bf16 = 0x600000;
  result.indexer_wgate_bf16 = 0x680000;
  result.indexer_ape_f32 = 0x700000;
  result.indexer_norm_bf16 = 0x800000;
  result.generation = 19;
  return result;
}

TEST(DeepSeekProjectedAttentionShapeAssemblerTest,
     PopulatesChunkRowsScratchAndDeferredPageResources) {
  DeviceAllocator allocator;
  auto scratch = DeepSeekCompressorProjectionDeviceResources::Allocate(
      allocator, 2, 9, 0).value();
  auto attention_scratch = DeepSeekAttentionDeviceScratchResources::Allocate(
      allocator, 2, 16, 9, 0).value();
  DeepSeekAttentionLayerPlanShapeSeed shape;
  shape.layer = 7;
  shape.ratio = 4;
  shape.positions = {3, 4};
  shape.ratio = 4;
  shape.recent_physical_offset = 0;
  shape.compressed_physical_offset = 128;
  ASSERT_TRUE(DeepSeekProjectedAttentionShapeAssembler::Populate(
      shape, 0x10000, 0x20000, 0x30000, weights(), scratch,
      attention_scratch, 0x900000, 0xa00000, 0xb00000, 1048576).ok());
  ASSERT_EQ(shape.updates.size(), 2U);
  EXPECT_EQ(shape.updates[0].main_state.projection.input_bf16, 0x10000U);
  EXPECT_EQ(shape.updates[1].main_state.projection.input_bf16, 0x12000U);
  EXPECT_NE(shape.updates[0].main_state.kv_projection_f32,
            shape.updates[1].main_state.kv_projection_f32);
  EXPECT_TRUE(shape.bind_compressed_page_mutations);
  EXPECT_EQ(shape.main_rms_weight_bf16, 0x300000U);
  EXPECT_EQ(shape.index_rms_weight_bf16, 0x800000U);
  EXPECT_EQ(shape.main_cos_sin_cache_f32, 0xb00000U);
  EXPECT_TRUE(shape.has_indexer_projection);
  EXPECT_EQ(shape.indexer_projection.qr_bf16, 0x20000U);
  EXPECT_EQ(shape.indexer_projection.positions_u32, 0x30000U);
  EXPECT_EQ(shape.indexer_projection.query_bf16,
            attention_scratch.indexer_query_bf16());
  EXPECT_EQ(shape.indexer_projection.head_weight_f32,
            attention_scratch.indexer_head_weight_f32());
  EXPECT_EQ(shape.index_arena.score_f32,
            attention_scratch.index_arena().score_f32);
}

TEST(DeepSeekProjectedAttentionShapeAssemblerTest,
     RejectsCallerUpdatesAndScratchUndersizing) {
  DeviceAllocator allocator;
  auto scratch = DeepSeekCompressorProjectionDeviceResources::Allocate(
      allocator, 1, 9, 0).value();
  auto attention_scratch = DeepSeekAttentionDeviceScratchResources::Allocate(
      allocator, 1, 16, 9, 0).value();
  DeepSeekAttentionLayerPlanShapeSeed shape;
  shape.layer = 7;
  shape.ratio = 4;
  shape.positions = {3, 4};
  EXPECT_FALSE(DeepSeekProjectedAttentionShapeAssembler::Populate(
      shape, 0x10000, 0x20000, 0x30000, weights(), scratch,
      attention_scratch, 0x900000, 0xa00000, 0xb00000, 1048576).ok());
  shape.positions = {3};
  shape.updates.resize(1);
  EXPECT_FALSE(DeepSeekProjectedAttentionShapeAssembler::Populate(
      shape, 0x10000, 0x20000, 0x30000, weights(), scratch,
      attention_scratch, 0x900000, 0xa00000, 0xb00000, 1048576).ok());
}

TEST(DeepSeekProjectedAttentionShapeAssemblerTest,
     ReplacesUntrustedSparsePointersWithRuntimeOwnedBindings) {
  DeviceAllocator allocator;
  auto attention_scratch = DeepSeekAttentionDeviceScratchResources::Allocate(
      allocator, 2, 16, 9, 0).value();
  DeepSeekAttentionLayerPlanShapeSeed shape;
  shape.positions = {3, 4};
  shape.ratio = 4;
  shape.recent_physical_offset = 0;
  shape.compressed_physical_offset = 128;
  shape.sparse_query_bf16 = 1;
  shape.sparse_kv_bf16 = 2;
  shape.sparse_output_bf16 = 3;
  shape.attention_sink_f32 = 4;
  shape.sparse_indices_i32 = 5;
  shape.sparse_error_u32 = 6;
  ASSERT_TRUE(DeepSeekProjectedAttentionShapeAssembler::BindSparseRuntime(
      shape, 0x10000, 0x20000, 0x30000, 0x40000,
      attention_scratch).ok());
  EXPECT_EQ(shape.sparse_query_bf16, 0x10000U);
  EXPECT_EQ(shape.sparse_kv_bf16, 0x20000U);
  EXPECT_EQ(shape.sparse_output_bf16, 0x30000U);
  EXPECT_EQ(shape.attention_sink_f32, 0x40000U);
  EXPECT_EQ(shape.sparse_indices_i32,
            attention_scratch.sparse_indices_i32());
  EXPECT_EQ(shape.sparse_page_slots_u32,
            attention_scratch.sparse_page_slots_u32());
  EXPECT_EQ(shape.sparse_error_u32,
            attention_scratch.sparse_error_u32());
  EXPECT_EQ(shape.sparse_kv_count, 129U);
  EXPECT_FALSE(DeepSeekProjectedAttentionShapeAssembler::BindSparseRuntime(
      shape, 0, 0x20000, 0x30000, 0x40000,
      attention_scratch).ok());
}

} }  // namespace pih
