#include "pih/model/deepseek_attention_device_scratch_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class DeviceAllocator final : public Allocator {
 public:
  explicit DeviceAllocator(std::int32_t device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test device allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  std::int32_t device_ = 0;
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekAttentionDeviceScratchResourcesTest,
     AllocatesExactDisjointDeviceViews) {
  DeviceAllocator allocator(2);
  auto resources = DeepSeekAttentionDeviceScratchResources::Allocate(
      allocator, 8, 16, 17, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_NE(resources->index_arena().score_f32, 0U);
  EXPECT_NE(resources->index_arena().page_slots_u32, 0U);
  EXPECT_EQ(resources->index_arena().page_slot_capacity, 16U);
  EXPECT_NE(resources->indexer_query_bf16(), 0U);
  EXPECT_NE(resources->indexer_qr_e4m3(), 0U);
  EXPECT_NE(resources->indexer_qr_scale_bits(), 0U);
  EXPECT_NE(resources->indexer_qr_e4m3(), resources->indexer_qr_scale_bits());
  EXPECT_NE(resources->indexer_qr_e4m3(), resources->indexer_query_bf16());
  EXPECT_NE(resources->indexer_head_weight_f32(), 0U);
  EXPECT_NE(resources->indexer_query_bf16(),
            resources->index_arena().score_f32);
  EXPECT_NE(resources->sparse_indices_i32(), 0U);
  EXPECT_EQ(resources->sparse_page_slots_u32(),
            resources->index_arena().page_slots_u32);
  EXPECT_NE(resources->compressor_error_u32(), resources->page_error_u32());
  EXPECT_NE(resources->index_error_u32(), resources->sparse_error_u32());
  EXPECT_NE(resources->indexer_projection_error_u32(),
            resources->index_error_u32());
  EXPECT_NE(resources->indexer_projection_error_u32(),
            resources->sparse_error_u32());
  EXPECT_EQ(resources->maximum_queries(), 8U);
}

TEST(DeepSeekAttentionDeviceScratchResourcesTest,
     RejectsWrongAllocatorDeviceIdentity) {
  DeviceAllocator allocator(1);
  EXPECT_FALSE(DeepSeekAttentionDeviceScratchResources::Allocate(
      allocator, 8, 16, 17, 2).ok());
}

}  // namespace
}  // namespace pih
