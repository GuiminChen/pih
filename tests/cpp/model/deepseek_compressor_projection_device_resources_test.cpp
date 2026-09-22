#include "pih/model/deepseek_compressor_projection_device_resources.h"

#include <gtest/gtest.h>

namespace pih { namespace {

class ProjectionDeviceAllocator final : public Allocator {
 public:
  explicit ProjectionDeviceAllocator(std::int32_t device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test allocation");
    return Allocation{data, bytes, alignment, 91,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  std::int32_t device_ = 0;
};

TEST(DeepSeekCompressorProjectionDeviceResourcesTest,
     ProvidesDisjointMainIndexerAndQuerySlices) {
  ProjectionDeviceAllocator allocator(2);
  auto resources = DeepSeekCompressorProjectionDeviceResources::Allocate(
      allocator, 8, 37, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  auto first = resources->slice(0);
  auto second = resources->slice(1);
  ASSERT_TRUE(first.ok());
  ASSERT_TRUE(second.ok());
  EXPECT_NE(first->main_kv_f32, first->main_gate_f32);
  EXPECT_NE(first->main_output_f32, first->index_kv_f32);
  EXPECT_NE(first->index_kv_f32, first->index_gate_f32);
  EXPECT_NE(first->index_output_f32, first->index_weights_f32);
  EXPECT_NE(first->main_kv_f32, second->main_kv_f32);
  EXPECT_EQ(resources->maximum_queries(), 8U);
}

TEST(DeepSeekCompressorProjectionDeviceResourcesTest,
     RejectsForeignDeviceAndOutOfRangeQuery) {
  ProjectionDeviceAllocator allocator(1);
  EXPECT_FALSE(DeepSeekCompressorProjectionDeviceResources::Allocate(
      allocator, 8, 37, 2).ok());
  ProjectionDeviceAllocator valid_allocator(2);
  auto resources = DeepSeekCompressorProjectionDeviceResources::Allocate(
      valid_allocator, 8, 37, 2).value();
  EXPECT_EQ(resources.slice(8).status().code(), StatusCode::kInvalidArgument);
}

} }  // namespace pih
