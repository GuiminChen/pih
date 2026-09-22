#include "pih/model/deepseek_attention_projection_device_resources.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>

namespace pih {
namespace {

class ProjectionDeviceAllocator final : public Allocator {
 public:
  explicit ProjectionDeviceAllocator(std::int32_t device) : device_(device) {}

  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocation_count_;
    requested_bytes_ = bytes;
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) {
      return Status::ResourceExhausted("projection device allocation");
    }
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }

  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }

  std::uint32_t allocation_count() const noexcept { return allocation_count_; }
  std::uint64_t requested_bytes() const noexcept { return requested_bytes_; }

 private:
  std::int32_t device_ = 0;
  std::uint64_t generation_ = 0;
  std::uint32_t allocation_count_ = 0;
  std::uint64_t requested_bytes_ = 0;
};

TEST(DeepSeekAttentionProjectionDeviceResourcesTest,
     OwnsOneAlignedBackingForEveryProjectionActivation) {
  ProjectionDeviceAllocator allocator(2);
  auto resources = DeepSeekAttentionProjectionDeviceResources::Allocate(
      allocator, 8, 17, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(allocator.allocation_count(), 1U);
  EXPECT_EQ(resources->backing_bytes(), allocator.requested_bytes());
  EXPECT_EQ(resources->maximum_tokens(), 8U);
  EXPECT_EQ(resources->context_identity(), 17U);
  EXPECT_EQ(resources->device_ordinal(), 2);

  const auto view = resources->view();
  const std::array addresses{
      view.input_e4m3,
      view.input_scale_ue8m0,
      view.q_a_bf16,
      view.q_norm_bf16,
      view.q_e4m3,
      view.q_scale_ue8m0,
      view.query_bf16,
      view.kv_bf16,
      view.attention_output_bf16,
      view.wo_a_activation_e4m3,
      view.wo_a_activation_scale_ue8m0,
      view.wo_a_output_bf16,
      view.wo_b_activation_e4m3,
      view.wo_b_activation_scale_ue8m0,
      view.branch_output_bf16,
      view.token_ids_u32,
      view.positions_u32,
      view.error_flag_u32,
  };
  for (const auto address : addresses) {
    EXPECT_NE(address, 0U);
    EXPECT_EQ(address % 256U, 0U);
  }
  auto sorted = addresses;
  std::ranges::sort(sorted);
  EXPECT_EQ(std::ranges::adjacent_find(sorted), sorted.end());
}

TEST(DeepSeekAttentionProjectionDeviceResourcesTest,
     RejectsInvalidCapacityAndAllocatorDevice) {
  ProjectionDeviceAllocator allocator(1);
  EXPECT_FALSE(DeepSeekAttentionProjectionDeviceResources::Allocate(
                   allocator, 0, 17, 1)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionProjectionDeviceResources::Allocate(
                   allocator, 4097, 17, 1)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionProjectionDeviceResources::Allocate(
                   allocator, 8, 0, 1)
                   .ok());
  EXPECT_FALSE(DeepSeekAttentionProjectionDeviceResources::Allocate(
                   allocator, 8, 17, 2)
                   .ok());
}

}  // namespace
}  // namespace pih
