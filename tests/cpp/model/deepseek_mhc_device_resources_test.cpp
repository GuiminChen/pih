#include "pih/model/deepseek_mhc_device_resources.h"

#include <gtest/gtest.h>

#include <array>

namespace pih {
namespace {

class MhcDeviceAllocator final : public Allocator {
 public:
  explicit MhcDeviceAllocator(std::int32_t device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocation_count_;
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("mHC device");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
  std::uint32_t allocation_count() const noexcept { return allocation_count_; }
 private:
  std::int32_t device_;
  std::uint64_t generation_ = 0;
  std::uint32_t allocation_count_ = 0;
};

TEST(DeepSeekMhcDeviceResourcesTest,
     AllocatesOneBackingForPingPongSequenceState) {
  MhcDeviceAllocator allocator(3);
  auto resources = DeepSeekMhcDeviceResources::Allocate(
      allocator, 8, 91, 3);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(allocator.allocation_count(), 1U);
  EXPECT_EQ(resources->maximum_tokens(), 8U);
  EXPECT_EQ(resources->context_identity(), 91U);
  EXPECT_EQ(resources->device_ordinal(), 3);
  EXPECT_GE(resources->backing_bytes(),
            8U * (2U * 4U * 4096U * sizeof(std::uint16_t) +
                  4096U * sizeof(std::uint16_t) +
                  4096U * sizeof(std::uint16_t) +
                  4U * sizeof(float) + 16U * sizeof(float)));
  const auto view = resources->view();
  const std::array addresses{
      view.residual_a_bf16, view.residual_b_bf16,
      view.layer_input_bf16, view.ffn_branch_output_bf16,
      view.post_mix_f32, view.residual_mix_f32};
  for (const auto address : addresses) {
    EXPECT_NE(address, 0U);
    EXPECT_EQ(address % 256U, 0U);
  }
  auto source = resources->create_boundary_send_source(
      view.residual_a_bf16, 3, 7001, 81);
  ASSERT_TRUE(source.ok()) << source.status().message();
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(source->data()),
            view.residual_a_bf16);
  EXPECT_EQ(source->wire_bytes(), 3U * 32768U);
  EXPECT_EQ(source->buffer_generation(), 1U);
  EXPECT_EQ(source->context_identity(), 91U);
  EXPECT_FALSE(resources->create_boundary_send_source(
      view.layer_input_bf16, 3, 7001, 81).ok());
  EXPECT_FALSE(resources->create_boundary_send_source(
      view.residual_b_bf16, 9, 7001, 81).ok());
}

TEST(DeepSeekMhcDeviceResourcesTest,
     RejectsInvalidIdentityAndWrongAllocatorDevice) {
  MhcDeviceAllocator allocator(1);
  EXPECT_FALSE(DeepSeekMhcDeviceResources::Allocate(allocator, 0, 91, 1).ok());
  EXPECT_FALSE(DeepSeekMhcDeviceResources::Allocate(allocator, 4097, 91, 1).ok());
  EXPECT_FALSE(DeepSeekMhcDeviceResources::Allocate(allocator, 8, 0, 1).ok());
  EXPECT_FALSE(DeepSeekMhcDeviceResources::Allocate(allocator, 8, 91, 2).ok());
}

}  // namespace
}  // namespace pih
