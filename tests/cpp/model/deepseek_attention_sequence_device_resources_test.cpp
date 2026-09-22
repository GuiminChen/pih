#include "pih/model/deepseek_attention_sequence_device_resources.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class StateAllocator final : public Allocator {
 public:
  explicit StateAllocator(std::int32_t device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("state allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, device_).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::int32_t device_ = 0;
  std::uint64_t generation_ = 0;
};

class FixedOperations final : public DeepSeekFixedStateBankOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekAttentionSequenceDeviceResourcesTest,
     OwnsSequenceScopedDoubleBanks) {
  StateAllocator allocator(2);
  FixedOperations operations;
  auto layout = DeepSeekFixedStateLayout::Build(
      std::vector<std::uint32_t>{10}, false).value();
  auto resources = DeepSeekAttentionSequenceDeviceResources::Allocate(
      allocator, layout, 19, operations, 17, 2);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->fixed_banks().bank_bytes(), layout.total_bytes());
  EXPECT_NE(resources->fixed_banks().committed_address(),
            resources->fixed_banks().tentative_address());
}

TEST(DeepSeekAttentionSequenceDeviceResourcesTest,
     RejectsAllocatorOnForeignDevice) {
  StateAllocator allocator(1);
  FixedOperations operations;
  auto layout = DeepSeekFixedStateLayout::Build(
      std::vector<std::uint32_t>{10}, false).value();
  EXPECT_FALSE(DeepSeekAttentionSequenceDeviceResources::Allocate(
      allocator, layout, 19, operations, 17, 2).ok());
}

}  // namespace
}  // namespace pih
