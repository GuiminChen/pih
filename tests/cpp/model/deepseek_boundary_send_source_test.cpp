#include "pih/model/deepseek_boundary_send_source.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

class SendSourceAllocator final : public Allocator {
 public:
  explicit SendSourceAllocator(Device device) : device_(device) {}
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    void* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, 17, device_};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  Device device_;
};

TEST(DeepSeekBoundarySendSourceTest, BindsExactDirectSendExtent) {
  SendSourceAllocator allocator(
      Device::Create(DeviceType::kCuda, 1).value());
  auto buffer = Buffer::Allocate(allocator, 131072, 256);
  ASSERT_TRUE(buffer.ok());
  auto source = DeepSeekBoundarySendSource::Create(
      *buffer, 4096, 65536, 2, 9, 77, 5);
  ASSERT_TRUE(source.ok()) << source.status().message();
  EXPECT_EQ(source->data(), static_cast<std::byte*>(buffer->data()) + 4096);
  EXPECT_EQ(source->wire_bytes(), 65536U);
  EXPECT_EQ(source->token_count(), 2U);
  EXPECT_EQ(source->buffer_generation(), 17U);
  EXPECT_EQ(source->producer_completion_generation(), 5U);
}

TEST(DeepSeekBoundarySendSourceTest, RejectsWrongBytesAndRangeOverflow) {
  SendSourceAllocator allocator(
      Device::Create(DeviceType::kCuda, 0).value());
  auto buffer = Buffer::Allocate(allocator, 65536, 256).value();
  EXPECT_FALSE(DeepSeekBoundarySendSource::Create(
      buffer, 0, 32768, 2, 9, 77, 5).ok());
  EXPECT_FALSE(DeepSeekBoundarySendSource::Create(
      buffer, 32768, 65536, 2, 9, 77, 5).ok());
}

TEST(DeepSeekBoundarySendSourceTest, RejectsCpuAndZeroIdentity) {
  SendSourceAllocator allocator(Device::Cpu());
  auto buffer = Buffer::Allocate(allocator, 32768, 256).value();
  EXPECT_FALSE(DeepSeekBoundarySendSource::Create(
      buffer, 0, 32768, 1, 9, 77, 5).ok());
}

}  // namespace
}  // namespace pih
