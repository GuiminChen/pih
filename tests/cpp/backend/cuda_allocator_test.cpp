#include <cstdint>
#include <memory>

#include <gtest/gtest.h>

#include "pih/backend/cuda/cuda_allocator.h"
#include "pih/core/buffer.h"

namespace pih {
namespace {

TEST(CudaAllocatorTest, RejectsInvalidDevice) {
  auto allocator = CudaAllocator::Create(-1);
  EXPECT_FALSE(allocator.ok());
  EXPECT_EQ(allocator.status().code(), StatusCode::kInvalidArgument);
}

TEST(CudaAllocatorTest, AllocatesOnExactDeviceAndTagsGenerations) {
  auto created = CudaAllocator::Create(0);
  if (!created.ok()) {
    GTEST_SKIP() << created.status().message();
  }
  std::unique_ptr<CudaAllocator> allocator = std::move(created).value();

  auto zero = allocator->allocate(0, 256);
  ASSERT_TRUE(zero.ok());
  EXPECT_EQ(zero->data, nullptr);
  EXPECT_EQ(zero->device, Device::Create(DeviceType::kCuda, 0).value());

  auto data = allocator->allocate(4096, 256);
  ASSERT_TRUE(data.ok());
  EXPECT_NE(data->data, nullptr);
  EXPECT_NE(data->generation, zero->generation);
  allocator->deallocate(data.value());
  allocator->deallocate(zero.value());
}

TEST(CudaAllocatorTest, BufferReleasesOnOwningDevice) {
  auto created = CudaAllocator::Create(0);
  if (!created.ok()) {
    GTEST_SKIP() << created.status().message();
  }
  std::unique_ptr<CudaAllocator> allocator = std::move(created).value();
  auto buffer = Buffer::Allocate(*allocator, 1024, 256);
  ASSERT_TRUE(buffer.ok());
  EXPECT_EQ(buffer->device().type(), DeviceType::kCuda);
}

}  // namespace
}  // namespace pih
