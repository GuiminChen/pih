#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "pih/backend/cuda/cuda_allocator.h"
#include "pih/backend/cuda/cuda_memory_copier.h"

namespace pih {
namespace {

TEST(CudaMemoryCopierTest, RejectsInvalidDeviceAndPointerRoutes) {
  EXPECT_FALSE(CudaMemoryCopier::Create(-1).ok());
  auto copier = CudaMemoryCopier::Create(0);
  ASSERT_TRUE(copier.ok());
  std::array<std::byte, 1> source{};
  EXPECT_FALSE((*copier)->copy(source.data(), Device::Cpu(), source.data(),
                               Device::Cpu(), source.size()).ok());
  const auto cuda_one = Device::Create(DeviceType::kCuda, 1);
  ASSERT_TRUE(cuda_one.ok());
  EXPECT_FALSE((*copier)->copy(source.data(), cuda_one.value(), source.data(),
                               Device::Cpu(), source.size()).ok());
}

TEST(CudaMemoryCopierTest, CopiesSynchronouslyToExactDeviceAndRestoresCallerDevice) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  ASSERT_GT(count, 0);
  const int target = count - 1;
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto allocator_result = CudaAllocator::Create(target);
  auto copier_result = CudaMemoryCopier::Create(target);
  ASSERT_TRUE(allocator_result.ok());
  ASSERT_TRUE(copier_result.ok());
  auto allocation = (*allocator_result)->allocate(4, 256);
  ASSERT_TRUE(allocation.ok());
  const std::array<std::uint8_t, 4> source{1, 3, 5, 7};
  ASSERT_TRUE((*copier_result)
                  ->copy(allocation->data, allocation->device, source.data(),
                         Device::Cpu(), source.size())
                  .ok());

  int current = -1;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  EXPECT_EQ(current, 0);
  std::array<std::uint8_t, 4> round_trip{};
  ASSERT_EQ(cudaSetDevice(target), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(round_trip.data(), allocation->data, round_trip.size(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_EQ(round_trip, source);
  (*allocator_result)->deallocate(allocation.value());
}

}  // namespace
}  // namespace pih
