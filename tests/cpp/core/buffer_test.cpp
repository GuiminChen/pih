#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "pih/core/allocator.h"
#include "pih/core/buffer.h"

namespace pih {
namespace {

class RecordingAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    ++allocate_calls;
    last_bytes = bytes;
    last_alignment = alignment;
    if (fail) {
      return Status::ResourceExhausted("injected allocation failure");
    }
    storage.resize(static_cast<std::size_t>(bytes));
    return Allocation{storage.empty() ? nullptr : storage.data(), bytes,
                      alignment, next_generation++, Device::Cpu()};
  }

  void deallocate(Allocation allocation) noexcept override {
    ++deallocate_calls;
    released_generation = allocation.generation;
  }

  bool fail = false;
  int allocate_calls = 0;
  int deallocate_calls = 0;
  std::uint64_t last_bytes = 0;
  std::uint64_t last_alignment = 0;
  std::uint64_t next_generation = 41;
  std::uint64_t released_generation = 0;
  std::vector<std::uint8_t> storage;
};

TEST(BufferTest, ReleasesExactlyOnceThroughCreatingAllocator) {
  RecordingAllocator allocator;
  {
    auto result = Buffer::Allocate(allocator, 64, 16);
    ASSERT_TRUE(result.ok());
    Buffer buffer = std::move(result).value();
    EXPECT_EQ(buffer.size_bytes(), 64);
    EXPECT_EQ(buffer.generation(), 41);
    EXPECT_EQ(allocator.deallocate_calls, 0);
  }
  EXPECT_EQ(allocator.deallocate_calls, 1);
  EXPECT_EQ(allocator.released_generation, 41);
}

TEST(BufferTest, MoveClearsSourceAndDoesNotDoubleRelease) {
  RecordingAllocator allocator;
  {
    Buffer first = std::move(Buffer::Allocate(allocator, 32, 8)).value();
    Buffer second = std::move(first);
    EXPECT_EQ(first.data(), nullptr);
    EXPECT_EQ(first.size_bytes(), 0);
    EXPECT_EQ(first.generation(), 0);
    EXPECT_NE(second.data(), nullptr);
  }
  EXPECT_EQ(allocator.deallocate_calls, 1);
}

TEST(BufferTest, ZeroByteAllocationIsAValidOwnedAllocation) {
  RecordingAllocator allocator;
  {
    auto result = Buffer::Allocate(allocator, 0, 64);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->data(), nullptr);
    EXPECT_EQ(result->size_bytes(), 0);
    EXPECT_EQ(result->generation(), 41);
  }
  EXPECT_EQ(allocator.allocate_calls, 1);
  EXPECT_EQ(allocator.deallocate_calls, 1);
}

TEST(BufferTest, FailureIsAtomicAndDoesNotRelease) {
  RecordingAllocator allocator;
  allocator.fail = true;
  auto result = Buffer::Allocate(allocator, 128, 16);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
  EXPECT_EQ(allocator.deallocate_calls, 0);
}

TEST(BufferTest, RejectsInvalidAlignmentBeforeCallingAllocator) {
  RecordingAllocator allocator;
  auto result = Buffer::Allocate(allocator, 16, 3);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument);
  EXPECT_EQ(allocator.allocate_calls, 0);
}

TEST(BufferTest, ViewPreservesGenerationAndRejectsOutOfBoundsSpan) {
  RecordingAllocator allocator;
  Buffer buffer = std::move(Buffer::Allocate(allocator, 16, 16)).value();
  const std::int64_t shape[] = {4};
  auto view = buffer.view(DType::kFloat32, shape);
  ASSERT_TRUE(view.ok());
  EXPECT_EQ(view->byte_span(), 16);
  EXPECT_EQ(view->generation(), buffer.generation());

  const std::int64_t too_large[] = {5};
  auto rejected = buffer.view(DType::kFloat32, too_large);
  EXPECT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuAllocatorTest, ReturnsRequestedAlignmentAndUniqueGenerations) {
  CpuAllocator allocator;
  auto first = allocator.allocate(257, 256);
  ASSERT_TRUE(first.ok());
  ASSERT_NE(first->data, nullptr);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(first->data) % 256, 0);

  auto second = allocator.allocate(1, 16);
  ASSERT_TRUE(second.ok());
  EXPECT_NE(first->generation, second->generation);

  allocator.deallocate(second.value());
  allocator.deallocate(first.value());
}

TEST(CpuAllocatorTest, RejectsImpossibleSizesWithoutTruncation) {
  CpuAllocator allocator;
  auto result = allocator.allocate(std::numeric_limits<std::uint64_t>::max(), 64);
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
}

}  // namespace
}  // namespace pih
