#include "pih/model/deepseek_request_input_staging_resources.h"

#include <gtest/gtest.h>

#include <vector>
#include <cstdlib>
#if !defined(_WIN32)
static void* _aligned_malloc(std::size_t bytes, std::size_t alignment) {
  void* result = nullptr;
  return posix_memalign(&result, alignment, bytes) == 0 ? result : nullptr;
}
static void _aligned_free(void* pointer) { std::free(pointer); }
#endif

namespace pih { namespace {

class RequestPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    return Allocation{data, bytes, alignment, 9, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
};

class RequestCopies final : public DeepSeekRequestInputCopyOperations {
 public:
  bool completion_proven = false;
  std::vector<std::uintptr_t> synchronized;
  Status synchronize(std::uintptr_t stream) override {
    synchronized.push_back(stream);
    return completion_proven ? Status::Ok() : Status::Unavailable("test completion pending");
  }
  Status copy_h2d_async(std::uintptr_t destination, const void* source,
                        std::uint64_t bytes,
                        std::uintptr_t stream) override {
    destinations.push_back(destination);
    streams.push_back(stream);
    const auto* values = static_cast<const std::uint32_t*>(source);
    payloads.emplace_back(values, values + bytes / sizeof(std::uint32_t));
    return Status::Ok();
  }
  std::vector<std::uintptr_t> destinations;
  std::vector<std::uintptr_t> streams;
  std::vector<std::vector<std::uint32_t>> payloads;
};

TEST(DeepSeekRequestInputStagingResourcesTest,
     StagesPinnedTokenAndAbsolutePositionCopiesUnderOneLease) {
  RequestPinnedAllocator allocator;
  auto staging = DeepSeekRequestInputStagingResources::Allocate(8, allocator);
  ASSERT_TRUE(staging.ok()) << staging.status().message();
  RequestCopies copies;
  copies.completion_proven = true;
  DeepSeekAttentionProjectionDeviceView destination{};
  destination.token_ids_u32 = 0x10000;
  destination.positions_u32 = 0x20000;
  const std::array<std::uint32_t, 3> tokens{7, 8, 9};
  const std::array<std::uint32_t, 3> positions{65534, 65535, 65536};
  auto lease = staging->stage(tokens, positions, destination, 0x30000, copies);
  ASSERT_TRUE(lease.ok()) << lease.status().message();
  EXPECT_EQ(copies.synchronized, std::vector<std::uintptr_t>({0x30000}));
  EXPECT_EQ(lease->token_count, 3U);
  EXPECT_EQ(copies.destinations,
            std::vector<std::uintptr_t>({0x10000, 0x20000}));
  EXPECT_EQ(copies.payloads[0],
            std::vector<std::uint32_t>(tokens.begin(), tokens.end()));
  EXPECT_EQ(copies.payloads[1],
            std::vector<std::uint32_t>(positions.begin(), positions.end()));
  EXPECT_FALSE(staging->stage(tokens, positions, destination, 0x30000, copies)
                   .ok());
  lease->owner.reset();
  EXPECT_TRUE(staging->stage(tokens, positions, destination, 0x30000, copies)
                  .ok());
}

TEST(DeepSeekRequestInputStagingResourcesTest,
     RejectsMismatchedOrOversizedInputBeforeCopy) {
  RequestPinnedAllocator allocator;
  auto staging = DeepSeekRequestInputStagingResources::Allocate(2, allocator);
  RequestCopies copies;
  copies.completion_proven = true;
  DeepSeekAttentionProjectionDeviceView destination{};
  destination.token_ids_u32 = 1;
  destination.positions_u32 = 2;
  const std::array<std::uint32_t, 3> three{1, 2, 3};
  const std::array<std::uint32_t, 2> two{1, 2};
  EXPECT_FALSE(staging->stage(three, three, destination, 3, copies).ok());
  EXPECT_FALSE(staging->stage(two, three, destination, 3, copies).ok());
  EXPECT_TRUE(copies.destinations.empty());
}

} }  // namespace pih::<anonymous>
