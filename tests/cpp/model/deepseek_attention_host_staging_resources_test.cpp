#include "pih/model/deepseek_attention_host_staging_resources.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace pih {
namespace {

class PinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("test pinned allocation");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekAttentionHostStagingResourcesTest,
     AllocatesExactDisjointPinnedViews) {
  PinnedAllocator allocator;
  auto resources = DeepSeekAttentionHostStagingResources::Allocate(
      allocator, 8, 16);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  const auto index = resources->index();
  const auto sparse = resources->sparse();
  EXPECT_EQ(index.score_capacity, 8U * 4096U);
  EXPECT_EQ(sparse.index_capacity, 8U * 8320U);
  EXPECT_NE(index.scores, nullptr);
  EXPECT_NE(sparse.indices, nullptr);
  EXPECT_NE(index.error_flag, sparse.error_flag);
  EXPECT_NE(index.page_slots, nullptr);
  EXPECT_EQ(index.page_slot_capacity, 16U);
  EXPECT_NE(resources->compressor_error(), resources->page_error());
  EXPECT_NE(resources->compressor_error(), index.error_flag);
  EXPECT_NE(resources->indexer_projection_error(), index.error_flag);
  EXPECT_NE(resources->indexer_projection_error(), sparse.error_flag);
  EXPECT_EQ(resources->maximum_queries(), 8U);
}

TEST(DeepSeekAttentionHostStagingResourcesTest, RejectsInvalidQueryCapacity) {
  PinnedAllocator allocator;
  EXPECT_FALSE(DeepSeekAttentionHostStagingResources::Allocate(
      allocator, 0, 16).ok());
  EXPECT_FALSE(DeepSeekAttentionHostStagingResources::Allocate(
      allocator, 4097, 16).ok());
  EXPECT_FALSE(DeepSeekAttentionHostStagingResources::Allocate(
      allocator, 8, 0).ok());
}

}  // namespace
}  // namespace pih
