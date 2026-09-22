#include "pih/model/deepseek_attention_runtime_resources.h"
#include "pih/model/deepseek_rank_attention_resources.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih {
namespace {

class RecentOps final : public DeepSeekRecentStateOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
};
class CompressorOps final : public DeepSeekCompressorStateOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status pooling(DeepSeekCompressorPoolingLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};
class PageOps final : public DeepSeekCompressedPageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status store(DeepSeekCompressorBf16StoreLaunch) override { return Status::Ok(); }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override { return Status::Ok(); }
};
class SelectionOps final : public DeepSeekIndexSelectionOperations {
 public:
  Status validate_host_staging(const DeepSeekIndexSelectionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status score(DeepSeekIndexScoreLaunch) override { return Status::Ok(); }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};
class ProjectionOps final : public DeepSeekIndexerProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status project(DeepSeekIndexerProjectionLaunch) override {
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t*, std::uintptr_t,
                              std::uintptr_t) override {
    return Status::Ok();
  }
};
class SparseOps final : public DeepSeekSparseAttentionOperations {
 public:
  Status validate_host_staging(const DeepSeekSparseAttentionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override { return Status::Ok(); }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status attention(DeepSeekSparseAttentionLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
};

class HostAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("host allocation");
    return Allocation{data, bytes, alignment, ++generation_, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

class CudaAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("device allocation");
    return Allocation{data, bytes, alignment, ++generation_,
                      Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekAttentionRuntimeResourcesTest,
     OwnsStableLayerGraphAcrossResourceSetMove) {
  RecentOps recent;
  CompressorOps compressor;
  PageOps page;
  ProjectionOps projection_ops;
  SelectionOps selection;
  SparseOps sparse;
  std::uint32_t compressor_error = 0;
  std::uint32_t page_error = 0;
  std::uint32_t projection_error = 0;
  std::uint32_t selection_error = 0;
  std::uint32_t sparse_error = 0;
  std::vector<float> scores(4096);
  std::vector<std::int32_t> indices(8320);
  auto layout = DeepSeekFixedStateLayout::Build(
      std::vector<std::uint32_t>{10}, false).value();
  auto arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 2U * 65536U}, {0x200000, 2U * 16384U},
      {0x300000, 2U * 65536U}, 2, 2).value();
  DeepSeekAttentionLayerRuntimeInput input{
      10, layout, arena, &compressor_error, &page_error,
      &projection_error,
      {scores.data(), &selection_error,
       static_cast<std::uint32_t>(scores.size())},
      19,
      {indices.data(), static_cast<std::uint32_t>(indices.size()),
       &sparse_error}};
  auto created = DeepSeekAttentionRuntimeResources::Create(
      {10, 10},
      {&recent, &compressor, &page, &projection_ops, &selection, &sparse},
      {std::move(input)});
  ASSERT_TRUE(created.ok()) << created.status().message();
  auto moved = std::move(*created);
  auto borrowed = moved.borrow_layer_resources();
  ASSERT_EQ(borrowed.size(), 1U);
  EXPECT_EQ(borrowed[0].layer, 10U);
  EXPECT_NE(borrowed[0].recent_writer, nullptr);
  EXPECT_NE(borrowed[0].update_coordinator, nullptr);
  EXPECT_NE(borrowed[0].attention_coordinator, nullptr);
  EXPECT_NE(borrowed[0].prefill_coordinator, nullptr);
  EXPECT_TRUE(DeepSeekAttentionWorkFactory::Create(
      {10, 10}, std::move(borrowed)).ok());
}

TEST(DeepSeekAttentionRuntimeResourcesTest, RejectsIncompleteOperations) {
  EXPECT_FALSE(DeepSeekAttentionRuntimeResources::Create(
      {10, 10}, {}, {}).ok());
}

TEST(DeepSeekAttentionRuntimeResourcesTest,
     AtomicallyAssemblesRankFactoryAndScratch) {
  RecentOps recent;
  CompressorOps compressor;
  PageOps page;
  ProjectionOps projection_ops;
  SelectionOps selection;
  SparseOps sparse;
  HostAllocator host;
  CudaAllocator device;
  auto layout = DeepSeekFixedStateLayout::Build(
      std::vector<std::uint32_t>{10}, false).value();
  auto arena = DeepSeekAttentionPageArena::Create(
      {0x100000, 2U * 65536U}, {0x200000, 2U * 16384U},
      {0x300000, 2U * 65536U}, 2, 2).value();
  auto resources = DeepSeekRankAttentionResources::Allocate(
      {10, 10}, 8, 2, layout, arena,
      {&recent, &compressor, &page, &projection_ops, &selection, &sparse},
      host, device, 19, 17, 0);
  ASSERT_TRUE(resources.ok()) << resources.status().message();
  EXPECT_EQ(resources->maximum_queries(), 8U);
  EXPECT_EQ(resources->work_factory().owned_layers().first_layer, 10U);
  EXPECT_NE(resources->device_scratch().index_arena().score_f32, 0U);
  auto projection = resources->compressor_projection().slice(0);
  ASSERT_TRUE(projection.ok());
  EXPECT_NE(projection->main_kv_f32, 0U);
  EXPECT_NE(projection->index_output_f32, 0U);
  auto moved = std::move(*resources);
  EXPECT_EQ(moved.work_factory().owned_layers().last_layer, 10U);
}

}  // namespace
}  // namespace pih
