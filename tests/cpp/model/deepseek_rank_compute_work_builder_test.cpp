#include "pih/model/deepseek_rank_compute_work_builder.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace pih {
namespace {

static_assert(!std::is_default_constructible_v<
              DeepSeekRankComputePlanWorkOwner>);
static_assert(!std::is_copy_constructible_v<
              DeepSeekRankComputePlanWorkOwner>);

template <typename T>
T* fake_resource(std::uintptr_t address) {
  return reinterpret_cast<T*>(address);
}

class BuilderFixedOperations final : public DeepSeekFixedStateBankOperations {
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

class BuilderPinnedAllocator final : public RegisteredPinnedAllocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(static_cast<std::size_t>(bytes),
                                 static_cast<std::size_t>(alignment));
    if (data == nullptr) return Status::ResourceExhausted("builder staging");
    return Allocation{data, bytes, alignment, 1, Device::Cpu()};
  }
  void deallocate(Allocation value) noexcept override {
    _aligned_free(value.data);
  }
};

class BuilderRouterOperations final : public DeepSeekLearnedRouterOperations {
 public:
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status gemm(DeepSeekRouterBf16GemmLaunch) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::uint64_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return DeepSeekExpertAsyncStatus::kSuccess;
  }
};

TEST(DeepSeekRankComputeWorkBuilderTest,
     RetainsAdditionalLifetimeBackingUntilWorkOwnerDies) {
  auto released = std::make_shared<bool>(false);
  auto backing = std::shared_ptr<const void>(
      new int(7), [released](const void* value) {
        delete static_cast<const int*>(value);
        *released = true;
      });
  DeepSeekRankComputePlanWork work;
  {
    DeepSeekRankComputeWorkBuilder builder;
    auto scratch = DeepSeekRouteScratchArena::Create(1).value();
    ASSERT_TRUE(builder.add_hash_router(
        0, {0},
        std::vector<float>(DeepSeekExpertSubwavePlan::kExpertCount, 0.0F), 1,
        std::vector<std::uint16_t>(
            DeepSeekExpertSubwavePlan::kRoutesPerToken, 0),
        &scratch).ok());
    ASSERT_TRUE(builder.own_lifetime_backing(backing).ok());
    EXPECT_FALSE(builder.own_lifetime_backing(nullptr).ok());
    backing.reset();
    work = std::move(builder).finish().value();
    EXPECT_FALSE(*released);
  }
  EXPECT_FALSE(*released);
  work.lifetime_owner.reset();
  EXPECT_TRUE(*released);
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     OwnsHashRouterTensorBackingAfterCallerStorageDies) {
  auto scratch = DeepSeekRouteScratchArena::Create(2);
  ASSERT_TRUE(scratch.ok());

  DeepSeekRankComputePlanWork work;
  {
    DeepSeekRankComputeWorkBuilder builder;
    std::vector<std::uint32_t> token_ids{4, 7};
    std::vector<float> scores(
        2 * DeepSeekExpertSubwavePlan::kExpertCount, 0.25F);
    std::vector<std::uint16_t> table(
        8 * DeepSeekExpertSubwavePlan::kRoutesPerToken);
    for (std::size_t index = 0; index < table.size(); ++index) {
      table[index] = static_cast<std::uint16_t>(
          index % DeepSeekExpertSubwavePlan::kExpertCount);
    }
    ASSERT_TRUE(builder.add_hash_router(
        0, std::move(token_ids), std::move(scores), 8,
        std::move(table), &*scratch).ok());
    auto finished = std::move(builder).finish();
    ASSERT_TRUE(finished.ok()) << finished.status().message();
    work = std::move(*finished);
  }

  ASSERT_NE(work.lifetime_owner, nullptr);
  ASSERT_EQ(work.hash_router.size(), 1U);
  EXPECT_EQ(work.hash_router[0].token_ids[0], 4U);
  EXPECT_EQ(work.hash_router[0].token_ids[1], 7U);
  EXPECT_EQ(work.hash_router[0].raw_scores.size(), 512U);
  EXPECT_EQ(work.hash_router[0].token_to_experts.size(),
            8U * DeepSeekExpertSubwavePlan::kRoutesPerToken);
  EXPECT_EQ(work.hash_router[0].scratch, &*scratch);
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     RejectsInvalidHashRouterShapeBeforePublishingViews) {
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  DeepSeekRankComputeWorkBuilder builder;
  EXPECT_FALSE(builder.add_hash_router(
      3, {0}, std::vector<float>(256), 1,
      std::vector<std::uint16_t>(8), &scratch).ok());
  EXPECT_FALSE(builder.add_hash_router(
      0, {0}, std::vector<float>(255), 1,
      std::vector<std::uint16_t>(8), &scratch).ok());
  EXPECT_FALSE(builder.add_hash_router(
      0, {0}, std::vector<float>(256), 1,
      std::vector<std::uint16_t>(7), &scratch).ok());
  EXPECT_FALSE(builder.add_hash_router(
      0, {0}, std::vector<float>(256), 1,
      std::vector<std::uint16_t>(8), nullptr).ok());
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     RetainsSharedImmutableHashTableWithoutPerPlanCopy) {
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  auto table = std::make_shared<const std::vector<std::uint16_t>>(
      DeepSeekExpertSubwavePlan::kRoutesPerToken, 0);
  const auto* table_data = table->data();
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(builder.add_hash_router_with_shared_table(
      0, {0}, std::vector<float>(256), 1, table, &scratch).ok());
  table.reset();
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.hash_router.size(), 1U);
  EXPECT_EQ(work.hash_router[0].token_to_experts.data(), table_data);
  EXPECT_EQ(work.hash_router[0].token_to_experts.size(),
            DeepSeekExpertSubwavePlan::kRoutesPerToken);
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     OwnsProjectedHashStagingAndRebindsPublishedViews) {
  const DeepSeekPipelinePlanDescriptor descriptor{
      1, 2, DeepSeekPlanPhase::kDecode, 1, 1};
  auto store = DeepSeekExpertPlanCatalog::Create(descriptor, {0, 0}).value();
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  BuilderRouterOperations operations;
  auto coordinator = DeepSeekHashRouterCoordinator::Create(
      1, scratch, store, operations).value();
  BuilderPinnedAllocator allocator;
  auto allocated = DeepSeekLearnedRouterHostStaging::Allocate(1, allocator);
  ASSERT_TRUE(allocated.ok());
  auto staging = std::make_shared<DeepSeekLearnedRouterHostStaging>(
      std::move(*allocated));
  auto table = std::make_shared<const std::vector<std::uint16_t>>(
      DeepSeekExpertSubwavePlan::kRoutesPerToken, 0);
  DeepSeekLearnedRouterSubmission projection{0, 1, 1, 2, 3, 4, 5, 6, {}, nullptr};
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(builder.add_projected_hash_router_with_shared_table(
      0, {0}, &coordinator, projection, 1, table, staging).ok());
  staging.reset();
  auto work = std::move(builder).finish().value();
  ASSERT_EQ(work.hash_router.size(), 1U);
  const auto& published = work.hash_router[0];
  EXPECT_EQ(published.coordinator, &coordinator);
  EXPECT_EQ(published.submission.projection.host_scores.size(), 256U);
  EXPECT_NE(published.submission.projection.host_error_flag, nullptr);
  EXPECT_EQ(published.submission.token_ids[0], 0U);
  EXPECT_EQ(published.submission.token_to_experts.data(), table->data());
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     RejectsLearnedRouterWithoutCompleteHostBacking) {
  DeepSeekRankComputeWorkBuilder builder;
  DeepSeekLearnedRouterSubmission submission;
  submission.layer = 3;
  submission.token_count = 1;
  std::vector<float> scores(DeepSeekLearnedRouterCoordinator::kExpertCount);
  std::uint32_t error = 0;
  submission.host_scores = scores;
  submission.host_error_flag = &error;
  EXPECT_FALSE(builder.add_learned_router(3, nullptr, submission).ok());
  submission.host_scores = std::span<float>(scores).first(scores.size() - 1);
  EXPECT_FALSE(builder.add_learned_router(
      3, reinterpret_cast<DeepSeekLearnedRouterCoordinator*>(1),
      submission).ok());
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     OwnsDecodeAttentionIndexSpansAndRejectsChunkMixing) {
  DeepSeekRankComputePlanWork published;
  {
    DeepSeekRankComputeWorkBuilder builder;
    DeepSeekDecodeAttentionWork decode;
    std::vector<std::uint32_t> positions{9, 10};
    std::vector<std::uint32_t> visible{4, 5};
    decode.attention.query_positions = positions;
    decode.attention.selection.visible_slot_counts = visible;
    decode.recent_writer = fake_resource<DeepSeekRecentStateWriter>(0x1000);
    decode.update_coordinator =
        fake_resource<DeepSeekCompressedLayerUpdateCoordinator>(0x2000);
    decode.coordinator =
        fake_resource<DeepSeekAttentionLayerCoordinator>(0x3000);
    decode.transaction =
        fake_resource<DeepSeekAttentionSequenceTransaction>(0x4000);
    ASSERT_TRUE(builder.add_decode_attention(4, decode).ok());

    DeepSeekChunkAttentionWork chunk;
    std::vector<DeepSeekRecentStateSubmission> recent(1);
    std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(1);
    chunk.submission.recent = recent;
    chunk.submission.updates = updates;
    chunk.submission.attention.query_positions = positions;
    chunk.submission.attention.selection.visible_slot_counts = visible;
    chunk.chunk_coordinator =
        fake_resource<DeepSeekPrefillLayerCoordinator>(0x5000);
    chunk.attention_coordinator = decode.coordinator;
    chunk.transaction = decode.transaction;
    EXPECT_FALSE(builder.add_chunk_attention(4, chunk).ok());

    published = std::move(builder).finish().value();
  }
  ASSERT_EQ(published.decode_attention.size(), 1U);
  EXPECT_EQ(published.decode_attention[0].work.attention.query_positions[1],
            10U);
  EXPECT_EQ(published.decode_attention[0]
                .work.attention.selection.visible_slot_counts[0], 4U);
  EXPECT_TRUE(published.chunk_attention.empty());
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     RetainsPlanScopedAttentionTransaction) {
  BuilderFixedOperations operations;
  auto banks = DeepSeekFixedStateBanks::Create(
      {0x100000, 4096}, {0x200000, 4096}, 19, operations).value();
  auto ratio4 = DeepSeekRatio4PagePool::Create(2).value();
  auto ratio128 = DeepSeekRatio128PagePool::Create(2).value();
  auto transaction = DeepSeekAttentionSequenceTransaction::Create(
      7, 1, 1, banks, ratio4, ratio128).value();
  auto owned = std::make_unique<DeepSeekAttentionSequenceTransaction>(
      std::move(transaction));
  auto* identity = owned.get();

  DeepSeekRankComputeWorkBuilder builder;
  std::vector<std::unique_ptr<DeepSeekAttentionSequenceTransaction>>
      transactions;
  transactions.push_back(std::move(owned));
  ASSERT_TRUE(
      builder.own_attention_transactions(std::move(transactions)).ok());
  DeepSeekDecodeAttentionWork decode;
  std::vector<std::uint32_t> positions{9};
  std::vector<std::uint32_t> visible{4};
  decode.attention.query_positions = positions;
  decode.attention.selection.visible_slot_counts = visible;
  decode.recent_writer = fake_resource<DeepSeekRecentStateWriter>(0x3000);
  decode.update_coordinator =
      fake_resource<DeepSeekCompressedLayerUpdateCoordinator>(0x4000);
  decode.coordinator =
      fake_resource<DeepSeekAttentionLayerCoordinator>(0x5000);
  decode.transaction = identity;
  ASSERT_TRUE(builder.add_decode_attention(4, decode).ok());
  auto work = std::move(builder).finish().value();
  EXPECT_EQ(work.decode_attention[0].work.transaction, identity);
  EXPECT_TRUE(identity->begin(7).ok());
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     OwnsChunkAttentionSubmissionSpansAfterInputsDie) {
  DeepSeekRankComputeWorkBuilder builder;
  DeepSeekChunkAttentionWork chunk;
  std::vector<DeepSeekRecentStateSubmission> recent(2);
  recent[1].absolute_position = 17;
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(2);
  updates[1].ratio = 128;
  std::vector<std::uint32_t> positions{15, 16};
  std::vector<std::uint32_t> visible{6, 7};
  chunk.submission.recent = recent;
  chunk.submission.updates = updates;
  chunk.submission.attention.query_positions = positions;
  chunk.submission.attention.selection.visible_slot_counts = visible;
  chunk.chunk_coordinator =
      fake_resource<DeepSeekPrefillLayerCoordinator>(0x5000);
  chunk.attention_coordinator =
      fake_resource<DeepSeekAttentionLayerCoordinator>(0x3000);
  chunk.transaction =
      fake_resource<DeepSeekAttentionSequenceTransaction>(0x4000);
  ASSERT_TRUE(builder.add_chunk_attention(5, chunk).ok());
  auto published = std::move(builder).finish().value();
  recent.clear();
  updates.clear();
  positions.clear();
  visible.clear();

  ASSERT_EQ(published.chunk_attention.size(), 1U);
  EXPECT_EQ(published.chunk_attention[0].work.submission.recent[1]
                .absolute_position, 17U);
  EXPECT_EQ(published.chunk_attention[0].work.submission.updates[1].ratio,
            128U);
  EXPECT_EQ(published.chunk_attention[0]
                .work.submission.attention.query_positions[0], 15U);
  EXPECT_EQ(published.chunk_attention[0]
                .work.submission.attention.selection.visible_slot_counts[1],
            7U);
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     OwnsDenseMhcEndpointAndDsparkCollections) {
  DeepSeekRankComputeWorkBuilder builder;

  DeepSeekDenseAttentionStageSequenceWork dense;
  dense.input_coordinator =
      fake_resource<DeepSeekAttentionProjectionCoordinator>(0x6000);
  dense.output_coordinator =
      fake_resource<DeepSeekAttentionOutputProjectionCoordinator>(0x7000);
  dense.transaction =
      fake_resource<DeepSeekAttentionSequenceTransaction>(0x8000);
  ASSERT_TRUE(builder.add_dense_attention(6, {dense}).ok());

  DeepSeekMhcStageSequenceWork mhc_attention;
  mhc_attention.executor =
      fake_resource<DeepSeekMhcSequenceExecutor>(0x9000);
  mhc_attention.transaction = dense.transaction;
  mhc_attention.submission.kind = DeepSeekMhcBranchKind::kAttention;
  mhc_attention.submission.layer_id = 6;
  mhc_attention.submission.token_count = 2;
  ASSERT_TRUE(builder.add_mhc_attention(6, {mhc_attention}).ok());

  DeepSeekMhcStageSequenceWork mhc_feed_forward = mhc_attention;
  mhc_feed_forward.executor =
      fake_resource<DeepSeekMhcSequenceExecutor>(0xA000);
  mhc_feed_forward.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
  ASSERT_TRUE(builder.add_mhc_feed_forward(6, {mhc_feed_forward}).ok());

  DeepSeekEndpointStageSequenceWork endpoint;
  endpoint.executor =
      fake_resource<DeepSeekEndpointSequenceExecutor>(0xB000);
  endpoint.transaction = dense.transaction;
  ASSERT_TRUE(builder.set_endpoint({endpoint}).ok());

  DeepSeekDsparkStageWork dspark;
  dspark.embed_coordinator =
      fake_resource<DeepSeekDsparkEmbedCoordinator>(0xC000);
  dspark.head_executor =
      fake_resource<DeepSeekDsparkHeadExecutor>(0xD000);
  dspark.transaction = dense.transaction;
  ASSERT_TRUE(builder.set_dspark(dspark).ok());

  auto published = std::move(builder).finish();
  ASSERT_TRUE(published.ok()) << published.status().message();
  ASSERT_EQ(published->dense_attention.size(), 1U);
  EXPECT_EQ(published->dense_attention[0].layer, 6U);
  ASSERT_EQ(published->mhc_attention.size(), 1U);
  EXPECT_EQ(published->mhc_attention[0].sequences[0].submission.kind,
            DeepSeekMhcBranchKind::kAttention);
  ASSERT_EQ(published->mhc_feed_forward.size(), 1U);
  EXPECT_EQ(published->mhc_feed_forward[0].sequences[0].submission.kind,
            DeepSeekMhcBranchKind::kFeedForward);
  ASSERT_EQ(published->endpoint.size(), 1U);
  ASSERT_NE(published->dspark, nullptr);
  EXPECT_EQ(published->dspark->head_executor, dspark.head_executor);
}

TEST(DeepSeekRankComputeWorkBuilderTest,
     RejectsDuplicateSingletonAndMhcBranchMismatch) {
  DeepSeekRankComputeWorkBuilder builder;
  DeepSeekEndpointStageSequenceWork endpoint;
  endpoint.executor =
      fake_resource<DeepSeekEndpointSequenceExecutor>(0xB000);
  endpoint.transaction =
      fake_resource<DeepSeekAttentionSequenceTransaction>(0x8000);
  ASSERT_TRUE(builder.set_endpoint({endpoint}).ok());
  EXPECT_FALSE(builder.set_endpoint({endpoint}).ok());

  DeepSeekMhcStageSequenceWork wrong;
  wrong.executor = fake_resource<DeepSeekMhcSequenceExecutor>(0x9000);
  wrong.transaction = endpoint.transaction;
  wrong.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
  wrong.submission.layer_id = 4;
  EXPECT_FALSE(builder.add_mhc_attention(4, {wrong}).ok());
}

}  // namespace
}  // namespace pih
