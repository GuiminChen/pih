#include "pih/model/deepseek_native_rank_compute_plan.h"

#include <gtest/gtest.h>

#include "pih/model/deepseek_rank_compute_work_builder.h"

namespace pih {
namespace {

template <typename T>
T* fake(std::uintptr_t value) {
  return reinterpret_cast<T*>(value);
}

DeepSeekRankComputePlanWork complete_decode_work(
    DeepSeekStagePlan stage, DeepSeekRouteScratchArena& scratch,
    std::optional<DeepSeekBoundarySendSource> outgoing = std::nullopt) {
  DeepSeekRankComputeWorkBuilder builder;
  std::uint32_t host_error = 0;
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    if (layer < 3) {
      EXPECT_TRUE(builder.add_hash_router(
          layer, {0}, std::vector<float>(256), 1,
          std::vector<std::uint16_t>(
              DeepSeekExpertSubwavePlan::kRoutesPerToken),
          &scratch).ok());
    } else {
      DeepSeekLearnedRouterSubmission learned;
      learned.layer = layer;
      learned.token_count = 1;
      std::vector<float> scores(
          DeepSeekLearnedRouterCoordinator::kExpertCount);
      learned.host_scores = scores;
      learned.host_error_flag = &host_error;
      EXPECT_TRUE(builder.add_learned_router(
          layer, fake<DeepSeekLearnedRouterCoordinator>(
                     0x10000 + layer * 0x100), learned).ok());
    }

    std::vector<std::uint32_t> positions{0};
    std::vector<std::uint32_t> visible{0};
    DeepSeekDecodeAttentionWork decode;
    decode.attention.query_positions = positions;
    decode.attention.selection.visible_slot_counts = visible;
    decode.recent_writer = fake<DeepSeekRecentStateWriter>(
        0x20000 + layer * 0x100);
    decode.update_coordinator =
        fake<DeepSeekCompressedLayerUpdateCoordinator>(
            0x30000 + layer * 0x100);
    decode.coordinator = fake<DeepSeekAttentionLayerCoordinator>(
        0x40000 + layer * 0x100);
    decode.transaction = fake<DeepSeekAttentionSequenceTransaction>(
        0x50000 + layer * 0x100);
    EXPECT_TRUE(builder.add_decode_attention(layer, decode).ok());

    DeepSeekDenseAttentionStageSequenceWork dense;
    dense.input_coordinator =
        fake<DeepSeekAttentionProjectionCoordinator>(
            0x60000 + layer * 0x100);
    dense.output_coordinator =
        fake<DeepSeekAttentionOutputProjectionCoordinator>(
            0x70000 + layer * 0x100);
    dense.transaction = decode.transaction;
    EXPECT_TRUE(builder.add_dense_attention(layer, {dense}).ok());

    DeepSeekMhcStageSequenceWork attention;
    attention.executor = fake<DeepSeekMhcSequenceExecutor>(
        0x80000 + layer * 0x100);
    attention.transaction = decode.transaction;
    attention.submission.kind = DeepSeekMhcBranchKind::kAttention;
    attention.submission.layer_id = layer;
    attention.submission.token_count = 1;
    EXPECT_TRUE(builder.add_mhc_attention(layer, {attention}).ok());
    auto feed_forward = attention;
    feed_forward.executor = fake<DeepSeekMhcSequenceExecutor>(
        0x90000 + layer * 0x100);
    feed_forward.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
    EXPECT_TRUE(builder.add_mhc_feed_forward(
        layer, {feed_forward}).ok());
  }
  if (stage.owns_embedding || stage.owns_lm_head) {
    DeepSeekEndpointStageSequenceWork endpoint;
    endpoint.executor = fake<DeepSeekEndpointSequenceExecutor>(
        0xA0000 + stage.rank * 0x100);
    endpoint.transaction = fake<DeepSeekAttentionSequenceTransaction>(
        0xB0000 + stage.rank * 0x100);
    EXPECT_TRUE(builder.set_endpoint({endpoint}).ok());
  }
  if (outgoing.has_value()) {
    EXPECT_TRUE(builder.set_outgoing_boundary(*outgoing).ok());
  }
  return std::move(builder).finish().value();
}

DeepSeekRankComputePlanWork complete_prefill_work(
    DeepSeekStagePlan stage, DeepSeekRouteScratchArena& scratch) {
  DeepSeekRankComputeWorkBuilder builder;
  std::uint32_t host_error = 0;
  for (std::uint32_t layer = stage.layers.first_layer;
       layer <= stage.layers.last_layer; ++layer) {
    if (layer < 3) {
      EXPECT_TRUE(builder.add_hash_router(
          layer, {0}, std::vector<float>(256), 1,
          std::vector<std::uint16_t>(
              DeepSeekExpertSubwavePlan::kRoutesPerToken),
          &scratch).ok());
    } else {
      DeepSeekLearnedRouterSubmission learned;
      learned.layer = layer;
      learned.token_count = 1;
      std::vector<float> scores(
          DeepSeekLearnedRouterCoordinator::kExpertCount);
      learned.host_scores = scores;
      learned.host_error_flag = &host_error;
      EXPECT_TRUE(builder.add_learned_router(
          layer, fake<DeepSeekLearnedRouterCoordinator>(
                     0x10000 + layer * 0x100), learned).ok());
    }
    std::vector<DeepSeekRecentStateSubmission> recent(1);
    std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(1);
    std::vector<std::uint32_t> positions{0};
    std::vector<std::uint32_t> visible{0};
    DeepSeekChunkAttentionWork chunk;
    chunk.submission.recent = recent;
    chunk.submission.updates = updates;
    chunk.submission.attention.query_positions = positions;
    chunk.submission.attention.selection.visible_slot_counts = visible;
    chunk.chunk_coordinator = fake<DeepSeekPrefillLayerCoordinator>(
        0x20000 + layer * 0x100);
    chunk.attention_coordinator = fake<DeepSeekAttentionLayerCoordinator>(
        0x30000 + layer * 0x100);
    chunk.transaction = fake<DeepSeekAttentionSequenceTransaction>(
        0x40000 + layer * 0x100);
    EXPECT_TRUE(builder.add_chunk_attention(layer, chunk).ok());

    DeepSeekDenseAttentionStageSequenceWork dense;
    dense.input_coordinator = fake<DeepSeekAttentionProjectionCoordinator>(
        0x50000 + layer * 0x100);
    dense.output_coordinator =
        fake<DeepSeekAttentionOutputProjectionCoordinator>(
            0x60000 + layer * 0x100);
    dense.transaction = chunk.transaction;
    EXPECT_TRUE(builder.add_dense_attention(layer, {dense}).ok());
    DeepSeekMhcStageSequenceWork attention;
    attention.executor = fake<DeepSeekMhcSequenceExecutor>(
        0x70000 + layer * 0x100);
    attention.transaction = chunk.transaction;
    attention.submission.kind = DeepSeekMhcBranchKind::kAttention;
    attention.submission.layer_id = layer;
    attention.submission.token_count = 1;
    EXPECT_TRUE(builder.add_mhc_attention(layer, {attention}).ok());
    auto feed_forward = attention;
    feed_forward.executor = fake<DeepSeekMhcSequenceExecutor>(
        0x80000 + layer * 0x100);
    feed_forward.submission.kind = DeepSeekMhcBranchKind::kFeedForward;
    EXPECT_TRUE(builder.add_mhc_feed_forward(layer, {feed_forward}).ok());
  }
  DeepSeekAttentionSequenceTransaction* endpoint_transaction =
      fake<DeepSeekAttentionSequenceTransaction>(0x90000 + stage.rank * 0x100);
  if (stage.owns_embedding || stage.owns_lm_head) {
    DeepSeekEndpointStageSequenceWork endpoint;
    endpoint.executor = fake<DeepSeekEndpointSequenceExecutor>(
        0xA0000 + stage.rank * 0x100);
    endpoint.transaction = endpoint_transaction;
    EXPECT_TRUE(builder.set_endpoint({endpoint}).ok());
  }
  if (stage.owns_dspark) {
    DeepSeekDsparkStageWork dspark;
    dspark.kind =
        DeepSeekDsparkStageWorkKind::kPrefillStateInitialization;
    dspark.embed_coordinator = fake<DeepSeekDsparkEmbedCoordinator>(0xB0000);
    dspark.transaction = endpoint_transaction;
    EXPECT_TRUE(builder.set_dspark(dspark).ok());
    std::vector<DeepSeekBoundDsparkMtpStageWork> mtp(
        kDeepSeekDsparkStageCount);
    EXPECT_TRUE(builder.set_dspark_mtp(std::move(mtp)).ok());
  }
  return std::move(builder).finish().value();
}

class PlanAllocator final : public Allocator {
 public:
  Result<Allocation> allocate(std::uint64_t bytes,
                              std::uint64_t alignment) override {
    auto* data = _aligned_malloc(
        static_cast<std::size_t>(bytes),
        static_cast<std::size_t>(alignment));
    return Allocation{
        data, bytes, alignment, ++generation_,
        Device::Create(DeviceType::kCuda, 0).value()};
  }
  void deallocate(Allocation allocation) noexcept override {
    _aligned_free(allocation.data);
  }

 private:
  std::uint64_t generation_ = 0;
};

TEST(DeepSeekNativeRankComputePlanTest,
     RejectsIncompleteStageCoverageAndDrainDescriptors) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  DeepSeekRankComputeWorkBuilder builder;
  ASSERT_TRUE(builder.add_hash_router(
      0, {0}, std::vector<float>(256), 1,
      std::vector<std::uint16_t>(
          DeepSeekExpertSubwavePlan::kRoutesPerToken),
      &scratch).ok());
  std::vector<DeepSeekRankComputePlanWork> incomplete;
  incomplete.push_back(std::move(builder).finish().value());
  EXPECT_FALSE(DeepSeekNativeRankComputePlan::Create(
      {9, 1, DeepSeekPlanPhase::kDecode, 1, 1}, topology,
      std::move(incomplete)).ok());
  EXPECT_FALSE(DeepSeekNativeRankComputePlan::Create(
      {9, 2, DeepSeekPlanPhase::kDrain, 0, 1}, topology, {}).ok());
}

TEST(DeepSeekNativeRankComputePlanTest,
     RejectsRankCountBeforeReadingStageWork) {
  auto topology = DeepSeekPipelinePlan::Create(2, false).value();
  EXPECT_FALSE(DeepSeekNativeRankComputePlan::Create(
      {9, 1, DeepSeekPlanPhase::kPrefill, 4, 1}, topology, {}).ok());
}

TEST(DeepSeekNativeRankComputePlanTest,
     PublishesDsparkEnabledPrefillStateInitializationWork) {
  auto topology = DeepSeekPipelinePlan::Create(1, true).value();
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  std::vector<DeepSeekRankComputePlanWork> work;
  work.push_back(complete_prefill_work(topology.rank(0), scratch));
  auto result = DeepSeekNativeRankComputePlan::Create(
      {9, 1, DeepSeekPlanPhase::kPrefill, 1, 1}, topology,
      std::move(work));
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_NE(result->rank_work()[0].dspark, nullptr);
  EXPECT_EQ(result->rank_work()[0].dspark->head_executor, nullptr);
  EXPECT_EQ(result->rank_work()[0].dspark_mtp.size(),
            kDeepSeekDsparkStageCount);
}

TEST(DeepSeekNativeRankComputePlanTest,
     PublishesCompletePp1DecodeStageWithOpaqueOwners) {
  auto topology = DeepSeekPipelinePlan::Create(1, false).value();
  auto scratch = DeepSeekRouteScratchArena::Create(1).value();
  std::vector<DeepSeekRankComputePlanWork> work;
  work.push_back(complete_decode_work(topology.rank(0), scratch));
  auto native = DeepSeekNativeRankComputePlan::Create(
      {9, 1, DeepSeekPlanPhase::kDecode, 1, 1}, topology,
      std::move(work));
  ASSERT_TRUE(native.ok()) << native.status().message();
  ASSERT_EQ(native->rank_work().size(), 1U);
  EXPECT_NE(native->rank_work()[0].lifetime_owner, nullptr);
  EXPECT_EQ(native->rank_work()[0].decode_attention.size(), 43U);
  EXPECT_FALSE(native->rank_work()[0].outgoing_boundary.has_value());
}

TEST(DeepSeekNativeRankComputePlanTest,
     PublishesCompletePp2ThroughPp4WithExactBoundaryOwnership) {
  for (std::uint32_t world_size = 2; world_size <= 4; ++world_size) {
    SCOPED_TRACE(world_size);
    auto topology = DeepSeekPipelinePlan::Create(world_size, false).value();
    PlanAllocator allocator;
    std::vector<Buffer> boundary_buffers;
    std::vector<DeepSeekRouteScratchArena> scratches;
    boundary_buffers.reserve(world_size - 1);
    scratches.reserve(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      scratches.push_back(DeepSeekRouteScratchArena::Create(1).value());
      if (rank + 1 < world_size) {
        boundary_buffers.push_back(
            Buffer::Allocate(allocator, 32768, 256).value());
      }
    }
    std::vector<DeepSeekRankComputePlanWork> work;
    work.reserve(world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      std::optional<DeepSeekBoundarySendSource> source;
      if (rank + 1 < world_size) {
        source = DeepSeekBoundarySendSource::Create(
            boundary_buffers[rank], 0, 32768, 1, 19 + rank, 77, 9).value();
      }
      work.push_back(complete_decode_work(
          topology.rank(rank), scratches[rank], source));
    }
    auto native = DeepSeekNativeRankComputePlan::Create(
        {9, world_size, DeepSeekPlanPhase::kDecode, 1, 1}, topology,
        std::move(work));
    ASSERT_TRUE(native.ok()) << native.status().message();
    ASSERT_EQ(native->rank_work().size(), world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_EQ(native->rank_work()[rank].outgoing_boundary.has_value(),
                rank + 1 < world_size);
    }
  }
}

}  // namespace
}  // namespace pih
