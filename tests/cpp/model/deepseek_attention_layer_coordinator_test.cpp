#include "pih/model/deepseek_attention_layer_coordinator.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

namespace pih { namespace {

class LayerFixedOperations final : public DeepSeekFixedStateBankOperations {
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

class LayerSelectionOperations final
    : public DeepSeekIndexSelectionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekIndexSelectionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status score(DeepSeekIndexScoreLaunch value) override {
    launch = value;
    tile_base = calls++ * DeepSeekIndexScoreLaunch::kMaximumSlotTile;
    return Status::Ok();
  }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t bytes,
                        std::uintptr_t) override {
    if (bytes == sizeof(std::uint32_t)) {
      *static_cast<std::uint32_t*>(host) = 0;
      return Status::Ok();
    }
    auto* scores = static_cast<float*>(host);
    for (std::uint32_t slot = 0; slot < launch.slot_count; ++slot) {
      scores[slot] = static_cast<float>(tile_base + slot);
    }
    return Status::Ok();
  }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t) override {
    return event;
  }
  DeepSeekIndexScoreLaunch launch;
  DeepSeekExpertAsyncStatus event = DeepSeekExpertAsyncStatus::kSuccess;
  std::uint32_t calls = 0;
  std::uint32_t tile_base = 0;
};

class LayerProjectionOperations final
    : public DeepSeekIndexerProjectionOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status project(DeepSeekIndexerProjectionLaunch) override {
    ++calls;
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = copied_error;
    return Status::Ok();
  }
  std::uint32_t calls = 0;
  std::uint32_t copied_error = 0;
};

class LayerSparseOperations final : public DeepSeekSparseAttentionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekSparseAttentionHostStaging&) override {
    return Status::Ok();
  }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status copy_h2d_async(std::uintptr_t, const void* host, std::size_t bytes,
                        std::uintptr_t) override {
    const auto* begin = static_cast<const std::int32_t*>(host);
    copied_indices.assign(begin, begin + bytes / sizeof(std::int32_t));
    return Status::Ok();
  }
  Status attention(DeepSeekSparseAttentionLaunch value) override {
    launch = value;
    return Status::Ok();
  }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    *static_cast<std::uint32_t*>(host) = 0;
    return Status::Ok();
  }
  std::vector<std::int32_t> copied_indices;
  DeepSeekSparseAttentionLaunch launch;
};

class LayerStateOperations final : public DeepSeekCompressorStateOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status pooling(DeepSeekCompressorPoolingLaunch) override {
    ++pooling_calls;
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 0;
    return Status::Ok();
  }
  std::uint32_t pooling_calls = 0;
};

class LayerPageOperations final : public DeepSeekCompressedPageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    return Status::Ok();
  }
  Status store(DeepSeekCompressorBf16StoreLaunch) override {
    ++store_calls;
    return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 0;
    return Status::Ok();
  }
  std::uint32_t store_calls = 0;
};

class LayerRecentOperations final : public DeepSeekRecentStateOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    ++calls;
    return Status::Ok();
  }
  std::uint32_t calls = 0;
};

struct LayerFixture final {
  DeepSeekFixedStateLayout fixed_layout =
      DeepSeekFixedStateLayout::Build(std::vector<std::uint32_t>{2, 3}, false)
          .value();
  LayerFixedOperations fixed_operations;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, fixed_layout.total_bytes()},
      {0x200000, fixed_layout.total_bytes()}, 5, fixed_operations).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(2).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(2).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  LayerSelectionOperations selection_operations;
  std::vector<float> score_staging = std::vector<float>(4096);
  std::vector<std::uint32_t> page_slot_staging =
      std::vector<std::uint32_t>(2);
  std::uint32_t selection_error = 0;
  std::uint32_t projection_error = 0;
  DeepSeekIndexSelectionDriver selection =
      DeepSeekIndexSelectionDriver::Create(
          selection_operations,
          {score_staging.data(), &selection_error,
           static_cast<std::uint32_t>(score_staging.size()),
           page_slot_staging.data(),
           static_cast<std::uint32_t>(page_slot_staging.size())},
          19).value();
  LayerProjectionOperations projection_operations;
  DeepSeekIndexerProjectionCoordinator projection =
      DeepSeekIndexerProjectionCoordinator::Create(
          projection_operations, &projection_error).value();
  LayerSparseOperations sparse_operations;
  std::vector<std::int32_t> index_staging = std::vector<std::int32_t>(8320);
  std::uint32_t shared_error = 0;
  DeepSeekSparseAttentionDriver attention =
      DeepSeekSparseAttentionDriver::Create(
          sparse_operations,
          {index_staging.data(), static_cast<std::uint32_t>(index_staging.size()),
           &shared_error, page_slot_staging.data(),
           static_cast<std::uint32_t>(page_slot_staging.size())}).value();
  LayerStateOperations state_operations;
  LayerPageOperations page_operations;
  LayerRecentOperations recent_operations;
  DeepSeekAttentionPageArena page_arena = DeepSeekAttentionPageArena::Create(
      {0x300000, 2U * 65536U}, {0x400000, 2U * 16384U},
      {0x500000, 2U * 65536U}, 2, 2).value();
  DeepSeekCompressorStateWriter state_writer =
      DeepSeekCompressorStateWriter::Create(
          fixed_layout, state_operations, &shared_error).value();
  DeepSeekRecentStateWriter recent_writer =
      DeepSeekRecentStateWriter::Create(fixed_layout, recent_operations)
          .value();
  DeepSeekCompressedPageWriter page_writer =
      DeepSeekCompressedPageWriter::Create(
          page_arena, page_operations, &shared_error).value();
  DeepSeekCompressedLayerUpdateCoordinator update_coordinator =
      DeepSeekCompressedLayerUpdateCoordinator::Create(
          state_writer, page_writer).value();
  DeepSeekAttentionLayerCoordinator coordinator =
      DeepSeekAttentionLayerCoordinator::Create(
          projection, selection, attention).value();
};

DeepSeekSparseAttentionSubmission attention_submission() {
  return {nullptr, 1, 2, 3, 4, 5, 6, 7, 64, 10000};
}

void attach_projection(DeepSeekAttentionLayerSubmission& submission) {
  submission.has_indexer_projection = true;
  submission.indexer_projection = {
      101, 102, 103, 107, 108, 109, 104, 105, 106,
      submission.selection.query_bf16,
      submission.selection.head_weight_f32,
      51,
      submission.selection.stream,
      submission.selection.query_count,
      1048576};
}

TEST(DeepSeekAttentionLayerCoordinatorTest, PostsDeterministicRatio128Directly) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{127};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio128;
  submission.query_positions = positions;
  submission.compressed_slot_count = 1;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kAttentionPosted);
  EXPECT_EQ(fixture.sparse_operations.launch.index_count, 129U);
  EXPECT_EQ(fixture.sparse_operations.copied_indices.back(), 128);
}

TEST(DeepSeekAttentionLayerCoordinatorTest, StreamsRatio4ThenPostsTop512) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{19999};
  const std::vector<std::uint32_t> visible{5000};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio4;
  submission.query_positions = positions;
  submission.compressed_slot_count = 5000;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.selection = {10, 20, 30, {40, 50}, visible,
                          5000, 1, 64, 7};
  attach_projection(submission);
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  while (fixture.coordinator.state() !=
         DeepSeekAttentionLayerCoordinatorState::kAttentionPosted) {
    ASSERT_TRUE(fixture.coordinator.launch_next_selection_tile().ok());
    ASSERT_EQ(*fixture.coordinator.poll_selection_tile(),
              DeepSeekExpertAsyncStatus::kSuccess);
  }
  EXPECT_EQ(fixture.selection_operations.calls, 2U);
  EXPECT_EQ(fixture.projection_operations.calls, 1U);
  EXPECT_EQ(fixture.sparse_operations.launch.index_count, 640U);
  EXPECT_EQ(fixture.sparse_operations.copied_indices[128], 5127);
  EXPECT_EQ(fixture.sparse_operations.copied_indices.back(), 4616);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     ProductionCoordinatorUsesTransactionPagedSelection) {
  LayerFixture fixture;
  auto page = fixture.ratio4.reserve(7, 2, 0).value();
  ASSERT_TRUE(fixture.ratio4.publish(page).ok());
  auto paged = DeepSeekAttentionLayerCoordinator::CreatePaged(
      2, fixture.fixed_layout, fixture.page_arena, fixture.projection,
      fixture.selection, fixture.attention).value();
  const std::vector<std::uint32_t> positions{3};
  const std::vector<std::uint32_t> visible{1};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio4;
  submission.query_positions = positions;
  submission.compressed_slot_count = 1;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.selection = {10, 20, 30, {40, 50, 60, 2}, visible,
                          1, 1, 64, 7};
  attach_projection(submission);
  submission.attention = attention_submission();
  submission.attention.device_page_slots_u32 = 60;
  ASSERT_TRUE(paged.begin(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.page_slot_staging[0], page.index.slot());
  ASSERT_TRUE(paged.launch_next_selection_tile().ok());
  EXPECT_EQ(fixture.selection_operations.launch.page_slots_u32, 60U);
  EXPECT_EQ(fixture.selection_operations.launch.logical_page_count, 1U);
  EXPECT_EQ(fixture.selection_operations.launch.physical_page_count, 2U);
  ASSERT_EQ(*paged.poll_selection_tile(),
            DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_EQ(fixture.sparse_operations.launch.latent_kv_bf16,
            0x200000U);
  EXPECT_EQ(fixture.sparse_operations.launch.compressed_kv_bf16,
            fixture.page_arena.ratio4_main_base());
  EXPECT_EQ(fixture.sparse_operations.launch.page_slots_u32, 60U);
  EXPECT_EQ(fixture.sparse_operations.launch.logical_page_count, 1U);
  EXPECT_EQ(fixture.sparse_operations.launch.physical_page_count, 2U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     ProductionCoordinatorPagesRatio128SparseAttention) {
  LayerFixture fixture;
  auto page = fixture.ratio128.reserve(7, 3, 0).value();
  ASSERT_TRUE(fixture.ratio128.publish(page).ok());
  auto paged = DeepSeekAttentionLayerCoordinator::CreatePaged(
      3, fixture.fixed_layout, fixture.page_arena, fixture.projection,
      fixture.selection, fixture.attention).value();
  const std::vector<std::uint32_t> positions{127};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio128;
  submission.query_positions = positions;
  submission.compressed_slot_count = 1;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.attention = attention_submission();
  submission.attention.device_page_slots_u32 = 60;
  ASSERT_TRUE(paged.begin(submission, fixture.transaction).ok());
  const auto fixed = fixture.fixed_layout.Resolve(
      3, {0x200000U, fixture.fixed_layout.total_bytes()}).value();
  EXPECT_EQ(fixture.sparse_operations.launch.latent_kv_bf16,
            fixed.recent_bf16.address);
  EXPECT_EQ(fixture.sparse_operations.launch.compressed_kv_bf16,
            fixture.page_arena.ratio128_main_base());
  EXPECT_EQ(fixture.sparse_operations.launch.page_slots_u32, 60U);
  EXPECT_EQ(fixture.sparse_operations.launch.logical_page_count, 1U);
  EXPECT_EQ(fixture.sparse_operations.launch.physical_page_count, 2U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest, SelectionErrorPoisonsLayer) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{3};
  const std::vector<std::uint32_t> visible{1};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio4;
  submission.query_positions = positions;
  submission.compressed_slot_count = 1;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.selection = {10, 20, 30, {40, 50}, visible, 1, 1, 64, 7};
  attach_projection(submission);
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  ASSERT_TRUE(fixture.coordinator.launch_next_selection_tile().ok());
  fixture.selection_operations.event = DeepSeekExpertAsyncStatus::kError;
  EXPECT_EQ(*fixture.coordinator.poll_selection_tile(),
            DeepSeekExpertAsyncStatus::kError);
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kPoisoned);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     ProjectionErrorSurvivesSelectionErrorClear) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{3};
  const std::vector<std::uint32_t> visible{1};
  fixture.projection_operations.copied_error = 17;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio4;
  submission.query_positions = positions;
  submission.compressed_slot_count = 1;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.selection = {10, 20, 30, {40, 50}, visible, 1, 1, 64, 7};
  attach_projection(submission);
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  ASSERT_TRUE(fixture.coordinator.launch_next_selection_tile().ok());
  ASSERT_EQ(fixture.coordinator.poll_selection_tile().value(),
            DeepSeekExpertAsyncStatus::kSuccess);
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  EXPECT_EQ(fixture.transaction.poll().value(),
            DeepSeekExpertAsyncStatus::kError);
  EXPECT_EQ(fixture.projection_error, 17U);
  EXPECT_EQ(fixture.selection_error, 0U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     Ratio4BeforeFirstCompressedSlotPostsRecentOnly) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{0, 1, 2};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio4;
  submission.query_positions = positions;
  submission.compressed_slot_count = 0;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kAttentionPosted);
  EXPECT_EQ(fixture.selection_operations.calls, 0U);
  EXPECT_EQ(fixture.sparse_operations.launch.query_count, 3U);
  EXPECT_EQ(fixture.sparse_operations.launch.index_count, 128U);
  EXPECT_EQ(fixture.sparse_operations.copied_indices.size(), 384U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     RearmsOnlyAfterTheSequenceTransactionCompletes) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> first_position{0};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekAttentionLayerSubmission submission;
  submission.kind = DeepSeekCompressedAttentionKind::kRatio128;
  submission.query_positions = first_position;
  submission.recent_physical_offset = 0;
  submission.compressed_physical_offset = 128;
  submission.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
  EXPECT_FALSE(fixture.coordinator.reset().ok());
  ASSERT_TRUE(fixture.transaction.seal(7).ok());
  ASSERT_EQ(fixture.transaction.poll().value(),
            DeepSeekExpertAsyncStatus::kSuccess);
  ASSERT_TRUE(fixture.transaction.commit().ok());
  ASSERT_TRUE(fixture.coordinator.reset().ok());
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kIdle);

  const std::vector<std::uint32_t> second_position{1};
  submission.query_positions = second_position;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  EXPECT_TRUE(fixture.coordinator.begin(submission, fixture.transaction).ok());
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     DecodeUpdatesRatio128StateBeforePostingAttention) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{127};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio128_append(3, 0).value();
  DeepSeekCompressedLayerUpdateSubmission update;
  update.ratio = 128;
  update.main_state = {3, false, 11, 12, 13, 14, 6, 7, 1, 127};
  update.has_completed_slot = true;
  update.ratio128_slot = {target, {}, false, 14, 21, 22, 6, 7, 127,
                          1.0e-6F};
  DeepSeekAttentionLayerSubmission attention;
  attention.kind = DeepSeekCompressedAttentionKind::kRatio128;
  attention.query_positions = positions;
  attention.compressed_slot_count = 1;
  attention.recent_physical_offset = 0;
  attention.compressed_physical_offset = 128;
  attention.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin_decode(
      {3, 50, 7, 1, 127}, update, attention, fixture.recent_writer,
      fixture.update_coordinator, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_operations.calls, 1U);
  EXPECT_EQ(fixture.state_operations.pooling_calls, 1U);
  EXPECT_EQ(fixture.page_operations.store_calls, 1U);
  EXPECT_EQ(fixture.sparse_operations.launch.query_count, 1U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     DecodeRejectsCompressedCountBeforeUpdatingState) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{127};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  DeepSeekCompressedLayerUpdateSubmission update;
  update.ratio = 128;
  update.main_state = {3, false, 11, 12, 13, 14, 6, 7, 1, 127};
  update.has_completed_slot = true;
  DeepSeekAttentionLayerSubmission attention;
  attention.kind = DeepSeekCompressedAttentionKind::kRatio128;
  attention.query_positions = positions;
  attention.compressed_slot_count = 0;
  attention.recent_physical_offset = 0;
  attention.compressed_physical_offset = 128;
  attention.attention = attention_submission();
  EXPECT_FALSE(fixture.coordinator.begin_decode(
      {3, 50, 7, 1, 127}, update, attention, fixture.recent_writer,
      fixture.update_coordinator, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_operations.calls, 0U);
  EXPECT_EQ(fixture.state_operations.pooling_calls, 0U);
}

TEST(DeepSeekAttentionLayerCoordinatorTest,
     DecodeRatio4WaitsForSelectionAfterPairedStateWrites) {
  LayerFixture fixture;
  const std::vector<std::uint32_t> positions{3};
  const std::vector<std::uint32_t> visible{1};
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  DeepSeekCompressedLayerUpdateSubmission update;
  update.ratio = 4;
  update.main_state = {2, false, 11, 12, 13, 14, 6, 7, 1, 3};
  update.index_state = {2, true, 15, 16, 17, 18, 6, 7, 1, 3};
  update.has_completed_slot = true;
  update.ratio4_slot = {target, {}, false, 14, 18, 21, 22, 23, 24,
                        6, 7, 3, 1.0e-6F};
  DeepSeekAttentionLayerSubmission attention;
  attention.kind = DeepSeekCompressedAttentionKind::kRatio4;
  attention.query_positions = positions;
  attention.compressed_slot_count = 1;
  attention.recent_physical_offset = 0;
  attention.compressed_physical_offset = 128;
  attention.selection = {31, 32, 33, {34, 35}, visible, 1, 1, 64, 7};
  attach_projection(attention);
  attention.attention = attention_submission();
  ASSERT_TRUE(fixture.coordinator.begin_decode(
      {2, 50, 7, 1, 3}, update, attention, fixture.recent_writer,
      fixture.update_coordinator, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_operations.calls, 1U);
  EXPECT_EQ(fixture.state_operations.pooling_calls, 2U);
  EXPECT_EQ(fixture.page_operations.store_calls, 2U);
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kSelectionReady);
  ASSERT_TRUE(fixture.coordinator.launch_next_selection_tile().ok());
  ASSERT_EQ(fixture.coordinator.poll_selection_tile().value(),
            DeepSeekExpertAsyncStatus::kSuccess);
  EXPECT_EQ(fixture.coordinator.state(),
            DeepSeekAttentionLayerCoordinatorState::kAttentionPosted);
  EXPECT_EQ(fixture.sparse_operations.launch.query_count, 1U);
}

} }  // namespace pih
