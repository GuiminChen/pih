#include "pih/model/deepseek_prefill_layer_coordinator.h"

#include <gtest/gtest.h>

#include <vector>

namespace pih { namespace {

class FixedOps final : public DeepSeekFixedStateBankOperations {
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

class RecentOps final : public DeepSeekRecentStateOperations {
 public:
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { ++calls; return Status::Ok(); }
  std::uint32_t calls = 0;
};

class StateOps final : public DeepSeekCompressorStateOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status pooling(DeepSeekCompressorPoolingLaunch) override {
    ++calls; return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 0; return Status::Ok();
  }
  std::uint32_t calls = 0;
};

class PageOps final : public DeepSeekCompressedPageOperations {
 public:
  Status validate_host_error(std::uint32_t*) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status copy_d2d_async(std::uintptr_t, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status store(DeepSeekCompressorBf16StoreLaunch) override {
    ++calls; return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 0; return Status::Ok();
  }
  std::uint32_t calls = 0;
};

class SelectionOps final : public DeepSeekIndexSelectionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekIndexSelectionHostStaging&) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status score(DeepSeekIndexScoreLaunch) override { return Status::Ok(); }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status copy_d2h_async(void*, std::uintptr_t, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status record_event(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
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
    ++calls; return Status::Ok();
  }
  Status copy_error_d2h_async(std::uint32_t* host, std::uintptr_t,
                              std::uintptr_t) override {
    *host = 0; return Status::Ok();
  }
  std::uint32_t calls = 0;
};

class SparseOps final : public DeepSeekSparseAttentionOperations {
 public:
  Status validate_host_staging(
      const DeepSeekSparseAttentionHostStaging&) override { return Status::Ok(); }
  Status zero_u32_async(std::uintptr_t, std::uintptr_t) override {
    return Status::Ok();
  }
  Status copy_h2d_async(std::uintptr_t, const void*, std::size_t,
                        std::uintptr_t) override { return Status::Ok(); }
  Status attention(DeepSeekSparseAttentionLaunch launch) override {
    ++calls; query_count = launch.query_count; return Status::Ok();
  }
  Status copy_d2h_async(void* host, std::uintptr_t, std::size_t,
                        std::uintptr_t) override {
    *static_cast<std::uint32_t*>(host) = 0; return Status::Ok();
  }
  std::uint32_t calls = 0;
  std::uint32_t query_count = 0;
};

struct Fixture final {
  DeepSeekFixedStateLayout layout = DeepSeekFixedStateLayout::Build(
      std::vector<std::uint32_t>{2, 3}, false).value();
  FixedOps fixed_ops;
  DeepSeekFixedStateBanks banks = DeepSeekFixedStateBanks::Create(
      {0x100000, layout.total_bytes()}, {0x200000, layout.total_bytes()}, 5,
      fixed_ops).value();
  DeepSeekRatio4PagePool ratio4 = DeepSeekRatio4PagePool::Create(1).value();
  DeepSeekRatio128PagePool ratio128 =
      DeepSeekRatio128PagePool::Create(1).value();
  DeepSeekAttentionSequenceTransaction transaction =
      DeepSeekAttentionSequenceTransaction::Create(
          7, 1, 1, banks, ratio4, ratio128).value();
  RecentOps recent_ops;
  StateOps state_ops;
  PageOps page_ops;
  std::uint32_t shared_error = 0;
  DeepSeekRecentStateWriter recent_writer =
      DeepSeekRecentStateWriter::Create(layout, recent_ops).value();
  DeepSeekCompressorStateWriter state_writer =
      DeepSeekCompressorStateWriter::Create(layout, state_ops, &shared_error)
          .value();
  DeepSeekAttentionPageArena arena = DeepSeekAttentionPageArena::Create(
      {0x300000, 65536}, {0x400000, 16384}, {0x500000, 65536}, 1, 1).value();
  DeepSeekCompressedPageWriter page_writer =
      DeepSeekCompressedPageWriter::Create(arena, page_ops, &shared_error)
          .value();
  DeepSeekCompressedLayerUpdateCoordinator update =
      DeepSeekCompressedLayerUpdateCoordinator::Create(state_writer,
                                                         page_writer).value();
  SelectionOps selection_ops;
  std::vector<float> scores = std::vector<float>(8192);
  std::uint32_t selection_error = 0;
  std::uint32_t projection_error = 0;
  DeepSeekIndexSelectionDriver selection =
      DeepSeekIndexSelectionDriver::Create(
          selection_ops, {scores.data(), &selection_error, 8192}, 19).value();
  ProjectionOps projection_ops;
  DeepSeekIndexerProjectionCoordinator projection =
      DeepSeekIndexerProjectionCoordinator::Create(
          projection_ops, &projection_error).value();
  SparseOps sparse_ops;
  std::vector<std::int32_t> indices = std::vector<std::int32_t>(16640);
  DeepSeekSparseAttentionDriver sparse =
      DeepSeekSparseAttentionDriver::Create(
          sparse_ops, {indices.data(), 16640, &shared_error}).value();
  DeepSeekAttentionLayerCoordinator attention =
      DeepSeekAttentionLayerCoordinator::Create(
          projection, selection, sparse).value();
  DeepSeekPrefillLayerCoordinator prefill =
      DeepSeekPrefillLayerCoordinator::Create(
          recent_writer, update, selection, sparse, attention).value();
};

DeepSeekCompressedLayerUpdateSubmission update(std::uint32_t position) {
  DeepSeekCompressedLayerUpdateSubmission value;
  value.ratio = 128;
  value.main_state = {3, false, 11, 12, 13, 14, 6, 7, 1, position};
  return value;
}

DeepSeekAttentionLayerSubmission attention(
    const std::vector<std::uint32_t>& positions) {
  DeepSeekAttentionLayerSubmission value;
  value.kind = DeepSeekCompressedAttentionKind::kRatio128;
  value.query_positions = positions;
  value.compressed_slot_count = 1;
  value.recent_physical_offset = 0;
  value.compressed_physical_offset = 128;
  value.attention = {nullptr, 1, 2, 3, 4, 5, 6, 7, 64, 129};
  return value;
}

TEST(DeepSeekPrefillLayerCoordinatorTest,
     WritesWholeChunkBeforePostingMultiQueryAttention) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio128_append(3, 0).value();
  std::vector<DeepSeekRecentStateSubmission> recent{
      {3, 21, 7, 1, 126}, {3, 22, 7, 1, 127}};
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates{
      update(126), update(127)};
  updates[1].has_completed_slot = true;
  updates[1].ratio128_slot =
      {target, {}, false, 14, 31, 32, 6, 7, 127, 1.0e-6F};
  const std::vector<std::uint32_t> positions{126, 127};
  ASSERT_TRUE(fixture.prefill.launch(
      {recent, updates, attention(positions)}, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_ops.calls, 2U);
  EXPECT_EQ(fixture.state_ops.calls, 2U);
  EXPECT_EQ(fixture.page_ops.calls, 1U);
  EXPECT_EQ(fixture.sparse_ops.calls, 1U);
  EXPECT_EQ(fixture.sparse_ops.query_count, 2U);
}

TEST(DeepSeekPrefillLayerCoordinatorTest,
     BindsDeferredChunkPageBeforeLaunchingAnyToken) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  std::vector<DeepSeekRecentStateSubmission> recent{
      {3, 21, 7, 1, 126}, {3, 22, 7, 1, 127}};
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates{
      update(126), update(127)};
  for (auto& item : updates) {
    item.defer_page_mutation = true;
    item.deferred_page = {31, 0, 32, 0, 1.0e-6F};
  }
  const std::vector<std::uint32_t> positions{126, 127};
  ASSERT_TRUE(fixture.prefill.launch(
      {recent, updates, attention(positions)}, fixture.transaction).ok());
  EXPECT_EQ(fixture.ratio128.reserved_pages(), 1U);
  EXPECT_EQ(fixture.recent_ops.calls, 2U);
  EXPECT_EQ(fixture.state_ops.calls, 2U);
  EXPECT_EQ(fixture.page_ops.calls, 1U);
}

TEST(DeepSeekPrefillLayerCoordinatorTest,
     RejectsLateMalformedTokenBeforeAnyStateWrite) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  std::vector<DeepSeekRecentStateSubmission> recent{
      {3, 21, 7, 1, 125}, {3, 22, 7, 1, 126}};
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates{
      update(125), update(126)};
  updates[1].main_state.kv_projection_f32 = 0;
  const std::vector<std::uint32_t> positions{125, 126};
  auto layer_attention = attention(positions);
  layer_attention.compressed_slot_count = 0;
  EXPECT_FALSE(fixture.prefill.launch(
      {recent, updates, layer_attention}, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_ops.calls, 0U);
  EXPECT_EQ(fixture.state_ops.calls, 0U);
  EXPECT_EQ(fixture.page_ops.calls, 0U);
  EXPECT_EQ(fixture.sparse_ops.calls, 0U);
}

TEST(DeepSeekPrefillLayerCoordinatorTest,
     UpdatesRatio4ChunkBeforeStartingCausalSelection) {
  Fixture fixture;
  ASSERT_TRUE(fixture.transaction.begin(7).ok());
  auto target = fixture.transaction.reserve_ratio4_append(2, 0).value();
  std::vector<DeepSeekRecentStateSubmission> recent{
      {2, 21, 7, 1, 2}, {2, 22, 7, 1, 3}};
  std::vector<DeepSeekCompressedLayerUpdateSubmission> updates(2);
  for (std::uint32_t token = 0; token < 2; ++token) {
    updates[token].ratio = 4;
    updates[token].main_state =
        {2, false, 11, 12, 13, 14, 6, 7, 1, token + 2};
    updates[token].index_state =
        {2, true, 15, 16, 17, 18, 6, 7, 1, token + 2};
  }
  updates[1].has_completed_slot = true;
  updates[1].ratio4_slot =
      {target, {}, false, 14, 18, 31, 32, 33, 34,
       6, 7, 3, 1.0e-6F};
  const std::vector<std::uint32_t> positions{2, 3};
  const std::vector<std::uint32_t> visible{0, 1};
  DeepSeekAttentionLayerSubmission layer_attention;
  layer_attention.kind = DeepSeekCompressedAttentionKind::kRatio4;
  layer_attention.query_positions = positions;
  layer_attention.compressed_slot_count = 1;
  layer_attention.recent_physical_offset = 0;
  layer_attention.compressed_physical_offset = 128;
  layer_attention.selection =
      {41, 42, 43, {44, 45}, visible, 1, 2, 64, 7};
  layer_attention.has_indexer_projection = true;
  layer_attention.indexer_projection =
      {101, 102, 103, 107, 108, 109, 104, 105, 106, 41, 43, 46, 7, 2, 1048576};
  layer_attention.attention =
      {nullptr, 1, 2, 3, 4, 5, 6, 7, 64, 129};
  ASSERT_TRUE(fixture.prefill.launch(
      {recent, updates, layer_attention}, fixture.transaction).ok());
  EXPECT_EQ(fixture.recent_ops.calls, 2U);
  EXPECT_EQ(fixture.state_ops.calls, 4U);
  EXPECT_EQ(fixture.page_ops.calls, 2U);
  EXPECT_EQ(fixture.projection_ops.calls, 1U);
  EXPECT_EQ(fixture.attention.state(),
            DeepSeekAttentionLayerCoordinatorState::kSelectionReady);
  EXPECT_EQ(fixture.sparse_ops.calls, 0U);
}

} }  // namespace pih
