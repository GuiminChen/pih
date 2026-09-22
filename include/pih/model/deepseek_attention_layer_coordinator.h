#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/model/deepseek_index_selection_driver.h"
#include "pih/model/deepseek_indexer_projection_coordinator.h"
#include "pih/model/deepseek_compressed_layer_update_coordinator.h"
#include "pih/model/deepseek_sparse_attention_driver.h"
#include "pih/model/deepseek_recent_state_writer.h"

namespace pih {

enum class DeepSeekCompressedAttentionKind : std::uint8_t {
  kRatio4,
  kRatio128,
  kRecentOnly,
};

enum class DeepSeekAttentionLayerCoordinatorState : std::uint8_t {
  kIdle,
  kSelectionReady,
  kSelectionInflight,
  kAttentionPosted,
  kPoisoned,
};

struct DeepSeekAttentionLayerSubmission final {
  DeepSeekCompressedAttentionKind kind =
      DeepSeekCompressedAttentionKind::kRatio4;
  std::span<const std::uint32_t> query_positions;
  std::uint32_t compressed_slot_count = 0;
  std::int32_t recent_physical_offset = 0;
  std::int32_t compressed_physical_offset = 0;
  bool has_indexer_projection = false;
  DeepSeekIndexerProjectionLaunch indexer_projection;
  DeepSeekIndexSelectionSubmission selection;
  DeepSeekSparseAttentionSubmission attention;
};

class DeepSeekAttentionLayerCoordinator final {
 public:
  static Result<DeepSeekAttentionLayerCoordinator> Create(
      DeepSeekIndexerProjectionCoordinator& indexer_projection,
      DeepSeekIndexSelectionDriver& selection,
      DeepSeekSparseAttentionDriver& attention);
  static Result<DeepSeekAttentionLayerCoordinator> CreatePaged(
      std::uint32_t layer_id, DeepSeekFixedStateLayout fixed_layout,
      DeepSeekAttentionPageArena page_arena,
      DeepSeekIndexerProjectionCoordinator& indexer_projection,
      DeepSeekIndexSelectionDriver& selection,
      DeepSeekSparseAttentionDriver& attention);

  Status begin(const DeepSeekAttentionLayerSubmission& submission,
               DeepSeekAttentionSequenceTransaction& transaction);
  Status begin_decode(
      const DeepSeekRecentStateSubmission& recent,
      const DeepSeekCompressedLayerUpdateSubmission& update,
      const DeepSeekAttentionLayerSubmission& attention,
      DeepSeekRecentStateWriter& recent_writer,
      DeepSeekCompressedLayerUpdateCoordinator& update_coordinator,
      DeepSeekAttentionSequenceTransaction& transaction);
  Status launch_next_selection_tile();
  Result<DeepSeekExpertAsyncStatus> poll_selection_tile();
  Status reset();

  [[nodiscard]] DeepSeekAttentionLayerCoordinatorState state() const noexcept {
    return state_;
  }

 private:
  Status assemble_and_launch(
      std::span<const std::vector<std::uint32_t>> selected);

  DeepSeekIndexerProjectionCoordinator* indexer_projection_ = nullptr;
  DeepSeekIndexSelectionDriver* selection_ = nullptr;
  DeepSeekSparseAttentionDriver* attention_ = nullptr;
  DeepSeekAttentionSequenceTransaction* transaction_ = nullptr;
  DeepSeekAttentionLayerSubmission submission_;
  std::vector<std::uint32_t> query_positions_;
  DeepSeekAttentionLayerCoordinatorState state_ =
      DeepSeekAttentionLayerCoordinatorState::kIdle;
  std::uint32_t layer_id_ = 0;
  bool paged_selection_ = false;
  DeepSeekFixedStateLayout fixed_layout_;
  DeepSeekAttentionPageArena page_arena_;
};

}  // namespace pih
