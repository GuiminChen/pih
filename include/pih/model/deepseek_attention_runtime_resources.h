#pragma once

#include <memory>
#include <vector>

#include "pih/model/deepseek_attention_work_factory.h"
#include "pih/model/deepseek_indexer_projection_coordinator.h"

namespace pih {

struct DeepSeekAttentionLayerRuntimeInput final {
  std::uint32_t layer = 0;
  DeepSeekFixedStateLayout fixed_layout;
  DeepSeekAttentionPageArena page_arena;
  std::uint32_t* compressor_error_flag = nullptr;
  std::uint32_t* page_error_flag = nullptr;
  std::uint32_t* indexer_projection_error_flag = nullptr;
  DeepSeekIndexSelectionHostStaging index_staging;
  std::uintptr_t index_completion_event = 0;
  DeepSeekSparseAttentionHostStaging sparse_staging;
};

struct DeepSeekAttentionRuntimeOperations final {
  DeepSeekRecentStateOperations* recent = nullptr;
  DeepSeekCompressorStateOperations* compressor = nullptr;
  DeepSeekCompressedPageOperations* page = nullptr;
  DeepSeekIndexerProjectionOperations* indexer_projection = nullptr;
  DeepSeekIndexSelectionOperations* index_selection = nullptr;
  DeepSeekSparseAttentionOperations* sparse_attention = nullptr;
};

class DeepSeekAttentionRuntimeResources final {
 public:
  static Result<DeepSeekAttentionRuntimeResources> Create(
      DeepSeekStageRange owned_layers,
      DeepSeekAttentionRuntimeOperations operations,
      std::vector<DeepSeekAttentionLayerRuntimeInput> layers);

  ~DeepSeekAttentionRuntimeResources();
  DeepSeekAttentionRuntimeResources(
      DeepSeekAttentionRuntimeResources&&) noexcept;
  DeepSeekAttentionRuntimeResources& operator=(
      DeepSeekAttentionRuntimeResources&&) noexcept;
  DeepSeekAttentionRuntimeResources(
      const DeepSeekAttentionRuntimeResources&) = delete;
  DeepSeekAttentionRuntimeResources& operator=(
      const DeepSeekAttentionRuntimeResources&) = delete;

  std::vector<DeepSeekAttentionLayerRuntimeResources>
  borrow_layer_resources() noexcept;

 private:
  struct LayerOwner;
  DeepSeekAttentionRuntimeResources() = default;
  DeepSeekStageRange owned_layers_;
  std::vector<std::unique_ptr<LayerOwner>> layers_;
};

}  // namespace pih
