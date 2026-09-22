#pragma once

#include <span>

#include "pih/model/deepseek_attention_layer_coordinator.h"
#include "pih/model/deepseek_recent_state_writer.h"

namespace pih {

struct DeepSeekPrefillLayerSubmission final {
  std::span<const DeepSeekRecentStateSubmission> recent;
  std::span<const DeepSeekCompressedLayerUpdateSubmission> updates;
  DeepSeekAttentionLayerSubmission attention;
};

class DeepSeekPrefillLayerCoordinator final {
 public:
  static Result<DeepSeekPrefillLayerCoordinator> Create(
      DeepSeekRecentStateWriter& recent_writer,
      DeepSeekCompressedLayerUpdateCoordinator& update_coordinator,
      DeepSeekIndexSelectionDriver& selection,
      DeepSeekSparseAttentionDriver& sparse_attention,
      DeepSeekAttentionLayerCoordinator& attention_coordinator);

  Status launch(const DeepSeekPrefillLayerSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekRecentStateWriter* recent_writer_ = nullptr;
  DeepSeekCompressedLayerUpdateCoordinator* update_coordinator_ = nullptr;
  DeepSeekIndexSelectionDriver* selection_ = nullptr;
  DeepSeekSparseAttentionDriver* sparse_attention_ = nullptr;
  DeepSeekAttentionLayerCoordinator* attention_coordinator_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
