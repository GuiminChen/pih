#pragma once

#include "pih/model/deepseek_compressed_layer_update_coordinator.h"
#include "pih/model/deepseek_compressor_projection_device_resources.h"
#include "pih/model/deepseek_compressor_weight_bindings.h"

namespace pih {

class DeepSeekProjectedCompressorUpdateAssembler final {
 public:
  static Result<DeepSeekCompressedLayerUpdateSubmission> AssembleDecode(
      std::uint32_t layer, DeepSeekCompressedAttentionKind kind,
      std::uintptr_t hidden_row_bf16,
      const DeepSeekCompressorWeightBindings& weights,
      const DeepSeekCompressorProjectionSlice& scratch,
      std::uintptr_t device_error_u32, std::uintptr_t stream,
      std::uint32_t absolute_position);
};

}  // namespace pih
