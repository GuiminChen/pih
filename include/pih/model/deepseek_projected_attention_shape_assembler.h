#pragma once

#include "pih/model/deepseek_attention_plan_input_shape_assembler.h"
#include "pih/model/deepseek_attention_device_scratch_resources.h"
#include "pih/model/deepseek_compressor_projection_device_resources.h"
#include "pih/model/deepseek_compressor_weight_bindings.h"

namespace pih {

class DeepSeekProjectedAttentionShapeAssembler final {
 public:
  static Status BindSparseRuntime(
      DeepSeekAttentionLayerPlanShapeSeed& shape,
      std::uintptr_t sparse_query_bf16, std::uintptr_t sparse_kv_bf16,
      std::uintptr_t sparse_output_bf16,
      std::uintptr_t attention_sink_f32,
      const DeepSeekAttentionDeviceScratchResources& attention_scratch);
  static Status Populate(
      DeepSeekAttentionLayerPlanShapeSeed& shape,
      std::uintptr_t hidden_rows_bf16, std::uintptr_t qr_bf16,
      std::uintptr_t positions_u32,
      const DeepSeekCompressorWeightBindings& weights,
      const DeepSeekCompressorProjectionDeviceResources& scratch,
      const DeepSeekAttentionDeviceScratchResources& attention_scratch,
      std::uintptr_t compressor_error_u32, std::uintptr_t stream,
      std::uintptr_t cos_sin_cache_f32, std::uint32_t table_position_count);
};

}  // namespace pih
