#pragma once

#include <span>
#include <vector>

#include "pih/model/deepseek_dense_mhc_layer_submission_assembler.h"
#include "pih/model/deepseek_rope_table_device_resources.h"

namespace pih {

struct DeepSeekDenseMhcLayerWeightInput final {
  std::uint32_t layer = 0;
  DeepSeekAttentionWeightBindings attention;
  DeepSeekMhcWeightBindings mhc;
};

struct DeepSeekDenseMhcStageSubmissions final {
  std::vector<DeepSeekDenseMhcLayerSubmissionInput> layers;
  std::uintptr_t final_residual_bf16 = 0;
  std::uint64_t weight_generation = 0;
};

class DeepSeekDenseMhcStageSubmissionAssembler final {
 public:
  static Result<DeepSeekDenseMhcStageSubmissions> Assemble(
      DeepSeekStageRange owned_layers,
      std::span<const DeepSeekDenseMhcLayerWeightInput> weights,
      const DeepSeekAttentionProjectionDeviceResources& attention_resources,
      const DeepSeekMhcDeviceResources& mhc_resources,
      const DeepSeekRopeTableDeviceResources& rope_tables,
      std::uint32_t token_count, std::uintptr_t initial_residual_bf16,
      std::uint32_t table_position_count, std::uintptr_t stream);
};

}  // namespace pih
