#include "pih/model/deepseek_dense_mhc_stage_submission_assembler.h"

namespace pih {

Result<DeepSeekDenseMhcStageSubmissions>
DeepSeekDenseMhcStageSubmissionAssembler::Assemble(
    DeepSeekStageRange owned_layers,
    std::span<const DeepSeekDenseMhcLayerWeightInput> weights,
    const DeepSeekAttentionProjectionDeviceResources& attention_resources,
    const DeepSeekMhcDeviceResources& mhc_resources,
    const DeepSeekRopeTableDeviceResources& rope_tables,
    std::uint32_t token_count, std::uintptr_t initial_residual_bf16,
    std::uint32_t table_position_count, std::uintptr_t stream) {
  if (owned_layers.first_layer > owned_layers.last_layer ||
      owned_layers.last_layer > 42 || initial_residual_bf16 == 0 ||
      table_position_count > rope_tables.maximum_positions() ||
      rope_tables.context_identity() != attention_resources.context_identity() ||
      rope_tables.device_ordinal() != attention_resources.device_ordinal() ||
      weights.size() !=
          owned_layers.last_layer - owned_layers.first_layer + 1) {
    return Status::InvalidArgument(
        "DeepSeek dense mHC stage submission coverage is invalid");
  }
  DeepSeekDenseMhcStageSubmissions result;
  result.layers.reserve(weights.size());
  std::uintptr_t residual = initial_residual_bf16;
  std::uint64_t generation = 0;
  for (std::size_t offset = 0; offset < weights.size(); ++offset) {
    const auto layer = owned_layers.first_layer +
                       static_cast<std::uint32_t>(offset);
    const auto& binding = weights[offset];
    if (binding.layer != layer ||
        binding.attention.generation != binding.mhc.generation ||
        binding.attention.generation == 0 ||
        (generation != 0 && generation != binding.attention.generation)) {
      return Status::InvalidArgument(
          "DeepSeek dense mHC stage weight generation is invalid");
    }
    generation = binding.attention.generation;
    const auto table = layer <= 1
                           ? rope_tables.view().base_f32
                           : rope_tables.view().yarn_f32;
    if (table == 0) {
      return Status::FailedPrecondition(
          "DeepSeek dense mHC stage lacks required RoPE table");
    }
    auto assembled = DeepSeekDenseMhcLayerSubmissionAssembler::Assemble(
        layer, binding.attention, binding.mhc, attention_resources,
        mhc_resources, token_count, residual, table,
        table_position_count, stream);
    if (!assembled.ok()) return assembled.status();
    residual = assembled->residual_output_bf16;
    result.layers.push_back(std::move(assembled->input));
  }
  result.final_residual_bf16 = residual;
  result.weight_generation = generation;
  return result;
}

}  // namespace pih
