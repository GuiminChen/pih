#include "pih/model/deepseek_dspark_target_hidden_capture_assembler.h"

#include <array>

namespace pih {

Status DeepSeekDsparkTargetHiddenCaptureAssembler::Bind(
    DeepSeekStagePlan stage,
    const DeepSeekDsparkDeviceResources& dspark_resources,
    std::uint32_t token_count,
    std::span<DeepSeekDenseMhcLayerSubmissionInput> layers) {
  const auto device = dspark_resources.view();
  if (!stage.owns_dspark || !stage.owns_lm_head ||
      stage.layers.first_layer > 40 || stage.layers.last_layer != 42 ||
      token_count == 0 ||
      token_count > dspark_resources.maximum_tokens() ||
      device.target_hidden_bf16 == 0 ||
      layers.size() !=
          stage.layers.last_layer - stage.layers.first_layer + 1) {
    return Status::InvalidArgument(
        "DeepSeek DSpark target hidden capture identity is invalid");
  }
  std::array<bool, 3> seen{};
  for (std::size_t offset = 0; offset < layers.size(); ++offset) {
    auto& layer = layers[offset];
    if (layer.layer != stage.layers.first_layer + offset) {
      return Status::InvalidArgument(
          "DeepSeek DSpark target hidden stage coverage is invalid");
    }
    if (layer.layer < 40) continue;
    if (layer.layer > 42 || layer.mhc_feed_forward.layer_id != layer.layer ||
        layer.mhc_feed_forward.kind !=
            DeepSeekMhcBranchKind::kFeedForward ||
        layer.mhc_feed_forward.token_count != token_count ||
        layer.mhc_feed_forward.target_hidden_bf16 != 0 ||
        layer.mhc_feed_forward.target_stage_index != 0) {
      return Status::InvalidArgument(
          "DeepSeek DSpark target hidden capture layer is invalid");
    }
    const auto index = layer.layer - 40;
    if (seen[index]) {
      return Status::InvalidArgument(
          "DeepSeek DSpark target hidden capture layer is duplicated");
    }
    auto candidate = layer.mhc_feed_forward;
    candidate.target_hidden_bf16 = device.target_hidden_bf16;
    candidate.target_stage_index = index;
    auto status = validate_deepseek_mhc_sequence_submission(
        candidate);
    if (!status.ok()) return status;
    seen[index] = true;
  }
  if (!seen[0] || !seen[1] || !seen[2]) {
    return Status::InvalidArgument(
        "DeepSeek DSpark target hidden capture coverage is incomplete");
  }
  for (auto& layer : layers) {
    if (layer.layer < 40) continue;
    layer.mhc_feed_forward.target_hidden_bf16 =
        device.target_hidden_bf16;
    layer.mhc_feed_forward.target_stage_index = layer.layer - 40;
  }
  return Status::Ok();
}

}  // namespace pih
