#pragma once

#include <cstdint>
#include <string>

#include "pih/core/result.h"

namespace pih {

inline constexpr char kDeepSeekDsparkThreeStageAbi[] =
    "deepseek_dspark_three_stage_v1";
inline constexpr std::uint32_t kDeepSeekDsparkStageCount = 3;

enum class DeepSeekDsparkStageId : std::uint8_t {
  kMtp0 = 0,
  kMtp1 = 1,
  kMtp2 = 2,
};

[[nodiscard]] constexpr bool is_valid_deepseek_dspark_stage(
    DeepSeekDsparkStageId stage) noexcept {
  return static_cast<std::uint32_t>(stage) < kDeepSeekDsparkStageCount;
}

[[nodiscard]] constexpr std::uint32_t deepseek_dspark_stage_index(
    DeepSeekDsparkStageId stage) noexcept {
  return static_cast<std::uint32_t>(stage);
}

inline Result<DeepSeekDsparkStageId> deepseek_dspark_stage_id(
    std::uint32_t index) {
  if (index >= kDeepSeekDsparkStageCount) {
    return Status::InvalidArgument("DeepSeek DSpark stage id is invalid");
  }
  return static_cast<DeepSeekDsparkStageId>(index);
}

inline Result<std::string> deepseek_dspark_namespace(
    DeepSeekDsparkStageId stage) {
  if (!is_valid_deepseek_dspark_stage(stage)) {
    return Status::InvalidArgument("DeepSeek DSpark stage id is invalid");
  }
  return "mtp." + std::to_string(deepseek_dspark_stage_index(stage));
}

}  // namespace pih
