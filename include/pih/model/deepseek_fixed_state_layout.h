#pragma once

#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_stage_identity.h"
#endif
#include "pih/model/deepseek_expert_compute_arena.h"

namespace pih {

inline constexpr std::uint64_t kDeepSeekRecentStateBytes =
    128ULL * 512ULL * 2ULL;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
inline constexpr std::uint64_t kDeepSeekDsparkRecentStateBytes =
    kDeepSeekRecentStateBytes;
#endif

enum class DeepSeekFixedLayerKind : std::uint8_t {
  kRecentOnly,
  kRatio4,
  kRatio128,
};

struct DeepSeekRelativeStateSpan final {
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct DeepSeekFixedLayerStateDescriptor final {
  std::uint32_t layer_id = 0;
  DeepSeekFixedLayerKind kind = DeepSeekFixedLayerKind::kRecentOnly;
  DeepSeekRelativeStateSpan recent_bf16;
  DeepSeekRelativeStateSpan main_kv_state_f32;
  DeepSeekRelativeStateSpan main_score_state_f32;
  DeepSeekRelativeStateSpan index_kv_state_f32;
  DeepSeekRelativeStateSpan index_score_state_f32;
};

struct DeepSeekFixedLayerDeviceView final {
  std::uint32_t layer_id = 0;
  DeepSeekFixedLayerKind kind = DeepSeekFixedLayerKind::kRecentOnly;
  DeepSeekExpertArenaSpan recent_bf16;
  DeepSeekExpertArenaSpan main_kv_state_f32;
  DeepSeekExpertArenaSpan main_score_state_f32;
  DeepSeekExpertArenaSpan index_kv_state_f32;
  DeepSeekExpertArenaSpan index_score_state_f32;
};

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
struct DeepSeekDsparkFixedStageStateDescriptor final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekRelativeStateSpan recent_bf16;
};

struct DeepSeekDsparkFixedStageDeviceView final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekExpertArenaSpan recent_bf16;
};
#endif

class DeepSeekFixedStateLayout final {
 public:
  static Result<DeepSeekFixedStateLayout> Build(
      std::span<const std::uint32_t> owned_main_layers,
      bool include_dspark);

  Result<DeepSeekFixedLayerDeviceView> Resolve(
      std::uint32_t layer_id, DeepSeekExpertArenaSpan bank) const;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  Result<DeepSeekDsparkFixedStageDeviceView> ResolveDspark(
      DeepSeekDsparkStageId stage, DeepSeekExpertArenaSpan bank) const;
#endif
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }
  [[nodiscard]] std::span<const DeepSeekFixedLayerStateDescriptor>
  descriptors() const noexcept {
    return descriptors_;
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] std::span<const DeepSeekDsparkFixedStageStateDescriptor>
  dspark_descriptors() const noexcept {
    return dspark_descriptors_;
  }
#endif

 private:
  std::vector<DeepSeekFixedLayerStateDescriptor> descriptors_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::vector<DeepSeekDsparkFixedStageStateDescriptor> dspark_descriptors_;
#endif
  std::uint64_t total_bytes_ = 0;
};

}  // namespace pih
