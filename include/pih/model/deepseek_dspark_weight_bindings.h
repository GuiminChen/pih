#pragma once

#include <array>
#include <functional>
#include <string_view>

#include "pih/model/deepseek_attention_weight_bindings.h"
#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_mhc_weight_bindings.h"
#include "pih/model/deepseek_resident_weight_arena.h"
#include "pih/model/deepseek_stage_mapping_plan.h"

namespace pih {

struct DeepSeekDsparkSharedExpertWeightBindings final {
  DeepSeekExpertMatrixDeviceView w1;
  DeepSeekExpertMatrixDeviceView w2;
  DeepSeekExpertMatrixDeviceView w3;
};

struct DeepSeekDsparkCommonWeightBindings final {
  DeepSeekDsparkStageId stage = DeepSeekDsparkStageId::kMtp0;
  DeepSeekAttentionWeightBindings attention;
  DeepSeekMhcWeightBindings mhc;
  std::uintptr_t router_weight_bf16 = 0;
  std::uintptr_t router_bias_f32 = 0;
  DeepSeekDsparkSharedExpertWeightBindings shared_expert;
  std::uint64_t generation = 0;
};

struct DeepSeekDsparkStage0BoundaryWeightBindings final {
  std::uintptr_t main_proj_fp8 = 0;
  std::uintptr_t main_proj_scale_ue8m0 = 0;
  std::uintptr_t main_norm_weight_bf16 = 0;
};

struct DeepSeekDsparkStage2HeadWeightBindings final {
  std::uintptr_t norm_weight_bf16 = 0;
  std::uintptr_t markov_embedding_bf16 = 0;
  std::uintptr_t markov_head_bf16 = 0;
  std::uintptr_t confidence_weight_bf16 = 0;
  std::uintptr_t hc_head_fn_f32 = 0;
  std::uintptr_t hc_head_scale_f32 = 0;
  std::uintptr_t hc_head_base_f32 = 0;
};

class DeepSeekDsparkWeightBindings final {
 public:
  using Resolver = std::function<Result<TensorView>(std::string_view)>;

  static Result<DeepSeekDsparkWeightBindings> Resolve(
      DeepSeekStagePlan stage, const Resolver& resolver);
  static Result<DeepSeekDsparkWeightBindings> Resolve(
      DeepSeekStagePlan stage, const DeepSeekResidentWeightArena& arena);

  [[nodiscard]] const DeepSeekDsparkCommonWeightBindings& common(
      DeepSeekDsparkStageId stage) const {
    return common_.at(deepseek_dspark_stage_index(stage));
  }
  [[nodiscard]] const DeepSeekDsparkStage0BoundaryWeightBindings& boundary()
      const noexcept {
    return boundary_;
  }
  [[nodiscard]] const DeepSeekDsparkStage2HeadWeightBindings& head()
      const noexcept {
    return head_;
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return generation_;
  }

 private:
  std::array<DeepSeekDsparkCommonWeightBindings,
             kDeepSeekDsparkStageCount> common_{};
  DeepSeekDsparkStage0BoundaryWeightBindings boundary_;
  DeepSeekDsparkStage2HeadWeightBindings head_;
  std::uint64_t generation_ = 0;
};

}  // namespace pih
