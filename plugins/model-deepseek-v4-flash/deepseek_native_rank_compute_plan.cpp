#include "pih/model/deepseek_native_rank_compute_plan.h"

#include <array>


namespace pih {
namespace {

template <typename Work>
bool exact_layer_coverage(
    std::span<const Work> work, DeepSeekStageRange owned) {
  const auto expected = owned.last_layer - owned.first_layer + 1;
  if (work.size() != expected) return false;
  std::array<bool, 43> seen{};
  for (const auto& item : work) {
    if (item.layer < owned.first_layer || item.layer > owned.last_layer ||
        seen[item.layer]) {
      return false;
    }
    seen[item.layer] = true;
  }
  return true;
}

bool exact_router_coverage(
    const DeepSeekRankComputePlanWork& work, DeepSeekStageRange owned) {
  std::array<bool, 43> seen{};
  std::uint32_t count = 0;
  for (const auto& item : work.hash_router) {
    if (item.layer >= 3 || item.layer < owned.first_layer ||
        item.layer > owned.last_layer || seen[item.layer]) return false;
    seen[item.layer] = true;
    ++count;
  }
  for (const auto& item : work.learned_router) {
    if (item.layer < 3 || item.layer < owned.first_layer ||
        item.layer > owned.last_layer || seen[item.layer]) return false;
    seen[item.layer] = true;
    ++count;
  }
  return count == owned.last_layer - owned.first_layer + 1;
}

Status validate_rank(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    const DeepSeekStagePlan& stage,
    const DeepSeekRankComputePlanWork& work) {
  if (work.lifetime_owner == nullptr ||
      !exact_router_coverage(work, stage.layers) ||
      !exact_layer_coverage(work.dense_attention, stage.layers) ||
      !exact_layer_coverage(work.mhc_attention, stage.layers) ||
      !exact_layer_coverage(work.mhc_feed_forward, stage.layers)) {
    return Status::InvalidArgument(
        "DeepSeek native rank work does not cover its stage");
  }
  const bool decode = descriptor.phase == DeepSeekPlanPhase::kDecode;
  if (decode ? (!exact_layer_coverage(work.decode_attention, stage.layers) ||
                !work.chunk_attention.empty())
             : (!exact_layer_coverage(work.chunk_attention, stage.layers) ||
                !work.decode_attention.empty())) {
    return Status::InvalidArgument(
        "DeepSeek native rank attention differs from plan phase");
  }
  const bool endpoint = stage.owns_embedding || stage.owns_lm_head;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  const bool runs_dspark = stage.owns_dspark &&
      (descriptor.phase == DeepSeekPlanPhase::kPrefill ||
       descriptor.phase == DeepSeekPlanPhase::kDecode);
  const bool dspark_matches =
      (work.dspark != nullptr) == runs_dspark &&
      (work.dspark_mtp.size() == kDeepSeekDsparkStageCount) == runs_dspark;
#else
  const bool dspark_matches = !stage.owns_dspark;
#endif
  if (work.endpoint.empty() == endpoint ||
      !dspark_matches) {
    return Status::InvalidArgument(
        "DeepSeek native endpoint or DSpark ownership differs from stage");
  }
  return Status::Ok();
}

}  // namespace

Result<DeepSeekNativeRankComputePlan>
DeepSeekNativeRankComputePlan::Create(
    DeepSeekPipelinePlanDescriptor descriptor,
    const DeepSeekPipelinePlan& topology,
    std::vector<DeepSeekRankComputePlanWork> rank_work) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.token_count == 0 || descriptor.sequence_count == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      rank_work.size() != topology.world_size()
      || topology.world_size() != 1
      ) {
    return Status::InvalidArgument(
        "DeepSeek native compute plan identity or rank count is invalid");
  }
  for (std::uint32_t rank = 0; rank < topology.world_size(); ++rank) {
    const auto status = validate_rank(
        descriptor, topology.rank(rank), rank_work[rank]);
    if (!status.ok()) return status;
  }
  return DeepSeekNativeRankComputePlan(descriptor, std::move(rank_work));
}

}  // namespace pih
