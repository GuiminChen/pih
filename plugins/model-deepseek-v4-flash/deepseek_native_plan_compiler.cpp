#include "pih/model/deepseek_native_plan_compiler.h"

namespace pih {
namespace {

bool same_stage(const DeepSeekStagePlan& left,
                const DeepSeekStagePlan& right) {
  return left.rank == right.rank && left.layers == right.layers &&
         left.owns_embedding == right.owns_embedding &&
         left.owns_lm_head == right.owns_lm_head &&
         left.owns_dspark == right.owns_dspark;
}

}  // namespace

Result<DeepSeekNativePlanCompiler> DeepSeekNativePlanCompiler::Create(
    DeepSeekPipelinePlan topology,
    std::vector<DeepSeekRankPlanCompiler> rank_compilers) {
  if (rank_compilers.size() != topology.world_size()) {
    return Status::InvalidArgument(
        "DeepSeek native plan compiler rank count differs from topology");
  }
  for (std::uint32_t rank = 0; rank < topology.world_size(); ++rank) {
    if (!same_stage(rank_compilers[rank].stage(), topology.rank(rank))) {
      return Status::InvalidArgument(
          "DeepSeek native plan compiler rank stage differs from topology");
    }
  }
  DeepSeekNativePlanCompiler result;
  result.topology_ = std::move(topology);
  result.rank_compilers_ = std::move(rank_compilers);
  return result;
}

Result<DeepSeekNativeRankComputePlan> DeepSeekNativePlanCompiler::compile(
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRankPlanCompilerInput> rank_inputs) {
  if (rank_inputs.size() != rank_compilers_.size()) {
    return Status::InvalidArgument(
        "DeepSeek native plan compiler input rank count is invalid");
  }
  std::vector<DeepSeekRankComputePlanWork> rank_work;
  rank_work.reserve(rank_compilers_.size());
  for (std::size_t rank = 0; rank < rank_compilers_.size(); ++rank) {
    auto work = rank_compilers_[rank].compile(
        descriptor, std::move(rank_inputs[rank]));
    if (!work.ok()) return work.status();
    rank_work.push_back(std::move(*work));
  }
  return DeepSeekNativeRankComputePlan::Create(
      descriptor, topology_, std::move(rank_work));
}

}  // namespace pih
