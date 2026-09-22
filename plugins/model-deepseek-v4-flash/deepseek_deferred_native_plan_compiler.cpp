#include "pih/model/deepseek_deferred_native_plan_compiler.h"

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

Result<DeepSeekDeferredNativePlanCompiler>
DeepSeekDeferredNativePlanCompiler::Create(
    DeepSeekPipelinePlan topology,
    DeepSeekPipelinePlanDescriptor descriptor,
    std::vector<DeepSeekRankPlanCompiler> rank_compilers,
    std::vector<std::unique_ptr<DeepSeekDeferredRankPlanInputAssembler>>
        input_assemblers) {
  const auto world_size = topology.world_size();
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.phase == DeepSeekPlanPhase::kDrain ||
      descriptor.token_count == 0 || descriptor.sequence_count != 1 ||
      rank_compilers.size() != world_size ||
      input_assemblers.size() != world_size) {
    return Status::InvalidArgument(
        "DeepSeek deferred native compiler identity is invalid");
  }
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (input_assemblers[rank] == nullptr ||
        !same_stage(rank_compilers[rank].stage(), topology.rank(rank))) {
      return Status::InvalidArgument(
          "DeepSeek deferred native compiler rank differs from topology");
    }
  }
  DeepSeekDeferredNativePlanCompiler result;
  result.topology_ = std::move(topology);
  result.descriptor_ = descriptor;
  result.rank_compilers_ = std::move(rank_compilers);
  result.input_assemblers_ = std::move(input_assemblers);
  result.compiled_.assign(world_size, false);
  return result;
}


Result<DeepSeekRankComputePlanWork>
DeepSeekDeferredNativePlanCompiler::compile(
    std::uint32_t rank, std::uintptr_t incoming_activation_bf16) {
  if (failure_.has_value()) return *failure_;
  if (rank >= rank_compilers_.size() || compiled_[rank] ||
      (rank == 0) != (incoming_activation_bf16 == 0)) {
    return Status::InvalidArgument(
        "DeepSeek deferred compile rank or incoming activation is invalid");
  }
  auto input = input_assemblers_[rank]->assemble(
      incoming_activation_bf16);
  if (!input.ok()) {
    failure_ = input.status();
    return *failure_;
  }
  auto work = rank_compilers_[rank].compile(descriptor_, std::move(*input));
  if (!work.ok()) {
    failure_ = work.status();
    return *failure_;
  }
  compiled_[rank] = true;
  return work;
}

}  // namespace pih
