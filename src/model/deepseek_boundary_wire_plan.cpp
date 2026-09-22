#include "pih/model/deepseek_boundary_wire_plan.h"

namespace pih {

Result<DeepSeekBoundaryWirePlan> DeepSeekBoundaryWirePlan::Create(
    DeepSeekPipelinePlanDescriptor descriptor, std::uint32_t world_size,
    std::span<const DeepSeekRankComputePlanWork> rank_work) {
  if (descriptor.engine_epoch == 0 || descriptor.plan_sequence == 0 ||
      descriptor.sequence_count == 0 || world_size == 0 || world_size > 4 ||
      rank_work.size() != world_size) {
    return Status::InvalidArgument(
        "DeepSeek boundary wire plan identity is invalid");
  }
  if (world_size == 1) {
    if (rank_work.front().outgoing_boundary.has_value()) {
      return Status::InvalidArgument(
          "DeepSeek PP1 cannot carry an outgoing boundary source");
    }
    return DeepSeekBoundaryWirePlan(0, 0);
  }

  std::uint32_t wire_tokens = 0;
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    const auto& source = rank_work[rank].outgoing_boundary;
    const bool required = rank + 1 < world_size;
    if (source.has_value() != required) {
      return Status::InvalidArgument(
          "DeepSeek outgoing boundary sources differ from PP topology");
    }
    if (!source.has_value()) continue;
    if (wire_tokens == 0) wire_tokens = source->token_count();
    if (source->token_count() != wire_tokens) {
      return Status::InvalidArgument(
          "DeepSeek outgoing boundary sources disagree on wire tokens");
    }
  }
  if (wire_tokens == 0) {
    return Status::InvalidArgument(
        "DeepSeek distributed boundary wire count must be positive");
  }
  if (descriptor.phase == DeepSeekPlanPhase::kDrain) {
    if (descriptor.token_count != 0) {
      return Status::InvalidArgument(
          "DeepSeek drain descriptor must keep logical token count zero");
    }
  } else if (descriptor.token_count == 0 ||
             descriptor.token_count != wire_tokens) {
    return Status::InvalidArgument(
        "DeepSeek boundary wire count differs from compute descriptor");
  }
  return DeepSeekBoundaryWirePlan(wire_tokens, world_size - 1);
}

}  // namespace pih
