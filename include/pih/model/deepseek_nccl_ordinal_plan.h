#pragma once

#include <cstdint>

#include "pih/model/deepseek_nccl_p2p_plan.h"

namespace pih {

struct DeepSeekNcclOrdinalIdentity final {
  std::uint64_t global_issue_ordinal = 0;
  std::uint64_t rank_local_ordinal = 0;
  std::uint32_t local_global_rank = 0;
  std::uint32_t peer_global_rank = 0;
};

class DeepSeekNcclOrdinalPlan final {
 public:
  static Result<DeepSeekNcclOrdinalIdentity> Create(
      std::uint32_t world_size, std::uint64_t pipeline_plan_sequence,
      std::uint32_t directed_boundary_id, DeepSeekNcclRole role);
};

}  // namespace pih
