#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

struct QwenSemanticObservationIdentityReservation final {
  std::uint64_t first_copy_plan_id;
  std::uint64_t copy_plan_count;
  std::uint64_t frontier_plan_id;
  std::uint64_t next_plan_id;
  std::uint64_t completion_event_generation;
  std::uint64_t next_event_generation;
};

Result<QwenSemanticObservationIdentityReservation>
reserve_qwen_semantic_observation_identity(
    std::uint64_t first_plan_id, std::uint64_t first_event_generation,
    std::uint64_t kv_slice_count);

}  // namespace pih
