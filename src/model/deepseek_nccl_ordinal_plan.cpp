#include "pih/model/deepseek_nccl_ordinal_plan.h"

#include "pih/core/checked_math.h"

namespace pih {

Result<DeepSeekNcclOrdinalIdentity> DeepSeekNcclOrdinalPlan::Create(
    std::uint32_t world_size, std::uint64_t pipeline_plan_sequence,
    std::uint32_t directed_boundary_id, DeepSeekNcclRole role) {
  if (world_size < 2 || world_size > 4 || pipeline_plan_sequence == 0 ||
      directed_boundary_id + 1 >= world_size ||
      static_cast<std::uint8_t>(role) >
          static_cast<std::uint8_t>(DeepSeekNcclRole::kRecv)) {
    return Status::InvalidArgument(
        "DeepSeek NCCL ordinal topology is invalid");
  }
  auto global_base = checked_mul_u64(
      pipeline_plan_sequence - 1, world_size - 1);
  if (!global_base.ok()) return global_base.status();
  auto global = checked_add_u64(*global_base, directed_boundary_id + 1);
  if (!global.ok()) return global.status();

  const bool send = role == DeepSeekNcclRole::kSend;
  const std::uint32_t local_rank =
      send ? directed_boundary_id : directed_boundary_id + 1;
  const std::uint32_t peer_rank =
      send ? directed_boundary_id + 1 : directed_boundary_id;
  const std::uint64_t degree =
      (local_rank == 0 || local_rank + 1 == world_size) ? 1 : 2;
  const std::uint64_t position = send && local_rank != 0 ? 2 : 1;
  auto local_base = checked_mul_u64(pipeline_plan_sequence - 1, degree);
  if (!local_base.ok()) return local_base.status();
  auto local = checked_add_u64(*local_base, position);
  if (!local.ok()) return local.status();
  return DeepSeekNcclOrdinalIdentity{*global, *local, local_rank, peer_rank};
}

}  // namespace pih
