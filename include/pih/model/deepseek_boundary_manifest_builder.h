#pragma once

#include "pih/model/deepseek_boundary_send_source.h"
#include "pih/model/deepseek_nccl_ordinal_plan.h"
#include "pih/model/deepseek_pipeline_transaction.h"

namespace pih {

class DeepSeekBoundaryManifestBuilder final {
 public:
  static Result<DeepSeekNcclP2pManifest> CreateSend(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::uint32_t world_size, std::uint32_t directed_boundary_id,
      std::uint64_t communicator_generation, std::uint32_t wire_token_count,
      const DeepSeekBoundarySendSource& source);

  static Result<DeepSeekNcclP2pManifest> CreateRecv(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      std::uint32_t world_size, std::uint32_t directed_boundary_id,
      std::uint64_t communicator_generation, std::uint32_t wire_token_count,
      Buffer& credit_slot, std::uint64_t buffer_owner_id,
      std::uintptr_t context_identity);
};

}  // namespace pih
