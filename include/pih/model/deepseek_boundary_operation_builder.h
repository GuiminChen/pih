#pragma once

#include <array>

#include "pih/model/deepseek_boundary_manifest_builder.h"
#include "pih/model/deepseek_rank_boundary_device_resources.h"
#include "pih/model/deepseek_tracked_boundary_operation.h"

namespace pih {

class DeepSeekBoundaryOperationBuilder final {
 public:
  static Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>> CreateSend(
      DeepSeekPipelineTransaction& transaction, std::uint32_t world_size,
      std::uint32_t directed_boundary_id,
      std::uint64_t communicator_generation, std::uint32_t wire_token_count,
      const DeepSeekBoundarySendSource& source,
      DeepSeekRankBoundaryDeviceResources& resources,
      DriverStreamHandle boundary_stream,
      DeepSeekNcclOperationSequencer& sequencer,
      DeepSeekBoundaryCreditTracker& tracker,
      std::uint64_t submit_ns, std::uint64_t deadline_ns);

  static Result<std::unique_ptr<DeepSeekTrackedBoundaryOperation>> CreateRecv(
      DeepSeekPipelineTransaction& transaction, std::uint32_t world_size,
      std::uint32_t directed_boundary_id,
      std::uint64_t communicator_generation, std::uint32_t wire_token_count,
      DeepSeekRankBoundaryDeviceResources& resources,
      const std::array<std::uint64_t,
                       DeepSeekRankBoundaryDeviceResources::kCreditCount>&
          buffer_owner_ids,
      std::uintptr_t context_identity, DriverStreamHandle boundary_stream,
      DeepSeekNcclOperationSequencer& sequencer,
      DeepSeekBoundaryCreditTracker& tracker,
      std::uint64_t submit_ns, std::uint64_t deadline_ns);
};

}  // namespace pih
