#pragma once

#include <memory>
#include <span>
#include <vector>

#include "pih/model/deepseek_nccl_endpoint_manifest_plan.h"
#include "pih/model/deepseek_nccl_generation_boundary_assembly.h"

namespace pih {

struct DeepSeekNcclBoundaryBootstrapConfig final {
  std::uint64_t engine_epoch = 0;
  std::uint64_t communicator_generation = 0;
  std::uint64_t config_identity = 0;
  std::uint32_t world_size = 0;
  std::uint32_t maximum_wire_tokens = 0;
  std::uint64_t first_lease_id = 0;
  std::uint64_t edge_timeout_ns = 0;
  std::uint64_t set_timeout_ns = 0;
};

class DeepSeekNcclGenerationWarmup {
 public:
  virtual ~DeepSeekNcclGenerationWarmup() = default;
  virtual Result<std::vector<DeepSeekNcclWarmupReceipt>> run(
      DeepSeekNcclCommunicatorGeneration& generation,
      const DeepSeekNcclEndpointManifestPlan& manifests,
      std::uint32_t maximum_wire_tokens) = 0;
};

class DeepSeekNcclBoundaryBootstrap final {
 public:
  static Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>> Build(
      DeepSeekNcclBoundaryBootstrapConfig config,
      std::span<const DeepSeekNcclRankEndpointIdentity> endpoint_identities,
      std::span<DeepSeekNcclCApi* const> endpoint_apis,
      std::span<const DeepSeekNcclReleaseConfig> release_configs,
      DeepSeekNcclUniqueIdSource& unique_id_source,
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekNcclGenerationWarmup& warmup,
      std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
          device_resources,
      std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
      std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
          driver_factories);
  static Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
  BuildWithDependencies(
      DeepSeekNcclBoundaryBootstrapConfig config,
      std::span<const DeepSeekNcclRankEndpointIdentity> endpoint_identities,
      std::span<DeepSeekNcclCApi* const> endpoint_apis,
      std::span<const DeepSeekNcclReleaseConfig> release_configs,
      DeepSeekNcclUniqueIdSource& unique_id_source,
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekNcclGenerationWarmup& warmup,
      std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
          device_resources,
      std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
      std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
          driver_factories,
      std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies);
};

}  // namespace pih
