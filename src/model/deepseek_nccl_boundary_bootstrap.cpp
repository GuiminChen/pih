#include "pih/model/deepseek_nccl_boundary_bootstrap.h"

namespace pih {
namespace {

Status validate_inputs(
    const DeepSeekNcclBoundaryBootstrapConfig& config,
    std::span<const DeepSeekNcclRankEndpointIdentity> endpoint_identities,
    std::span<DeepSeekNcclCApi* const> endpoint_apis,
    std::span<const DeepSeekNcclReleaseConfig> release_configs,
    const std::vector<DeepSeekRankBoundaryFactoryIdentity>& rank_identities,
    const std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>&
        device_resources,
    const std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>>& clocks,
    const std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>&
        driver_factories) {
  const auto world_size = config.world_size;
  if (config.engine_epoch == 0 || config.communicator_generation == 0 ||
      config.config_identity == 0 || world_size < 2 || world_size > 4 ||
      config.maximum_wire_tokens == 0 || config.first_lease_id == 0 ||
      config.edge_timeout_ns == 0 || config.set_timeout_ns == 0 ||
      endpoint_identities.size() != world_size ||
      endpoint_apis.size() != world_size ||
      release_configs.size() != world_size ||
      rank_identities.size() != world_size ||
      device_resources.size() != world_size || clocks.size() != world_size ||
      driver_factories.size() != world_size) {
    return Status::InvalidArgument(
        "DeepSeek NCCL boundary bootstrap topology is invalid");
  }
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (endpoint_identities[rank].rank != rank ||
        endpoint_identities[rank].device_identity == 0 ||
        endpoint_identities[rank].context_identity == 0 ||
        endpoint_apis[rank] == nullptr || device_resources[rank] == nullptr ||
        clocks[rank] == nullptr || driver_factories[rank] == nullptr ||
        rank_identities[rank].engine_epoch != config.engine_epoch ||
        rank_identities[rank].communicator_generation !=
            config.communicator_generation ||
        rank_identities[rank].rank != rank ||
        rank_identities[rank].world_size != world_size ||
        rank_identities[rank].context_identity !=
            endpoint_identities[rank].context_identity) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL boundary bootstrap rank identity is inconsistent");
    }
  }
  return Status::Ok();
}

}  // namespace

Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
DeepSeekNcclBoundaryBootstrap::Build(
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
        driver_factories) {
  class EmptyDependencies final
      : public DeepSeekNcclGenerationDependencyOwner {};
  return BuildWithDependencies(
      config, endpoint_identities, endpoint_apis, release_configs,
      unique_id_source, pinned_allocator, warmup,
      std::move(rank_identities), std::move(device_resources),
      std::move(clocks), std::move(driver_factories),
      std::make_unique<EmptyDependencies>());
}

Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
DeepSeekNcclBoundaryBootstrap::BuildWithDependencies(
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
    std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies) {
  if (dependencies == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek NCCL boundary bootstrap dependencies are null");
  }
  auto valid = validate_inputs(config, endpoint_identities, endpoint_apis,
                               release_configs, rank_identities,
                               device_resources, clocks, driver_factories);
  if (!valid.ok()) return valid;
  auto capabilities = DeepSeekNcclBootstrapCapabilityIssuer::Issue(
      config.engine_epoch, config.world_size, config.first_lease_id,
      unique_id_source, pinned_allocator);
  if (!capabilities.ok()) return capabilities.status();
  auto manifests = DeepSeekNcclEndpointManifestPlan::Create(
      config.engine_epoch, config.communicator_generation,
      config.config_identity, endpoint_identities, *capabilities);
  if (!manifests.ok()) return manifests.status();
  std::vector<DeepSeekNcclEdgePair> edges;
  edges.reserve(config.world_size - 1);
  for (std::uint32_t edge = 0; edge + 1 < config.world_size; ++edge) {
    auto lower = DeepSeekNcclEdgeEndpoint::Create(
        *endpoint_apis[edge], std::move((*capabilities)[edge].lower),
        release_configs[edge], manifests->edges()[edge].lower);
    if (!lower.ok()) return lower.status();
    auto upper = DeepSeekNcclEdgeEndpoint::Create(
        *endpoint_apis[edge + 1], std::move((*capabilities)[edge].upper),
        release_configs[edge + 1], manifests->edges()[edge].upper);
    if (!upper.ok()) return upper.status();
    auto pair = DeepSeekNcclEdgePair::Create(
        std::make_unique<DeepSeekNcclEdgeEndpoint>(std::move(*lower)),
        std::make_unique<DeepSeekNcclEdgeEndpoint>(std::move(*upper)));
    if (!pair.ok()) return pair.status();
    edges.push_back(std::move(*pair));
  }
  auto generation = DeepSeekNcclCommunicatorGeneration::Create(
      {config.engine_epoch, config.communicator_generation, config.world_size,
       config.edge_timeout_ns, config.set_timeout_ns},
      std::move(edges));
  if (!generation.ok()) return generation.status();
  auto now = clocks.front()->now_ns();
  if (!now.ok()) return now.status();
  auto initialized = generation->begin_init(*now);
  while (!initialized.ok() &&
         initialized.code() == StatusCode::kUnavailable) {
    now = clocks.front()->now_ns();
    if (!now.ok()) {
      (void)generation->abort();
      return now.status();
    }
    initialized = generation->advance_init(*now);
  }
  if (!initialized.ok()) return initialized;
  auto receipts = warmup.run(*generation, *manifests,
                             config.maximum_wire_tokens);
  if (!receipts.ok()) {
    (void)generation->abort();
    return receipts.status();
  }
  auto sealed = generation->seal(*receipts);
  if (!sealed.ok()) return sealed;
  return DeepSeekNcclGenerationBoundaryAssembly::CreateWithDependencies(
      std::move(*generation), std::move(rank_identities),
      std::move(device_resources), std::move(clocks),
      std::move(driver_factories), std::move(dependencies));
}

}  // namespace pih
