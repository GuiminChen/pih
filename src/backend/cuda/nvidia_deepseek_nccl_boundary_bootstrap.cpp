#include "pih/backend/cuda/nvidia_deepseek_nccl_boundary_bootstrap.h"

#include "pih/backend/cuda/atomic_completion_evidence_provider.h"
#include "pih/backend/cuda/nvidia_completion_event_driver.h"
#include "pih/backend/cuda/nvidia_cuda_context_activator.h"
#include "pih/backend/cuda/nvidia_deepseek_nccl_warmup_payload_operations.h"
#include "pih/core/checked_math.h"
#include "pih/model/deepseek_boundary_health_state.h"
#include "pih/model/deepseek_context_bound_boundary_drivers.h"
#include "pih/model/deepseek_context_bound_nccl_api.h"
#include "pih/model/deepseek_nccl_generation_warmup_runner.h"
#include "pih/model/nvidia_deepseek_boundary_driver_owner_factory.h"

namespace pih {
namespace {

class NvidiaNcclGenerationDependencies final
    : public DeepSeekNcclGenerationDependencyOwner {
 public:
  NvidiaNcclGenerationDependencies(
      std::unique_ptr<DeepSeekNcclCApi> api,
      std::unique_ptr<DeepSeekNcclContextActivator> activator,
      std::vector<std::unique_ptr<DeepSeekNcclCApi>> endpoints) noexcept
      : api_(std::move(api)), activator_(std::move(activator)),
        endpoints_(std::move(endpoints)) {}

 private:
  // Endpoint wrappers are destroyed first, before what they borrow.
  std::unique_ptr<DeepSeekNcclCApi> api_;
  std::unique_ptr<DeepSeekNcclContextActivator> activator_;
  std::vector<std::unique_ptr<DeepSeekNcclCApi>> endpoints_;
};

std::uint64_t incoming_owner_id(std::uint64_t epoch, std::uint32_t rank,
                                std::uint32_t credit) {
  return (epoch << 16U) ^ (std::uint64_t{rank + 1} << 8U) ^ (credit + 1U);
}

}  // namespace

Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
NvidiaDeepSeekNcclBoundaryBootstrap::Build(
    DeepSeekNcclBoundaryBootstrapConfig config,
    std::span<const std::uint64_t> device_identities,
    std::span<NvidiaDeepSeekRankRuntimeView* const> runtimes,
    std::span<Allocator* const> device_allocators,
    std::span<const DeepSeekNcclReleaseConfig> release_configs,
    RegisteredPinnedAllocator& pinned_allocator,
    std::unique_ptr<DeepSeekNcclCApi> api_owner,
    DeepSeekNcclUniqueIdSource& unique_id_source) {
  const auto world_size = config.world_size;
  if (world_size < 2 || world_size > 4 ||
      device_identities.size() != world_size || runtimes.size() != world_size ||
      device_allocators.size() != world_size ||
      release_configs.size() != world_size || api_owner == nullptr) {
    return Status::InvalidArgument(
        "NVIDIA DeepSeek NCCL boundary topology is invalid");
  }
  auto activator = std::make_unique<NvidiaCudaContextActivator>();
  std::vector<std::unique_ptr<DeepSeekNcclCApi>> endpoint_api_owners;
  std::vector<DeepSeekNcclCApi*> endpoint_apis;
  std::vector<DeepSeekNcclRankEndpointIdentity> endpoint_identities;
  std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
      boundary_resources;
  std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks;
  std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>> factories;
  std::vector<DeepSeekRankBoundaryFactoryIdentity> factory_identities;
  std::vector<std::shared_ptr<DeepSeekBoundaryHealthState>> health;
  endpoint_api_owners.reserve(world_size);
  endpoint_apis.reserve(world_size);
  endpoint_identities.reserve(world_size);
  boundary_resources.reserve(world_size);
  clocks.reserve(world_size);
  factories.reserve(world_size);
  factory_identities.reserve(world_size);
  health.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (device_identities[rank] == 0 || runtimes[rank] == nullptr ||
        device_allocators[rank] == nullptr ||
        runtimes[rank]->identity().rank != rank ||
        runtimes[rank]->identity().context == 0 ||
        runtimes[rank]->identity().stream == 0) {
      return Status::FailedPrecondition(
          "NVIDIA DeepSeek NCCL rank runtime identity is invalid");
    }
    auto endpoint_api = DeepSeekContextBoundNcclCApi::Create(
        runtimes[rank]->identity().context, *api_owner, *activator);
    if (!endpoint_api.ok()) return endpoint_api.status();
    endpoint_api_owners.push_back(
        std::make_unique<DeepSeekContextBoundNcclCApi>(
            std::move(*endpoint_api)));
    endpoint_apis.push_back(endpoint_api_owners.back().get());
    endpoint_identities.push_back(
        {rank, device_identities[rank], runtimes[rank]->identity().context});
    auto resources = DeepSeekRankBoundaryDeviceResources::Allocate(
        rank, world_size, config.maximum_wire_tokens,
        runtimes[rank]->identity().context, *device_allocators[rank],
        runtimes[rank]->resource_driver());
    if (!resources.ok()) return resources.status();
    boundary_resources.push_back(std::move(*resources));
    clocks.push_back(std::make_unique<DeepSeekSteadyBoundaryClock>());
    auto rank_health = std::make_shared<DeepSeekBoundaryHealthState>();
    auto factory = NvidiaDeepSeekBoundaryDriverOwnerFactory::CreateOwned(
        runtimes[rank]->identity().context, rank_health);
    if (!factory.ok()) return factory.status();
    factories.push_back(
        std::make_unique<NvidiaDeepSeekBoundaryDriverOwnerFactory>(
            std::move(*factory)));
    health.push_back(std::move(rank_health));
    const std::array<std::uint64_t, 2> owner_ids = rank == 0
        ? std::array<std::uint64_t, 2>{0, 0}
        : std::array<std::uint64_t, 2>{
              incoming_owner_id(config.engine_epoch, rank, 0),
              incoming_owner_id(config.engine_epoch, rank, 1)};
    factory_identities.push_back(
        {config.engine_epoch, rank, world_size,
         runtimes[rank]->identity().context,
         runtimes[rank]->identity().stream,
         config.communicator_generation, config.edge_timeout_ns, owner_ids});
  }

  NvidiaDeepSeekNcclWarmupPayloadOperations raw_payload;
  std::vector<std::unique_ptr<NvidiaCompletionEventDriver>> raw_events;
  std::vector<std::unique_ptr<DeepSeekContextBoundCompletionDriver>>
      bound_events;
  std::vector<std::unique_ptr<AtomicCompletionEvidenceProvider>> evidence;
  std::vector<std::unique_ptr<DeepSeekContextBoundWarmupPayloadOperations>>
      bound_payloads;
  std::vector<DeepSeekNcclGenerationWarmupEndpointResources>
      warmup_resources;
  const auto endpoint_count = static_cast<std::size_t>(world_size - 1) * 2;
  raw_events.reserve(endpoint_count);
  bound_events.reserve(endpoint_count);
  evidence.reserve(endpoint_count);
  bound_payloads.reserve(endpoint_count);
  warmup_resources.reserve(endpoint_count);
  for (std::uint32_t edge = 0; edge + 1 < world_size; ++edge) {
    for (std::uint32_t local = 0; local < 2; ++local) {
      const auto rank = edge + local;
      auto activated = activator->activate(runtimes[rank]->identity().context);
      if (!activated.ok()) return activated;
      auto raw_event = NvidiaCompletionEventDriver::Create();
      if (!raw_event.ok()) return raw_event.status();
      raw_events.push_back(
          std::make_unique<NvidiaCompletionEventDriver>(
              std::move(*raw_event)));
      auto bound_event = DeepSeekContextBoundCompletionDriver::Create(
          runtimes[rank]->identity().context, *raw_events.back(),
          *raw_events.back(), *activator);
      if (!bound_event.ok()) return bound_event.status();
      bound_events.push_back(
          std::make_unique<DeepSeekContextBoundCompletionDriver>(
              std::move(*bound_event)));
      auto endpoint_evidence = AtomicCompletionEvidenceProvider::Create(
          *bound_events.back(), health[rank]->device_error_atomic(),
          health[rank]->engine_poisoned_atomic());
      if (!endpoint_evidence.ok()) return endpoint_evidence.status();
      evidence.push_back(
          std::make_unique<AtomicCompletionEvidenceProvider>(
              std::move(*endpoint_evidence)));
      auto payload = DeepSeekContextBoundWarmupPayloadOperations::Create(
          runtimes[rank]->identity().context, raw_payload, *activator);
      if (!payload.ok()) return payload.status();
      bound_payloads.push_back(
          std::make_unique<DeepSeekContextBoundWarmupPayloadOperations>(
              std::move(*payload)));
      auto* buffer = local == 0
          ? boundary_resources[rank]->outgoing_warmup_buffer()
          : boundary_resources[rank]->incoming_slot(0);
      const auto event0 = local == 0
          ? boundary_resources[rank]->outgoing_event(0)
          : boundary_resources[rank]->incoming_event(0);
      const auto event1 = local == 0
          ? boundary_resources[rank]->outgoing_event(1)
          : boundary_resources[rank]->incoming_event(1);
      if (buffer == nullptr || event0 == 0 || event1 == 0) {
        return Status::Internal(
            "NVIDIA DeepSeek NCCL warm-up resources are incomplete");
      }
      auto now = clocks[rank]->now_ns();
      if (!now.ok()) return now.status();
      auto deadline = checked_add_u64(*now, config.edge_timeout_ns);
      if (!deadline.ok()) return deadline.status();
      const auto endpoint_index = static_cast<std::uint64_t>(edge) * 2 + local;
      warmup_resources.push_back(
          {1 + endpoint_index * 2,
           incoming_owner_id(config.engine_epoch, rank, local),
           buffer->generation(), buffer->data(), buffer->size_bytes(),
           runtimes[rank]->identity().stream, {event0, event1}, *now,
           *deadline, bound_payloads.back().get(), bound_events.back().get(),
           evidence.back().get(), clocks[rank].get()});
    }
  }
  auto warmup = DeepSeekNcclGenerationWarmupRunner::Create(
      std::move(warmup_resources));
  if (!warmup.ok()) return warmup.status();
  auto dependencies = std::make_unique<NvidiaNcclGenerationDependencies>(
      std::move(api_owner), std::move(activator),
      std::move(endpoint_api_owners));
  return DeepSeekNcclBoundaryBootstrap::BuildWithDependencies(
      config, endpoint_identities, endpoint_apis, release_configs,
      unique_id_source, pinned_allocator, *warmup,
      std::move(factory_identities), std::move(boundary_resources),
      std::move(clocks), std::move(factories), std::move(dependencies));
}

}  // namespace pih
