#include "pih/model/deepseek_nccl_generation_boundary_assembly.h"

namespace pih {
namespace {

class EmptyGenerationDependencyOwner final
    : public DeepSeekNcclGenerationDependencyOwner {};

}  // namespace

Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
DeepSeekNcclGenerationBoundaryAssembly::Create(
    DeepSeekNcclCommunicatorGeneration generation,
    std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
    std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
        device_resources,
    std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
    std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
        driver_factories) {
  return CreateWithDependencies(
      std::move(generation), std::move(rank_identities),
      std::move(device_resources), std::move(clocks),
      std::move(driver_factories),
      std::make_unique<EmptyGenerationDependencyOwner>());
}

Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
DeepSeekNcclGenerationBoundaryAssembly::CreateWithDependencies(
    DeepSeekNcclCommunicatorGeneration generation,
    std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
    std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
        device_resources,
    std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
    std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
        driver_factories,
    std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies) {
  const auto identity = generation.identity();
  const auto world_size = identity.world_size;
  if (generation.state() != DeepSeekNcclGenerationState::kSealed ||
      rank_identities.size() != world_size ||
      device_resources.size() != world_size || clocks.size() != world_size ||
      driver_factories.size() != world_size || dependencies == nullptr) {
    return Status::InvalidArgument(
        "DeepSeek NCCL boundary assembly topology is invalid");
  }
  std::vector<std::unique_ptr<DeepSeekRankBoundaryFactory>> factories;
  factories.reserve(world_size);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    auto& rank_identity = rank_identities[rank];
    if (rank_identity.engine_epoch != identity.engine_epoch ||
        rank_identity.rank != rank ||
        rank_identity.world_size != world_size ||
        rank_identity.communicator_generation !=
            identity.communicator_generation ||
        device_resources[rank] == nullptr || clocks[rank] == nullptr ||
        driver_factories[rank] == nullptr) {
      return Status::FailedPrecondition(
          "DeepSeek NCCL boundary assembly rank identity is inconsistent");
    }
    auto factory = DeepSeekProductionRankBoundaryFactory::Create(
        rank_identity, generation.incoming_transport(rank),
        generation.outgoing_transport(rank),
        std::move(device_resources[rank]), *clocks[rank],
        *driver_factories[rank]);
    if (!factory.ok()) return factory.status();
    factories.push_back(std::move(*factory));
  }
  auto factory_set = DeepSeekRankBoundaryFactorySet::Create(
      world_size, std::move(factories));
  if (!factory_set.ok()) return factory_set.status();
  return std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>(
      new DeepSeekNcclGenerationBoundaryAssembly(
          std::make_unique<DeepSeekNcclCommunicatorGeneration>(
              std::move(generation)),
          std::move(clocks), std::move(driver_factories),
          std::make_unique<DeepSeekRankBoundaryFactorySet>(
              std::move(*factory_set)),
          std::move(dependencies)));
}

DeepSeekNcclGenerationBoundaryAssembly::
    ~DeepSeekNcclGenerationBoundaryAssembly() {
  factories_.reset();
  driver_factories_.clear();
  clocks_.clear();
  // A live sealed generation is suspect unless the owner explicitly performs
  // clean teardown. Fail-stop abort is the only safe destructor fallback.
  if (generation_ != nullptr &&
      generation_->state() != DeepSeekNcclGenerationState::kDestroyed &&
      generation_->state() != DeepSeekNcclGenerationState::kAborted) {
    (void)generation_->abort();
  }
  generation_.reset();
  dependencies_.reset();
}

Status DeepSeekNcclGenerationBoundaryAssembly::advance_clean_teardown() {
  if (generation_->state() == DeepSeekNcclGenerationState::kDestroyed) {
    return Status::Ok();
  }
  if (generation_->state() == DeepSeekNcclGenerationState::kAborted) {
    return Status::FailedPrecondition(
        "DeepSeek NCCL generation was aborted before clean teardown");
  }
  auto now = clocks_.front()->now_ns();
  if (!now.ok()) return now.status();
  if (generation_->state() == DeepSeekNcclGenerationState::kSealed) {
    return generation_->begin_teardown(*now);
  }
  if (generation_->state() == DeepSeekNcclGenerationState::kFinalizing) {
    return generation_->advance_teardown(*now);
  }
  return Status::FailedPrecondition(
      "DeepSeek NCCL generation is not cleanly tear-downable");
}

}  // namespace pih
