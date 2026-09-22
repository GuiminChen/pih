#pragma once

#include <memory>
#include <vector>

#include "pih/model/deepseek_nccl_communicator_generation.h"
#include "pih/model/deepseek_production_rank_boundary_factory.h"
#include "pih/model/deepseek_rank_boundary_factory_set.h"

namespace pih {

class DeepSeekNcclGenerationDependencyOwner {
 public:
  virtual ~DeepSeekNcclGenerationDependencyOwner() = default;
};

class DeepSeekNcclGenerationBoundaryAssembly final {
 public:
  static Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
  Create(
      DeepSeekNcclCommunicatorGeneration generation,
      std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
          device_resources,
      std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
      std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
          driver_factories);
  static Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>>
  CreateWithDependencies(
      DeepSeekNcclCommunicatorGeneration generation,
      std::vector<DeepSeekRankBoundaryFactoryIdentity> rank_identities,
      std::vector<std::unique_ptr<DeepSeekRankBoundaryDeviceResources>>
          device_resources,
      std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
      std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
          driver_factories,
      std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies);

  DeepSeekNcclGenerationBoundaryAssembly(
      const DeepSeekNcclGenerationBoundaryAssembly&) = delete;
  DeepSeekNcclGenerationBoundaryAssembly& operator=(
      const DeepSeekNcclGenerationBoundaryAssembly&) = delete;
  ~DeepSeekNcclGenerationBoundaryAssembly();

  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return generation_->identity().world_size;
  }
  [[nodiscard]] DeepSeekRankBoundaryFactory* factory(
      std::uint32_t rank) noexcept {
    return factories_->factory(rank);
  }
  [[nodiscard]] DeepSeekNcclGenerationState generation_state() const noexcept {
    return generation_->state();
  }
  Status begin_teardown(std::uint64_t now_ns) {
    return generation_->begin_teardown(now_ns);
  }
  Status advance_teardown(std::uint64_t now_ns) {
    return generation_->advance_teardown(now_ns);
  }
  Status advance_clean_teardown();
  Status abort() noexcept { return generation_->abort(); }

 private:
  DeepSeekNcclGenerationBoundaryAssembly(
      std::unique_ptr<DeepSeekNcclCommunicatorGeneration> generation,
      std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks,
      std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
          driver_factories,
      std::unique_ptr<DeepSeekRankBoundaryFactorySet> factories,
      std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies)
      noexcept
      : dependencies_(std::move(dependencies)),
        generation_(std::move(generation)), clocks_(std::move(clocks)),
        driver_factories_(std::move(driver_factories)),
        factories_(std::move(factories)) {}

  // Destruction is reverse declaration order: factories stop borrowing first,
  // then their clock/evidence dependencies, then communicator transports,
  // and only then the API/context dependency owner.
  std::unique_ptr<DeepSeekNcclGenerationDependencyOwner> dependencies_;
  std::unique_ptr<DeepSeekNcclCommunicatorGeneration> generation_;
  std::vector<std::unique_ptr<DeepSeekBoundaryPreparationClock>> clocks_;
  std::vector<std::unique_ptr<DeepSeekBoundaryDriverOwnerFactory>>
      driver_factories_;
  std::unique_ptr<DeepSeekRankBoundaryFactorySet> factories_;
};

}  // namespace pih
