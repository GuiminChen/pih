#pragma once

#include <array>
#include <memory>

#include "pih/model/deepseek_boundary_operation_builder.h"
#include "pih/model/deepseek_rank_plan_runtime.h"

namespace pih {

class DeepSeekBoundaryPreparationClock : public DeepSeekBoundaryRuntimeClock {
 public:
  virtual ~DeepSeekBoundaryPreparationClock() = default;
  virtual Result<std::uint64_t> now_ns() override = 0;
};

class DeepSeekSteadyBoundaryClock final
    : public DeepSeekBoundaryPreparationClock {
 public:
  Result<std::uint64_t> now_ns() override;
};

class DeepSeekBoundaryDriverOwnerFactory {
 public:
  virtual ~DeepSeekBoundaryDriverOwnerFactory() = default;
  virtual Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>> create(
      DeepSeekBoundaryTransportDriver& transport) = 0;
};

struct DeepSeekRankBoundaryFactoryIdentity final {
  std::uint64_t engine_epoch = 0;
  std::uint32_t rank = 0;
  std::uint32_t world_size = 0;
  std::uintptr_t context_identity = 0;
  DriverStreamHandle boundary_stream = 0;
  std::uint64_t communicator_generation = 0;
  std::uint64_t timeout_ns = 0;
  std::array<std::uint64_t,
             DeepSeekRankBoundaryDeviceResources::kCreditCount>
      incoming_buffer_owner_ids{};
};

class DeepSeekProductionRankBoundaryFactory final
    : public DeepSeekRankBoundaryFactory {
 public:
  static Result<std::unique_ptr<DeepSeekProductionRankBoundaryFactory>> Create(
      DeepSeekRankBoundaryFactoryIdentity identity,
      DeepSeekBoundaryTransportDriver* incoming_transport,
      DeepSeekBoundaryTransportDriver* outgoing_transport,
      std::unique_ptr<DeepSeekRankBoundaryDeviceResources> resources,
      DeepSeekBoundaryPreparationClock& clock,
      DeepSeekBoundaryDriverOwnerFactory& driver_factory);

  Result<DeepSeekPreparedRankBoundaries> prepare(
      DeepSeekPipelineTransaction& transaction,
      const DeepSeekStagePlan& stage, std::uint32_t world_size,
      std::uint32_t wire_token_count,
      const std::optional<DeepSeekBoundarySendSource>& outgoing_source)
      override;

 private:
  DeepSeekProductionRankBoundaryFactory(
      DeepSeekRankBoundaryFactoryIdentity identity,
      DeepSeekBoundaryTransportDriver* incoming_transport,
      DeepSeekBoundaryTransportDriver* outgoing_transport,
      std::unique_ptr<DeepSeekRankBoundaryDeviceResources> resources,
      DeepSeekBoundaryPreparationClock& clock,
      DeepSeekBoundaryDriverOwnerFactory& driver_factory,
      DeepSeekNcclOperationSequencer sequencer,
      DeepSeekBoundaryCreditTracker incoming_tracker,
      DeepSeekBoundaryCreditTracker outgoing_tracker) noexcept
      : identity_(identity), incoming_transport_(incoming_transport),
        outgoing_transport_(outgoing_transport),
        resources_(std::move(resources)), clock_(&clock),
        driver_factory_(&driver_factory), sequencer_(std::move(sequencer)),
        incoming_tracker_(std::move(incoming_tracker)),
        outgoing_tracker_(std::move(outgoing_tracker)) {}

  DeepSeekRankBoundaryFactoryIdentity identity_;
  DeepSeekBoundaryTransportDriver* incoming_transport_ = nullptr;
  DeepSeekBoundaryTransportDriver* outgoing_transport_ = nullptr;
  std::unique_ptr<DeepSeekRankBoundaryDeviceResources> resources_;
  DeepSeekBoundaryPreparationClock* clock_ = nullptr;
  DeepSeekBoundaryDriverOwnerFactory* driver_factory_ = nullptr;
  DeepSeekNcclOperationSequencer sequencer_;
  DeepSeekBoundaryCreditTracker incoming_tracker_;
  DeepSeekBoundaryCreditTracker outgoing_tracker_;
};

}  // namespace pih
