#pragma once

#include <memory>
#include <optional>

#include "pih/model/deepseek_rank_plan_runtime.h"
#include "pih/model/deepseek_rank_serving_worker_loop.h"

namespace pih {

struct DeepSeekRankServingResolvedLeases final {
  std::uint64_t input_lease_identity = 0;
  std::uint64_t output_lease_identity = 0;
  std::shared_ptr<const void> input_owner;
  std::shared_ptr<const void> output_owner;
};

class DeepSeekRankServingLeaseResolver {
 public:
  virtual ~DeepSeekRankServingLeaseResolver() = default;
  virtual Result<DeepSeekRankServingResolvedLeases> resolve(
      const DeepSeekRankServingCommand& command) = 0;
};

// A production implementation owns one already-admitted rank plan runtime.
// The service layer intentionally cannot construct the runtime: that would
// bypass resource reservations, stage ownership, and boundary preparation.
class DeepSeekRankServingPlanExecution {
 public:
  virtual ~DeepSeekRankServingPlanExecution() = default;

  // Requests an orderly drain. It is not permission to publish an output after
  // cancellation; the protocol adapter converts a terminal drain to a
  // cancellation completion with no output lease.
  virtual Status cancel() = 0;

  // Advances nonblocking work. `false` means work remains; `true` means all
  // rank-local resources have reached their terminal execution state.
  virtual Result<bool> advance() = 0;
};

class DeepSeekRankServingPlanExecutionFactory {
 public:
  virtual ~DeepSeekRankServingPlanExecutionFactory() = default;

  // The factory is the only point where a command may acquire rank-local
  // compute, staging and boundary resources. It must return an execution that
  // is bound to the command's exact session, request and plan.
  virtual Result<std::unique_ptr<DeepSeekRankServingPlanExecution>> start(
      const DeepSeekRankServingCommand& command,
      DeepSeekRankServingResolvedLeases leases) = 0;
};

// Production factories which have already prepared and committed a
// DeepSeekRankPlanRuntime use this adapter instead of reimplementing the
// runtime's drain and resource-release state machine. Creation binds the
// runtime's rank and immutable descriptor to one execute command.
class DeepSeekRankPlanRuntimeServingExecution final
    : public DeepSeekRankServingPlanExecution {
 public:
  static Result<DeepSeekRankPlanRuntimeServingExecution> Create(
      DeepSeekRankPlanRuntime runtime,
      const DeepSeekRankServingCommand& command);

  DeepSeekRankPlanRuntimeServingExecution(
      const DeepSeekRankPlanRuntimeServingExecution&) = delete;
  DeepSeekRankPlanRuntimeServingExecution& operator=(
      const DeepSeekRankPlanRuntimeServingExecution&) = delete;
  DeepSeekRankPlanRuntimeServingExecution(
      DeepSeekRankPlanRuntimeServingExecution&&) noexcept = default;
  DeepSeekRankPlanRuntimeServingExecution& operator=(
      DeepSeekRankPlanRuntimeServingExecution&&) noexcept = default;

  Status cancel() override;
  Result<bool> advance() override;

 private:
  explicit DeepSeekRankPlanRuntimeServingExecution(
      DeepSeekRankPlanRuntime runtime) noexcept
      : runtime_(std::move(runtime)) {}

  DeepSeekRankPlanRuntime runtime_;
};

// Bridges the transport-neutral worker loop to a prepared plan execution.
// It maintains an independent copy of command identity so a future executor
// cannot accidentally complete a different request after a cancel/control
// transition. The surrounding worker gate remains the authoritative protocol
// validator and fail-stop boundary.
class DeepSeekRankServingPlanWorkerExecutor final
    : public DeepSeekRankServingWorkerExecutor {
 public:
  static Result<DeepSeekRankServingPlanWorkerExecutor> Create(
      DeepSeekRankServingPlanExecutionFactory& factory,
      DeepSeekRankServingLeaseResolver& resolver);

  DeepSeekRankServingPlanWorkerExecutor(
      const DeepSeekRankServingPlanWorkerExecutor&) = delete;
  DeepSeekRankServingPlanWorkerExecutor& operator=(
      const DeepSeekRankServingPlanWorkerExecutor&) = delete;
  DeepSeekRankServingPlanWorkerExecutor(
      DeepSeekRankServingPlanWorkerExecutor&&) noexcept = default;
  DeepSeekRankServingPlanWorkerExecutor& operator=(
      DeepSeekRankServingPlanWorkerExecutor&&) noexcept = default;

  Status start(const DeepSeekRankServingCommand& command) override;
  Status cancel(const DeepSeekRankServingCommand& command) override;
  Result<std::optional<DeepSeekRankServingCompletion>> poll() override;

  [[nodiscard]] bool execution_active() const noexcept {
    return active_command_.has_value();
  }

 private:
  explicit DeepSeekRankServingPlanWorkerExecutor(
      DeepSeekRankServingPlanExecutionFactory& factory,
      DeepSeekRankServingLeaseResolver& resolver) noexcept
      : factory_(&factory), resolver_(&resolver) {}

  DeepSeekRankServingPlanExecutionFactory* factory_ = nullptr;
  DeepSeekRankServingLeaseResolver* resolver_ = nullptr;
  std::unique_ptr<DeepSeekRankServingPlanExecution> execution_;
  std::optional<DeepSeekRankServingCommand> active_command_;
  bool cancellation_requested_ = false;
};

}  // namespace pih
