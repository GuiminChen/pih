#include "pih/model/deepseek_stage_compute_stack.h"

#include <utility>

namespace pih {

class DeepSeekStageComputeStack::RejectBackend final
    : public DeepSeekStageOperatorBackend {
 public:
  Status launch(const DeepSeekStageOperatorCommand&,
                const DeepSeekPipelinePlanDescriptor&) override {
    return Status::FailedPrecondition(
        "DeepSeek operator escaped the canonical production backend chain");
  }
  Result<DeepSeekStageComputeStatus> poll() override {
    return Status::FailedPrecondition(
        "DeepSeek reject backend is never pollable");
  }
};

class DeepSeekStageComputeStack::SharedProviderSlot final : public DeepSeekSharedExpertProvider {
 public:
  DeepSeekSharedExpertProvider* provider = nullptr;
  Status prepare(std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) override {
    if (provider == nullptr) return Status::FailedPrecondition("Shared expert provider is unbound");
    return provider->prepare(layer, plan);
  }
  Result<DeepSeekSharedExpertDriver*> resolve(std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override {
    if (provider == nullptr) return Status::FailedPrecondition("Shared expert provider is unbound");
    return provider->resolve(layer, plan);
  }
};

DeepSeekStageComputeStack::DeepSeekStageComputeStack(
    DeepSeekStageComputeStack&&) noexcept = default;
DeepSeekStageComputeStack& DeepSeekStageComputeStack::operator=(
    DeepSeekStageComputeStack&&) noexcept = default;
DeepSeekStageComputeStack::~DeepSeekStageComputeStack() = default;

namespace {

bool complete(const DeepSeekStageComputeStackDependencies& value) {
  const bool paged = value.expert_pager != nullptr &&
                     value.expert_transfer != nullptr &&
                     value.resident_experts == nullptr;
  const bool resident = value.expert_pager == nullptr &&
                        value.expert_transfer == nullptr &&
                        value.resident_experts != nullptr;
  return value.decode_attention != nullptr &&
         value.chunk_attention != nullptr &&
         value.dense_attention != nullptr && value.router != nullptr &&
         value.expert_plan != nullptr && (paged || resident) &&
         value.expert_kernel != nullptr &&
         value.mhc != nullptr;
}

bool dspark_complete(
    const DeepSeekStagePlan& stage,
    const DeepSeekStageComputeStackDependencies& dependencies) noexcept {
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  return !stage.owns_dspark ||
         (dependencies.dspark != nullptr && dependencies.dspark_mtp != nullptr);
#else
  (void)dependencies;
  return !stage.owns_dspark;
#endif
}

template <typename T>
Result<std::unique_ptr<T>> own(Result<T> result) {
  if (!result.ok()) return result.status();
  return std::make_unique<T>(std::move(*result));
}

}  // namespace

Result<DeepSeekStageComputeStack> DeepSeekStageComputeStack::Create(
    DeepSeekStagePlan stage,
    DeepSeekStageComputeStackDependencies dependencies) {
  const bool endpoint_required = stage.owns_embedding || stage.owns_lm_head;
  if (!complete(dependencies) ||
      (endpoint_required && dependencies.endpoint == nullptr) ||
      !dspark_complete(stage, dependencies) ||
      stage.layers.first_layer > stage.layers.last_layer ||
      stage.layers.last_layer >= 43) {
    return Status::InvalidArgument(
        "DeepSeek production compute stack dependencies are incomplete");
  }
  if (dependencies.router->plan_provider() != dependencies.expert_plan) {
    return Status::InvalidArgument(
        "DeepSeek router publisher and routed expert provider differ");
  }
  DeepSeekStageComputeStack stack;
  stack.reject_ = std::make_unique<RejectBackend>();

  auto routed = dependencies.resident_experts == nullptr
      ? own(DeepSeekRoutedStageOperatorBackend::Create(
            *stack.reject_, *dependencies.expert_plan,
            *dependencies.expert_pager, *dependencies.expert_transfer,
            *dependencies.expert_kernel))
      : own(DeepSeekRoutedStageOperatorBackend::CreateResident(
            *stack.reject_, *dependencies.expert_plan,
            *dependencies.expert_kernel, *dependencies.resident_experts));
  if (!routed.ok()) return routed.status();
  stack.routed_ = std::move(*routed);
  stack.shared_provider_ = std::make_unique<SharedProviderSlot>();
  stack.shared_provider_->provider = dependencies.shared_experts;
  stack.shared_ = std::make_unique<DeepSeekSharedExpertStageBackend>(
      *stack.routed_, *stack.shared_provider_);

  auto router = own(DeepSeekRouterStageOperatorBackend::Create(
      *stack.shared_, *dependencies.router));
  if (!router.ok()) return router.status();
  stack.router_ = std::move(*router);

  auto attention = own(DeepSeekAttentionStageOperatorBackend::Create(
      *stack.router_, *dependencies.decode_attention,
      *dependencies.chunk_attention));
  if (!attention.ok()) return attention.status();
  stack.attention_ = std::move(*attention);

  auto dense_attention = own(DeepSeekDenseAttentionStageOperatorBackend::Create(
      *stack.attention_, *dependencies.dense_attention));
  if (!dense_attention.ok()) return dense_attention.status();
  stack.dense_attention_ = std::move(*dense_attention);

  auto mhc = own(DeepSeekMhcStageOperatorBackend::Create(
      *stack.dense_attention_, *dependencies.mhc));
  if (!mhc.ok()) return mhc.status();
  stack.mhc_ = std::move(*mhc);

  DeepSeekStageOperatorBackend* top = stack.mhc_.get();

  if (endpoint_required) {
    auto endpoint = own(DeepSeekEndpointStageOperatorBackend::Create(
        *top, *dependencies.endpoint));
    if (!endpoint.ok()) return endpoint.status();
    stack.endpoint_ = std::move(*endpoint);
    top = stack.endpoint_.get();
  }

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  if (stage.owns_dspark) {
    auto block = own(DeepSeekDsparkMtpBlockExecutor::Create(
        *dependencies.dspark_mtp));
    if (!block.ok()) return block.status();
    stack.dspark_block_ = std::move(*block);
    auto dspark = own(DeepSeekDsparkStageOperatorBackend::Create(
        *top, *dependencies.dspark, *stack.dspark_block_));
    if (!dspark.ok()) return dspark.status();
    stack.dspark_ = std::move(*dspark);
    top = stack.dspark_.get();
  }
#else
  if (stage.owns_dspark) {
    return Status::FailedPrecondition(
        "DeepSeek DSpark is not present in this model bundle");
  }
#endif
  stack.driver_ = std::make_unique<DeepSeekModelStageComputeDriver>(*top);
  return stack;
}

Status DeepSeekStageComputeStack::bind_shared_experts(DeepSeekSharedExpertProvider& provider) {
  if (started_ || shared_provider_ == nullptr || shared_provider_->provider != nullptr) {
    return Status::FailedPrecondition("Shared expert provider cannot be rebound");
  }
  shared_provider_->provider = &provider;
  return Status::Ok();
}

Status DeepSeekStageComputeStack::launch(
    const DeepSeekPipelinePlanDescriptor& plan,
    const DeepSeekStagePlan& stage) {
  if (driver_ == nullptr || shared_provider_ == nullptr ||
      (plan.phase != DeepSeekPlanPhase::kDrain && shared_provider_->provider == nullptr)) {
    return Status::FailedPrecondition(
        "DeepSeek production compute stack or shared expert provider is not initialized");
  }
  started_ = true;
  return driver_->launch(plan, stage);
}

Result<DeepSeekStageComputeStatus> DeepSeekStageComputeStack::poll() {
  if (driver_ == nullptr) {
    return Status::FailedPrecondition(
        "DeepSeek production compute stack is not initialized");
  }
  return driver_->poll();
}

}  // namespace pih
