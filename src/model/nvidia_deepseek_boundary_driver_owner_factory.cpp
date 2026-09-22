#include "pih/model/nvidia_deepseek_boundary_driver_owner_factory.h"

#include "pih/backend/cuda/nvidia_completion_event_driver.h"

namespace pih {

Result<NvidiaDeepSeekBoundaryDriverOwnerFactory>
NvidiaDeepSeekBoundaryDriverOwnerFactory::Create(
    std::uintptr_t context_identity,
    const std::atomic<std::uint32_t>& device_error_code,
    const std::atomic<bool>& engine_poisoned) {
  if (context_identity == 0) {
    return Status::InvalidArgument(
        "NVIDIA DeepSeek boundary driver context is null");
  }
  return NvidiaDeepSeekBoundaryDriverOwnerFactory(
      context_identity, device_error_code, engine_poisoned);
}

Result<NvidiaDeepSeekBoundaryDriverOwnerFactory>
NvidiaDeepSeekBoundaryDriverOwnerFactory::CreateOwned(
    std::uintptr_t context_identity,
    std::shared_ptr<DeepSeekBoundaryHealthState> health) {
  if (context_identity == 0 || health == nullptr) {
    return Status::InvalidArgument(
        "NVIDIA DeepSeek owned boundary health is invalid");
  }
  return NvidiaDeepSeekBoundaryDriverOwnerFactory(
      context_identity, health->device_error_atomic(),
      health->engine_poisoned_atomic(), std::move(health));
}

Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>>
NvidiaDeepSeekBoundaryDriverOwnerFactory::create(
    DeepSeekBoundaryTransportDriver& transport) {
  auto event = NvidiaCompletionEventDriver::Create();
  if (!event.ok()) return event.status();
  if (event->context_identity() != context_identity_) {
    return Status::FailedPrecondition(
        "NVIDIA DeepSeek boundary driver current context changed");
  }
  auto event_owner =
      std::make_unique<NvidiaCompletionEventDriver>(std::move(*event));
  auto evidence = AtomicCompletionEvidenceProvider::Create(
      *event_owner, *device_error_code_, *engine_poisoned_);
  if (!evidence.ok()) return evidence.status();
  auto owner = DeepSeekBorrowedBoundaryDriverOwner::Create(
      transport, std::move(event_owner),
      std::make_unique<AtomicCompletionEvidenceProvider>(
          std::move(*evidence)));
  if (!owner.ok()) return owner.status();
  return std::unique_ptr<DeepSeekBoundaryDriverOwner>(
      std::make_unique<DeepSeekBorrowedBoundaryDriverOwner>(
          std::move(*owner)));
}

}  // namespace pih
