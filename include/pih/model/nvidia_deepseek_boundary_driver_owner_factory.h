#pragma once

#include <atomic>
#include <memory>

#include "pih/model/deepseek_boundary_health_state.h"
#include "pih/model/deepseek_production_rank_boundary_factory.h"

namespace pih {

class NvidiaDeepSeekBoundaryDriverOwnerFactory final
    : public DeepSeekBoundaryDriverOwnerFactory {
 public:
  static Result<NvidiaDeepSeekBoundaryDriverOwnerFactory> Create(
      std::uintptr_t context_identity,
      const std::atomic<std::uint32_t>& device_error_code,
      const std::atomic<bool>& engine_poisoned);
  static Result<NvidiaDeepSeekBoundaryDriverOwnerFactory> CreateOwned(
      std::uintptr_t context_identity,
      std::shared_ptr<DeepSeekBoundaryHealthState> health);

  Result<std::unique_ptr<DeepSeekBoundaryDriverOwner>> create(
      DeepSeekBoundaryTransportDriver& transport) override;

 private:
  NvidiaDeepSeekBoundaryDriverOwnerFactory(
      std::uintptr_t context_identity,
      const std::atomic<std::uint32_t>& device_error_code,
      const std::atomic<bool>& engine_poisoned,
      std::shared_ptr<DeepSeekBoundaryHealthState> health = nullptr) noexcept
      : context_identity_(context_identity),
        health_(std::move(health)),
        device_error_code_(&device_error_code),
        engine_poisoned_(&engine_poisoned) {}

  std::uintptr_t context_identity_ = 0;
  std::shared_ptr<DeepSeekBoundaryHealthState> health_;
  const std::atomic<std::uint32_t>* device_error_code_ = nullptr;
  const std::atomic<bool>* engine_poisoned_ = nullptr;
};

}  // namespace pih
