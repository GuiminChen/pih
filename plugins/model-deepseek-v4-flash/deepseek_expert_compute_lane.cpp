#include "pih/model/deepseek_expert_compute_lane.h"

#include <new>
#include <utility>

namespace pih {

DeepSeekExpertComputeLane& DeepSeekExpertComputeLane::operator=(
    DeepSeekExpertComputeLane&& other) noexcept {
  if (this != &other) {
    this->~DeepSeekExpertComputeLane();
    ::new (static_cast<void*>(this))
        DeepSeekExpertComputeLane(std::move(other));
  }
  return *this;
}

Result<DeepSeekExpertComputeLane> DeepSeekExpertComputeLane::Create(
    DeepSeekExpertCudaOperations& operations,
    RegisteredPinnedAllocator& pinned_allocator,
    DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
    std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
    std::uintptr_t accumulator_f32, std::uintptr_t stream,
    std::uintptr_t completion_event, std::uint64_t context_identity
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
    , std::uintptr_t dspark_source_hidden_bf16
#endif
    ) {
  auto staging = DeepSeekExpertHostStagingOwner::Allocate(
      packed_token_count, pinned_allocator);
  if (!staging.ok()) return staging.status();
  auto backend_value = DeepSeekExpertCudaBackend::Create(
      operations, staging->staging(), completion_event, context_identity);
  if (!backend_value.ok()) return backend_value.status();
  auto backend = std::make_unique<DeepSeekExpertCudaBackend>(
      std::move(*backend_value));
  auto driver_value = DeepSeekExpertComputeDriver::Create(
      std::move(slots), arena, packed_token_count, source_hidden_bf16,
      accumulator_f32, stream, *backend);
  if (!driver_value.ok()) return driver_value.status();
  auto driver = std::make_unique<DeepSeekExpertComputeDriver>(
      std::move(*driver_value));
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkExpertComputeDriver> dspark_driver;
  if (dspark_source_hidden_bf16 != 0) {
    auto dspark_value = DeepSeekDsparkExpertComputeDriver::Create(
        arena, dspark_source_hidden_bf16, accumulator_f32, stream,
        context_identity, *backend);
    if (!dspark_value.ok()) return dspark_value.status();
    dspark_driver = std::make_unique<DeepSeekDsparkExpertComputeDriver>(
        std::move(*dspark_value));
  }
#endif
  return DeepSeekExpertComputeLane(
      std::move(*staging), std::move(backend), std::move(driver)
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      ,
      std::move(dspark_driver));
#else
      );
#endif
}

}  // namespace pih
