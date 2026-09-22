#pragma once

#include <memory>

#include "pih/backend/cuda/deepseek_expert_host_staging.h"
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
#include "pih/model/deepseek_dspark_expert_compute_driver.h"
#endif

namespace pih {

#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
class DeepSeekDsparkExpertComputeDriver;
class DeepSeekDsparkExpertKernelDriver;
#endif

class DeepSeekExpertComputeLane final {
 public:
  static Result<DeepSeekExpertComputeLane> Create(
      DeepSeekExpertCudaOperations& operations,
      RegisteredPinnedAllocator& pinned_allocator,
      DeepSeekExpertSlotTable slots, DeepSeekExpertComputeArena arena,
      std::uint32_t packed_token_count, std::uintptr_t source_hidden_bf16,
      std::uintptr_t accumulator_f32, std::uintptr_t stream,
      std::uintptr_t completion_event, std::uint64_t context_identity
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      , std::uintptr_t dspark_source_hidden_bf16 = 0
#endif
      );

  DeepSeekExpertComputeLane(const DeepSeekExpertComputeLane&) = delete;
  DeepSeekExpertComputeLane& operator=(const DeepSeekExpertComputeLane&) = delete;
  DeepSeekExpertComputeLane(DeepSeekExpertComputeLane&&) noexcept = default;
  DeepSeekExpertComputeLane& operator=(
      DeepSeekExpertComputeLane&& other) noexcept;

  [[nodiscard]] DeepSeekExpertKernelDriver& kernel() noexcept {
    return *driver_;
  }
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  [[nodiscard]] DeepSeekDsparkExpertKernelDriver* dspark_kernel() noexcept {
    return dspark_driver_.get();
  }
#endif
  [[nodiscard]] const DeepSeekExpertHostStagingOwner& host_staging() const noexcept {
    return staging_;
  }

 private:
  DeepSeekExpertComputeLane(
      DeepSeekExpertHostStagingOwner staging,
      std::unique_ptr<DeepSeekExpertCudaBackend> backend,
      std::unique_ptr<DeepSeekExpertComputeDriver> driver
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
      ,
      std::unique_ptr<DeepSeekDsparkExpertComputeDriver> dspark_driver)
#else
      )
#endif
      : staging_(std::move(staging)), backend_(std::move(backend)),
        driver_(std::move(driver))
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
        ,
        dspark_driver_(std::move(dspark_driver)) {}
#else
        {}
#endif

  DeepSeekExpertHostStagingOwner staging_;
  std::unique_ptr<DeepSeekExpertCudaBackend> backend_;
  std::unique_ptr<DeepSeekExpertComputeDriver> driver_;
#if !defined(PIH_DEEPSEEK_DSPARK_DISABLED)
  std::unique_ptr<DeepSeekDsparkExpertComputeDriver> dspark_driver_;
#endif
};

}  // namespace pih
