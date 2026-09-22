#pragma once

#include <array>
#include <span>

#include "pih/backend/cuda/deepseek_router_bf16_gemm.h"
#include "pih/model/deepseek_learned_router.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

struct DeepSeekLearnedRouterSubmission final {
  std::uint32_t layer = 0;
  std::uint32_t token_count = 0;
  std::uintptr_t input_bf16 = 0;
  std::uintptr_t weight_bf16 = 0;
  std::uintptr_t scores_f32 = 0;
  std::uintptr_t error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uintptr_t completion_event = 0;
  std::span<float> host_scores;
  std::uint32_t* host_error_flag = nullptr;
};

class DeepSeekLearnedRouterOperations {
 public:
  virtual ~DeepSeekLearnedRouterOperations() = default;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status gemm(DeepSeekRouterBf16GemmLaunch launch) = 0;
  virtual Status copy_d2h_async(void* host, std::uintptr_t device,
                                std::uint64_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event,
                              std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(
      std::uintptr_t event) = 0;
};

class DeepSeekLearnedRouterCoordinator final {
 public:
  static constexpr std::uint32_t kHiddenSize = 4096;
  static constexpr std::uint32_t kExpertCount = 256;
  static Result<DeepSeekLearnedRouterCoordinator> Create(
      std::uint32_t maximum_tokens,
      std::array<float, kExpertCount> bias,
      DeepSeekRouteScratchArena& scratch,
      DeepSeekExpertPlanStore& store,
      DeepSeekLearnedRouterOperations& operations);
  Status launch(const DeepSeekLearnedRouterSubmission& submission);
  Result<DeepSeekStageComputeStatus> poll();
  [[nodiscard]] DeepSeekExpertPlanProvider* plan_provider() const noexcept {
    return store_;
  }

 private:
  Status poison(Status status);
  std::uint32_t maximum_tokens_ = 0;
  std::array<float, kExpertCount> bias_{};
  DeepSeekRouteScratchArena* scratch_ = nullptr;
  DeepSeekExpertPlanStore* store_ = nullptr;
  DeepSeekLearnedRouterOperations* operations_ = nullptr;
  DeepSeekLearnedRouterSubmission active_;
  bool inflight_ = false;
  bool poisoned_ = false;
};

}  // namespace pih
