#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string_view>
#include <vector>

#include "pih/backend/cuda/deepseek_expert_accumulate.h"
#include "pih/backend/cuda/deepseek_expert_swiglu.h"
#include "pih/backend/cuda/deepseek_fp8_activation_quant.h"
#include "pih/backend/cuda/deepseek_fp8_gemm.h"
#include "pih/core/result.h"
#include "pih/core/buffer.h"
#include "pih/core/tensor_view.h"
#include "pih/model/deepseek_expert_compute_arena.h"
#include "pih/model/deepseek_expert_subwave_executor.h"
#include "pih/model/deepseek_stage_compute_driver.h"

namespace pih {

class DeepSeekResidentWeightArena;

struct DeepSeekSharedExpertWeights final {
  using Resolver = std::function<Result<TensorView>(std::string_view)>;
  static Result<DeepSeekSharedExpertWeights> Resolve(
      std::uint32_t layer, const Resolver& resolver);
  static Result<DeepSeekSharedExpertWeights> Resolve(
      std::uint32_t layer, const DeepSeekResidentWeightArena& arena);
  std::uintptr_t w1_e4m3 = 0;
  std::uintptr_t w1_scale_bits = 0;
  std::uintptr_t w3_e4m3 = 0;
  std::uintptr_t w3_scale_bits = 0;
  std::uintptr_t w2_e4m3 = 0;
  std::uintptr_t w2_scale_bits = 0;
  std::uint64_t generation = 0;
  std::int32_t device_ordinal = -1;
};

struct DeepSeekSharedExpertArena final {
  DeepSeekExpertArenaSpan activation_e4m3;
  DeepSeekExpertArenaSpan activation_scale_bits;
  DeepSeekExpertArenaSpan gate_or_middle_bf16;
  DeepSeekExpertArenaSpan up_bf16;
  DeepSeekExpertArenaSpan shared_output_bf16;
  DeepSeekExpertArenaSpan error_flag_u32;
};

class DeepSeekSharedExpertOperations {
 public:
  virtual ~DeepSeekSharedExpertOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host) = 0;
  virtual Status prepare_moe(std::uintptr_t layer_input_bf16,
      std::uintptr_t routed_input_bf16, std::uintptr_t accumulator_f32,
      std::uint32_t token_count, std::uintptr_t stream) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status quantize(DeepSeekFp8ActivationQuantLaunch launch) = 0;
  virtual Status gemm(DeepSeekFp8GemmLaunch launch) = 0;
  virtual Status swiglu(DeepSeekSharedExpertSwiGluLaunch launch) = 0;
  virtual Status finalize(DeepSeekExpertFinalizeLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
      std::uintptr_t device, std::uintptr_t stream) = 0;
  virtual Status record_event(std::uintptr_t event, std::uintptr_t stream) = 0;
  virtual Result<DeepSeekExpertAsyncStatus> query_event(std::uintptr_t event) = 0;
};

class DeepSeekSharedExpertDriver final {
 public:
  static constexpr std::uint32_t kHiddenSize = 4096;
  static constexpr std::uint32_t kIntermediateSize = 2048;
  static constexpr std::uint32_t kMaximumTokens = 4096;

  static Result<DeepSeekSharedExpertDriver> Create(
      std::uint32_t maximum_tokens, DeepSeekSharedExpertWeights weights,
      DeepSeekSharedExpertArena arena, std::uintptr_t input_bf16,
      std::uintptr_t accumulator_f32, std::uintptr_t output_bf16,
      std::uintptr_t stream, std::uintptr_t completion_event,
      std::uint32_t* host_error, DeepSeekSharedExpertOperations& operations);

  // Borrows all resources through completion (including after submission failure,
  // when the owner must drain the stream before releasing them). No host wait.
  Status execute(std::uint32_t token_count);
  Result<DeepSeekExpertAsyncStatus> poll();
  [[nodiscard]] bool can_execute(std::uint32_t token_count) const noexcept {
    return !poisoned_ && !inflight_ && token_count != 0 && token_count <= maximum_tokens_;
  }

 private:
  std::uint32_t maximum_tokens_ = 0;
  DeepSeekSharedExpertWeights weights_;
  DeepSeekSharedExpertArena arena_;
  std::uintptr_t input_bf16_ = 0;
  std::uintptr_t accumulator_f32_ = 0;
  std::uintptr_t output_bf16_ = 0;
  std::uintptr_t stream_ = 0;
  std::uintptr_t completion_event_ = 0;
  std::uint32_t* host_error_ = nullptr;
  DeepSeekSharedExpertOperations* operations_ = nullptr;
  bool poisoned_ = false;
  bool inflight_ = false;
};

class DeepSeekSharedExpertProvider {
 public:
  virtual ~DeepSeekSharedExpertProvider() = default;
  // Driver, weights, scratch, event and host error must outlive this command.
  virtual Result<DeepSeekSharedExpertDriver*> resolve(
      std::uint32_t layer, const DeepSeekPipelinePlanDescriptor& plan) = 0;
  virtual Status prepare(std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) = 0;
};

// Heap-stable ownership for per-layer drivers and their shared pinned error
// channel and operation adapter. Device/weight allocations must outlive this owner.
// The enclosing runtime must drain its stream before destroying this object.
class DeepSeekSharedExpertRuntimeResources final : public DeepSeekSharedExpertProvider {
 public:
  static Result<std::unique_ptr<DeepSeekSharedExpertRuntimeResources>> Create(
      std::uint32_t first_layer, std::uint32_t last_layer,
      std::uint32_t maximum_tokens, const DeepSeekResidentWeightArena& weights,
      DeepSeekSharedExpertArena scratch, std::uintptr_t input_bf16,
      std::uintptr_t routed_input_bf16,
      std::uintptr_t accumulator_f32, std::uintptr_t output_bf16,
      std::uintptr_t stream, std::uintptr_t completion_event,
      std::unique_ptr<DeepSeekSharedExpertOperations> operations, Allocator& pinned_allocator);
  Result<DeepSeekSharedExpertDriver*> resolve(std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override;
  Status prepare(std::uint32_t layer,
      const DeepSeekPipelinePlanDescriptor& plan) override;
  DeepSeekSharedExpertRuntimeResources(const DeepSeekSharedExpertRuntimeResources&) = delete;
  DeepSeekSharedExpertRuntimeResources& operator=(const DeepSeekSharedExpertRuntimeResources&) = delete;

 private:
  explicit DeepSeekSharedExpertRuntimeResources(Buffer host_error)
      : host_error_(std::move(host_error)) {}
  std::uint32_t first_layer_ = 0;
  std::unique_ptr<DeepSeekSharedExpertOperations> owned_operations_;
  Buffer host_error_;
  std::vector<DeepSeekSharedExpertDriver> drivers_;
  DeepSeekSharedExpertOperations* operations_ = nullptr;
  std::uintptr_t input_bf16_ = 0;
  std::uintptr_t routed_input_bf16_ = 0;
  std::uintptr_t accumulator_f32_ = 0;
  std::uintptr_t stream_ = 0;
  bool poisoned_ = false;
};

// Wraps the routed backend inside the mHC branch boundary. A MoE command is
// successful only after routed experts AND shared finalization have completed.
class DeepSeekSharedExpertStageBackend final : public DeepSeekStageOperatorBackend {
 public:
  DeepSeekSharedExpertStageBackend(DeepSeekStageOperatorBackend& routed,
      DeepSeekSharedExpertProvider& provider) : routed_(&routed), provider_(&provider) {}
  Status launch(const DeepSeekStageOperatorCommand& command,
      const DeepSeekPipelinePlanDescriptor& plan) override;
  Result<DeepSeekStageComputeStatus> poll() override;

 private:
  enum class State { kIdle, kPassthrough, kRouted, kShared, kPoisoned };
  Status poison(Status status);
  DeepSeekStageOperatorBackend* routed_;
  DeepSeekSharedExpertProvider* provider_;
  DeepSeekSharedExpertDriver* shared_ = nullptr;
  std::uint32_t tokens_ = 0;
  State state_ = State::kIdle;
};

}  // namespace pih
