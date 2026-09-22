#pragma once

#include "pih/core/buffer.h"
#include "pih/model/qwen3_bf16_engine_resource_plan.h"
#include "pih/model/qwen3_bf16_step_resource_factory.h"

namespace pih {

class QwenBf16EngineDeviceArenas final {
 public:
  static Result<QwenBf16EngineDeviceArenas> Allocate(
      const QwenBf16EngineResourcePlan& plan, Allocator& allocator,
      std::int32_t owning_rank);

  QwenBf16EngineDeviceArenas(const QwenBf16EngineDeviceArenas&) = delete;
  QwenBf16EngineDeviceArenas& operator=(
      const QwenBf16EngineDeviceArenas&) = delete;
  QwenBf16EngineDeviceArenas(QwenBf16EngineDeviceArenas&&) noexcept = default;
  QwenBf16EngineDeviceArenas& operator=(
      QwenBf16EngineDeviceArenas&&) noexcept = default;

  [[nodiscard]] QwenBf16StepDeviceOwners step_owners() const noexcept;
  [[nodiscard]] void* linear_workspace() const noexcept {
    return linear_workspace_.data();
  }
  [[nodiscard]] std::uint64_t linear_workspace_bytes() const noexcept {
    return linear_workspace_.size_bytes();
  }
  [[nodiscard]] std::uint64_t total_bytes() const noexcept {
    return total_bytes_;
  }

 private:
  QwenBf16EngineDeviceArenas(
      Buffer step_staging, Buffer activations, Buffer mlp, Buffer rope,
      Buffer logits, Buffer sampled_token, Buffer device_error,
      Buffer kv_backing, Buffer kv_metadata, Buffer linear_workspace,
      std::uint64_t total_bytes)
      : step_staging_(std::move(step_staging)),
        activations_(std::move(activations)), mlp_(std::move(mlp)),
        rope_(std::move(rope)), logits_(std::move(logits)),
        sampled_token_(std::move(sampled_token)),
        device_error_(std::move(device_error)),
        kv_backing_(std::move(kv_backing)),
        kv_metadata_(std::move(kv_metadata)),
        linear_workspace_(std::move(linear_workspace)),
        total_bytes_(total_bytes) {}

  Buffer step_staging_;
  Buffer activations_;
  Buffer mlp_;
  Buffer rope_;
  Buffer logits_;
  Buffer sampled_token_;
  Buffer device_error_;
  Buffer kv_backing_;
  Buffer kv_metadata_;
  Buffer linear_workspace_;
  std::uint64_t total_bytes_;
};

}  // namespace pih
