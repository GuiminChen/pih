#pragma once

#include <cstdint>

#include "pih/model/qwen3_bf16_execution_arena.h"
#include "pih/model/qwen3_bf16_resource_set.h"
#include "pih/model/qwen3_bf16_step_staging_layout.h"

namespace pih {

struct QwenBf16DeviceArenaOwner final {
  std::uintptr_t base;
  std::uint64_t bytes;
  std::uint64_t generation;
};

struct QwenBf16StepDeviceOwners final {
  QwenBf16DeviceArenaOwner step_staging;
  QwenBf16DeviceArenaOwner activations;
  QwenBf16DeviceArenaOwner mlp;
  QwenBf16DeviceArenaOwner rope;
  QwenBf16DeviceArenaOwner logits;
  QwenBf16DeviceArenaOwner sampled_token;
  QwenBf16DeviceArenaOwner device_error;
  QwenBf16DeviceArenaOwner kv_backing;
  QwenBf16DeviceArenaOwner kv_slot_states;
};

class QwenBf16StepResourceFactory final {
 public:
  static Result<QwenBf16ResourceSet> Create(
      std::uint64_t request_generation, std::int32_t owning_rank,
      std::uint32_t slot_count,
      const QwenBf16ExecutionArenaLayout& execution_layout,
      const QwenBf16StepStagingLayout& staging_layout,
      const QwenBf16StepDeviceOwners& owners);
};

}  // namespace pih
