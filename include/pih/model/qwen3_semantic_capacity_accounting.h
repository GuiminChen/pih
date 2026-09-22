#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

struct QwenSemanticCapacityPair final {
  std::uint64_t instrumented_device_peak_bytes;
  std::uint64_t instrumented_pinned_peak_bytes;
  std::uint64_t control_device_peak_bytes;
  std::uint64_t control_pinned_peak_bytes;
};

Result<QwenSemanticCapacityPair> qwen_semantic_capacity_pair(
    std::uint64_t device_baseline_bytes,
    std::uint64_t pinned_baseline_bytes,
    std::uint64_t tap_device_peak_bytes,
    std::uint64_t tap_pinned_peak_bytes,
    std::uint64_t semantic_pinned_arena_bytes);

}  // namespace pih
