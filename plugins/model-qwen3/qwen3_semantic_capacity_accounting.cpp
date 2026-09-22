#include "pih/model/qwen3_semantic_capacity_accounting.h"

#include <algorithm>

#include "pih/core/checked_math.h"

namespace pih {

Result<QwenSemanticCapacityPair> qwen_semantic_capacity_pair(
    std::uint64_t device_baseline_bytes,
    std::uint64_t pinned_baseline_bytes,
    std::uint64_t tap_device_peak_bytes,
    std::uint64_t tap_pinned_peak_bytes,
    std::uint64_t semantic_pinned_arena_bytes) {
  if (device_baseline_bytes == 0 || pinned_baseline_bytes == 0 ||
      tap_device_peak_bytes == 0 || tap_pinned_peak_bytes == 0 ||
      semantic_pinned_arena_bytes == 0) {
    return Status::InvalidArgument(
        "Qwen semantic capacity accounting lacks a required owner");
  }
  auto instrumented_device =
      checked_add_u64(device_baseline_bytes, tap_device_peak_bytes);
  if (!instrumented_device.ok()) return instrumented_device.status();
  auto instrumented_pinned = checked_add_u64(
      pinned_baseline_bytes,
      std::max(tap_pinned_peak_bytes, semantic_pinned_arena_bytes));
  if (!instrumented_pinned.ok()) return instrumented_pinned.status();
  auto control_pinned = checked_add_u64(
      pinned_baseline_bytes, semantic_pinned_arena_bytes);
  if (!control_pinned.ok()) return control_pinned.status();
  return QwenSemanticCapacityPair{
      *instrumented_device, *instrumented_pinned, device_baseline_bytes,
      *control_pinned};
}

}  // namespace pih
