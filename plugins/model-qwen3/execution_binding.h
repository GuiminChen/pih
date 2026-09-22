#pragma once

#include <cstdint>
#include <cstring>

#include "pih/contracts/execution_default_v1.h"

namespace pih::qwen_plugin {

// The provider contract and both precision-specific engine plans share these
// fixed ceilings. Binding runs before engine construction, never per token.
inline constexpr std::uint32_t kMaximumStepTokens = 4096;
inline constexpr std::uint32_t kMaximumSequenceTokens = 40960;
inline constexpr std::uint32_t kSlotCount = 2560;
inline constexpr std::uint32_t kMaximumBatchSequences = 32;

inline bool ValidExecutionApi(const pih_execution_default_api_v1* api) noexcept {
  return api && api->struct_size == sizeof(*api) &&
      api->contract_version == PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 &&
      api->context && api->compile_capacity;
}

inline pih_status_v1 ExecutionBindingStatus(std::uint32_t code,
                                           const char* message) noexcept {
  pih_status_v1 status{};
  status.struct_size = sizeof(status);
  status.abi_version = PIH_STATUS_ABI_VERSION_V1;
  status.code = code;
  std::strncpy(status.message, message, sizeof(status.message) - 1);
  return status;
}

inline pih_status_v1 CompileExecutionCapacity(
    const pih_execution_default_api_v1* api) noexcept {
  if (!ValidExecutionApi(api))
    return ExecutionBindingStatus(PIH_STATUS_FAILED_PRECONDITION_V1,
                                  "qwen_execution_provider_invalid");
  const pih_execution_capacity_request_v1 request{
      sizeof(request), PIH_EXECUTION_DEFAULT_ABI_VERSION_V1,
      kMaximumStepTokens, kMaximumBatchSequences, kMaximumBatchSequences, 0};
  pih_execution_capacity_v1 capacity{};
  capacity.struct_size = sizeof(capacity);
  capacity.abi_version = PIH_EXECUTION_DEFAULT_ABI_VERSION_V1;
  try {
    const auto status = api->compile_capacity(api->context, &request, &capacity);
    if (!pih_status_is_valid_v1(&status))
      return ExecutionBindingStatus(PIH_STATUS_INTERNAL_V1,
                                    "qwen_execution_status_invalid");
    if (!pih_status_is_ok_v1(&status)) return status;
    if (capacity.struct_size != sizeof(capacity) ||
        capacity.abi_version != PIH_EXECUTION_DEFAULT_ABI_VERSION_V1 ||
        capacity.max_pipeline_tokens != kMaximumStepTokens ||
        capacity.maximum_sequences != kMaximumBatchSequences ||
        capacity.expert_tokens != kMaximumStepTokens)
      return ExecutionBindingStatus(PIH_STATUS_FAILED_PRECONDITION_V1,
                                    "qwen_execution_capacity_mismatch");
    return status;
  } catch (...) {
    return ExecutionBindingStatus(PIH_STATUS_INTERNAL_V1,
                                  "qwen_execution_provider_exception");
  }
}

}  // namespace pih::qwen_plugin
