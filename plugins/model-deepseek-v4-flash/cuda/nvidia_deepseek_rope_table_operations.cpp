#include "pih/backend/cuda/nvidia_deepseek_rope_table_operations.h"

#include "pih/plugin_sdk/status_bridge.h"

namespace pih {

Status NvidiaDeepSeekRopeTableOperations::initialize(
    DeepSeekRopeTableLaunch launch) {
  if (kernels_ == nullptr) {
    return Status::FailedPrecondition("Kernel Pack capability is unavailable");
  }
  pih_deepseek_rope_table_launch_v1 request{
      sizeof(pih_deepseek_rope_table_launch_v1),
      PIH_DEEPSEEK_KERNELS_ABI_VERSION_V1,
      launch.output_f32, launch.error_flag_u32, launch.stream,
      launch.position_count, launch.rope_dimension, launch.theta,
      launch.scaling_factor, launch.original_maximum_positions,
      launch.beta_fast, launch.beta_slow, launch.yarn ? 1U : 0U};
  return plugin_status(kernels_->launch_rope_table(&request));
}

Status NvidiaDeepSeekRopeTableOperations::synchronize(std::uintptr_t stream) {
  if (stream == 0 || context_identity_ == 0 || async_api_ == nullptr ||
      async_api_->synchronize_stream == nullptr)
    return Status::FailedPrecondition("CUDA RoPE stream is unavailable");
  return plugin_status(async_api_->synchronize_stream(
      async_api_->context, context_identity_, stream));
}

}  // namespace pih
