#include "pih/backend/cuda/deepseek_indexer_projection.h"

#include <algorithm>
#include <array>

namespace pih {

Status validate_deepseek_indexer_projection_launch(
    const DeepSeekIndexerProjectionLaunch& value) {
  const std::array pointers{
      value.qr_bf16, value.hidden_bf16, value.wq_b_e4m3,
      value.wq_b_scale_bits, value.qr_e4m3, value.qr_scale_bits,
      value.weights_proj_bf16, value.frequencies_f32, value.positions_u32,
      value.query_bf16, value.head_weight_f32, value.error_flag_u32,
      value.stream};
  if (!std::ranges::all_of(pointers, [](auto pointer) { return pointer != 0; }) ||
      value.token_count == 0 || value.token_count > 4096 ||
      value.table_position_count == 0 ||
      value.table_position_count > 1048576 ||
      value.query_bf16 == value.qr_bf16 ||
      value.head_weight_f32 == value.hidden_bf16) {
    return Status::InvalidArgument(
        "DeepSeek indexer projection launch is invalid");
  }
  // Writable allocations must not share a base address with any other view.
  // The resource owner is responsible for capacities and non-overlapping ranges.
  const std::array inputs{value.qr_bf16, value.hidden_bf16, value.wq_b_e4m3,
                          value.wq_b_scale_bits, value.weights_proj_bf16,
                          value.frequencies_f32, value.positions_u32};
  const std::array outputs{value.qr_e4m3, value.qr_scale_bits, value.query_bf16,
                           value.head_weight_f32, value.error_flag_u32};
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (std::ranges::find(inputs, outputs[i]) != inputs.end() ||
        std::find(outputs.begin(), outputs.begin() + i, outputs[i]) !=
            outputs.begin() + i) {
      return Status::InvalidArgument("DeepSeek indexer projection aliases writable storage");
    }
  }
  return Status::Ok();
}

}  // namespace pih
