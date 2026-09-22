#include "pih/model/deepseek_expert_swiglu_oracle.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "pih/core/checked_math.h"

namespace pih {

Result<std::vector<float>> DeepSeekExpertSwiGluOracle::Apply(
    std::span<const float> gate, std::span<const float> up,
    std::uint32_t token_count, std::uint32_t intermediate_size, float limit) {
  if (token_count == 0 || intermediate_size == 0 || !std::isfinite(limit) ||
      limit <= 0.0F) {
    return Status::InvalidArgument("DeepSeek expert SwiGLU geometry is invalid");
  }
  auto element_count = checked_mul_u64(token_count, intermediate_size);
  if (!element_count.ok() || *element_count > std::numeric_limits<std::size_t>::max() ||
      gate.size() != *element_count || up.size() != *element_count) {
    return Status::InvalidArgument("DeepSeek expert SwiGLU shape is invalid");
  }

  std::vector<float> output(static_cast<std::size_t>(*element_count));
  for (std::size_t index = 0; index < output.size(); ++index) {
    const auto gate_value = gate[index];
    const auto up_value = up[index];
    if (!std::isfinite(gate_value) || !std::isfinite(up_value)) {
      return Status::InvalidArgument(
          "DeepSeek expert SwiGLU input is nonfinite");
    }
    const auto clamped_gate = std::min(gate_value, limit);
    const auto clamped_up = std::clamp(up_value, -limit, limit);
    const auto silu = clamped_gate / (1.0F + std::exp(-clamped_gate));
    const auto value = silu * clamped_up;
    if (!std::isfinite(value)) {
      return Status::InvalidArgument(
          "DeepSeek expert SwiGLU output is nonfinite");
    }
    output[index] = value;
  }
  return output;
}

}  // namespace pih
