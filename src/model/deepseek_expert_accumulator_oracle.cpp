#include "pih/model/deepseek_expert_accumulator_oracle.h"

#include <cmath>

namespace pih {

Result<DeepSeekExpertAccumulatorOracle>
DeepSeekExpertAccumulatorOracle::Create(std::uint32_t token_count) {
  if (token_count == 0) {
    return Status::InvalidArgument(
        "DeepSeek expert accumulator requires packed tokens");
  }
  DeepSeekExpertAccumulatorOracle oracle;
  oracle.token_count_ = token_count;
  oracle.accumulator_.resize(
      static_cast<std::size_t>(token_count) * kHiddenSize, 0.0F);
  return oracle;
}

Status DeepSeekExpertAccumulatorOracle::add_expert(
    std::uint16_t expert_id, std::span<const DeepSeekExpertRoute> routes,
    std::span<const float> expert_output) {
  if (finalized_ || expert_id < next_minimum_expert_ || expert_id >= 256 ||
      routes.empty() ||
      expert_output.size() != routes.size() * kHiddenSize) {
    return Status::FailedPrecondition(
        "DeepSeek expert accumulation order or shape is invalid");
  }
  std::uint32_t previous_token = 0;
  bool first = true;
  for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
    const auto& route = routes[route_index];
    if (route.expert_id != expert_id || route.token_index >= token_count_ ||
        !std::isfinite(route.weight) ||
        (!first && route.token_index <= previous_token)) {
      return Status::InvalidArgument(
          "DeepSeek expert route slice is not canonical");
    }
    first = false;
    previous_token = route.token_index;
    const auto source = route_index * kHiddenSize;
    for (std::uint32_t column = 0; column < kHiddenSize; ++column) {
      const auto value = expert_output[source + column];
      if (!std::isfinite(value)) {
        return Status::InvalidArgument(
            "DeepSeek expert output is non-finite");
      }
    }
  }
  auto candidate = accumulator_;
  for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
    const auto destination =
        static_cast<std::size_t>(routes[route_index].token_index) * kHiddenSize;
    const auto source = route_index * kHiddenSize;
    for (std::uint32_t column = 0; column < kHiddenSize; ++column) {
      // The official Expert applies the route weight to the FP32 SwiGLU
      // intermediate before the BF16 boundary and W2 quantization. Therefore
      // expert_output is already weighted and must not be scaled again here.
      candidate[destination + column] += expert_output[source + column];
      if (!std::isfinite(candidate[destination + column])) {
        return Status::Internal("DeepSeek expert accumulator overflowed");
      }
    }
  }
  accumulator_ = std::move(candidate);
  next_minimum_expert_ = static_cast<std::uint16_t>(expert_id + 1U);
  return Status::Ok();
}

Result<std::vector<float>> DeepSeekExpertAccumulatorOracle::finalize(
    std::span<const float> shared_expert_output) {
  if (finalized_ || shared_expert_output.size() != accumulator_.size()) {
    return Status::FailedPrecondition(
        "DeepSeek shared expert finalize shape is invalid");
  }
  std::vector<float> result(accumulator_.size());
  for (std::size_t index = 0; index < result.size(); ++index) {
    result[index] = accumulator_[index] + shared_expert_output[index];
    if (!std::isfinite(result[index])) {
      return Status::InvalidArgument(
          "DeepSeek shared expert finalize is non-finite");
    }
  }
  finalized_ = true;
  return result;
}

}  // namespace pih
