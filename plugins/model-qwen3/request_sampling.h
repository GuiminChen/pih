#pragma once

#include "pih/core/bounded_json.h"
#include "pih/scheduler/controller_request_arena.h"
#include <cmath>
#include <stdexcept>

namespace pih::qwen_plugin {
inline ControllerRequestSampling RequestSampling(const JsonValue& request) {
  ControllerRequestSampling result;
  const auto* n = request.at("n");
  if (n && (!n->is_integer() || n->integer() != 1))
    throw std::invalid_argument("native Qwen supports integer n=1 only");
  auto number = [&](const char* name, double fallback) {
    const auto* field = request.at(name);
    if (!field) return fallback;
    if (!field->is_number() || !std::isfinite(field->number()))
      throw std::invalid_argument("sampling parameter must be a finite number");
    return field->number();
  };
  const double temperature = number("temperature", 0.0);
  const double top_p = number("top_p", 1.0);
  if (temperature < 0.0 || temperature > 2.0 || top_p <= 0.0 || top_p > 1.0)
    throw std::invalid_argument("temperature must be 0..2 and top_p must be (0,1]");
  if (temperature == 0.0 && top_p != 1.0)
    throw std::invalid_argument("greedy temperature=0 requires top_p=1");
  result.temperature = static_cast<float>(temperature);
  result.top_p = static_cast<float>(top_p);
  if ((temperature > 0.0 && result.temperature == 0.0F) || result.top_p == 0.0F)
    throw std::invalid_argument("sampling parameter underflows the native FP32 contract");
  result.mode = temperature == 0.0 ? ControllerSamplingMode::kGreedy
                                  : ControllerSamplingMode::kStochastic;
  const auto* seed = request.at("seed");
  if (seed) {
    if (!seed->is_integer() || seed->integer() < 0)
      throw std::invalid_argument("seed must be an integer in 0..INT64_MAX");
    result.effective_seed = static_cast<uint64_t>(seed->integer());
  }
  // An omitted seed is deliberately zero, not hidden host RNG state.
  result.stop_token_count = 2;
  result.stop_token_ids[0] = 151643;
  result.stop_token_ids[1] = 151645;
  return result;
}
}  // namespace pih::qwen_plugin
