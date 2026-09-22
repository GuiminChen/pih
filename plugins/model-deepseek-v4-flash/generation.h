#pragma once

#include <chrono>
#include <functional>
#include <span>
#include "pih/model/deepseek_engine.h"

namespace pih::deepseek_plugin {

struct GenerationCallbacks final {
  // Borrowed synchronous callbacks; must not reenter or mutate the engine.
  // A false token callback requests cancellation (e.g. a disconnected sink).
  std::function<bool()> cancelled;
  std::function<bool(const DeepSeekAcceptedTokenSnapshot&)> on_token;
};

// Plugin-private scheduler. Callers serialize it with engine teardown and own
// the monotonically increasing request/plan identities for this activation.
// Cancellation/deadline drains the current committed plan (up to five minutes),
// acknowledges output, then retires the request. No new output is delivered
// after interruption. Device steps are cooperative, not preemptible.
Result<DeepSeekAcceptedTokenSnapshot> Generate(
    DeepSeekEngine& engine, std::span<const std::uint32_t> prompt,
    std::uint32_t maximum_completion, std::uint32_t minimum_completion,
    DeepSeekRequestSamplingConfig sampling, std::uint32_t prefill_chunk,
    std::uint32_t context_capacity, std::uint64_t request_id,
    std::uint64_t request_generation, std::uint64_t& next_plan,
    std::chrono::steady_clock::time_point deadline,
    const GenerationCallbacks& callbacks = {});

}  // namespace pih::deepseek_plugin
