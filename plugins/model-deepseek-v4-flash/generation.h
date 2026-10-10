#pragma once

#include <chrono>
#include <functional>
#include <span>
#include <memory>
#include "pih/model/deepseek_engine.h"

namespace pih::deepseek_plugin {

class BatchEngine {
 public:
  virtual ~BatchEngine() = default;
  virtual Status bind_execution_thread() { return Status::Ok(); }
  virtual Status submit_request(std::uint64_t, std::uint64_t) = 0;
  virtual Status configure_ledger(std::uint64_t, std::uint64_t, std::uint64_t, std::uint32_t, std::uint32_t) = 0;
  virtual Status configure_sampling(std::uint64_t, std::uint64_t, DeepSeekRequestSamplingConfig) = 0;
  virtual Status cancel_request(std::uint64_t, std::uint64_t) = 0;
  virtual Status finish_request(std::uint64_t, std::uint64_t) = 0;
  virtual Status retire_request(std::uint64_t, std::uint64_t) = 0;
  virtual Status prepare_production_pipeline(DeepSeekPipelinePlanDescriptor,
      std::vector<DeepSeekRequestIdentity>, std::vector<DeepSeekProductionRankPlanSeed>,
      std::uint32_t, std::uint32_t, std::uint32_t, bool) = 0;
  virtual Status commit_pipeline() = 0;
  virtual bool execution_live() const = 0;
  virtual Status advance_bound_pipeline() = 0;
  virtual Status acknowledge_output_plan(std::uint64_t) = 0;
  virtual Result<DeepSeekAcceptedTokenSnapshot> accepted_token_snapshot(std::uint64_t, std::uint64_t) = 0;
  virtual void fail_all() = 0;
  virtual std::uint64_t epoch() const = 0;
  virtual std::uint32_t world_size() const = 0;
};

class BatchGenerator final {
 public:
  BatchGenerator(DeepSeekEngine& engine, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, std::uint32_t maximum_pending);
  BatchGenerator(std::unique_ptr<BatchEngine> engine, std::uint32_t maximum_sequences,
      std::uint32_t maximum_tokens, std::uint32_t maximum_pending);
  ~BatchGenerator();
  BatchGenerator(const BatchGenerator&) = delete;
  BatchGenerator& operator=(const BatchGenerator&) = delete;
  Result<DeepSeekAcceptedTokenSnapshot> Generate(
      std::span<const std::uint32_t> prompt, std::uint32_t maximum_completion,
      std::uint32_t minimum_completion, DeepSeekRequestSamplingConfig sampling,
      std::uint32_t context_capacity, std::uint64_t request_id,
      std::uint64_t request_generation, std::chrono::steady_clock::time_point deadline,
      const struct GenerationCallbacks& callbacks);
  std::uint32_t maximum_sequences() const noexcept;
 private:
  struct Impl;
  std::unique_ptr<BatchEngine> engine_;
  std::unique_ptr<Impl> impl_;
};

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
