#include "generation.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <thread>

namespace pih::deepseek_plugin {
namespace {
// Never release attention or plan owners while device work is live. Failure
// to prove retirement fails the engine; close retains the fail-stop graph.
class Request final {
 public:
  Request(DeepSeekEngine& engine, std::uint64_t id, std::uint64_t generation)
      : engine_(engine), id_(id), generation_(generation) {}
  ~Request() {
    if (!live_) return;
    try {
      const auto status = Abort();
      if (!status.ok()) (void)engine_.fail_all();
    } catch (...) {
      try { (void)engine_.fail_all(); } catch (...) {}
    }
  }
  Status Abort() {
    if (!live_) return Status::Ok();
    if (abort_attempted_) return abort_status_;
    abort_attempted_ = true;
    abort_status_ = Status::Internal("generation cleanup did not complete");
    // A committed device plan cannot be preempted. Complete exactly that plan,
    // with no output callbacks/new decode plans, before cancelling ownership.
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (output_ != 0 && engine_.execution_live()) {
      if (std::chrono::steady_clock::now() >= cleanup_deadline)
        return abort_status_ = Status::Internal("generation cancellation drain timed out");
      auto progress = engine_.advance_bound_pipeline();
      if (!progress.ok() && progress.code() != StatusCode::kUnavailable)
        return abort_status_ = Status::Internal("generation cancellation drain failed");
      if (engine_.execution_live()) std::this_thread::yield();
    }
    const auto output = ReleaseOutput();
    if (!output.ok()) return abort_status_ = output;
    auto status = engine_.cancel_request(id_, generation_);
    if (!status.ok()) return abort_status_ = status;
    status = engine_.retire_request(id_, generation_);
    if (status.ok()) live_ = false;
    return abort_status_ = status;
  }
  Status ReleaseOutput() {
    if (output_ == 0) return Status::Ok();
    if (engine_.execution_live())
      return Status::FailedPrecondition("generation output still executing");
    const auto status = engine_.acknowledge_output_plan(output_);
    if (status.ok()) output_ = 0;
    return status;
  }
  void Committed(std::uint64_t plan) { output_ = plan; }
  Status Finish() {
    auto status = engine_.finish_request(id_, generation_);
    if (!status.ok()) return status;
    status = engine_.retire_request(id_, generation_);
    if (status.ok()) live_ = false;
    return status;
  }
 private:
  DeepSeekEngine& engine_;
  std::uint64_t id_, generation_, output_ = 0;
  bool live_ = true;
  bool abort_attempted_ = false;
  Status abort_status_ = Status::Ok();
};
}  // namespace

Result<DeepSeekAcceptedTokenSnapshot> Generate(
    DeepSeekEngine& engine, std::span<const std::uint32_t> prompt,
    std::uint32_t maximum_completion, std::uint32_t minimum_completion,
    DeepSeekRequestSamplingConfig sampling, std::uint32_t prefill_chunk,
    std::uint32_t context_capacity, std::uint64_t request_id,
    std::uint64_t request_generation, std::uint64_t& next_plan,
    std::chrono::steady_clock::time_point deadline,
    const GenerationCallbacks& callbacks) {
  constexpr std::uint32_t vocabulary = 129280;
  if (prompt.empty() || prompt.size() > context_capacity ||
      maximum_completion == 0 || minimum_completion > maximum_completion ||
      maximum_completion > context_capacity - prompt.size() ||
      prefill_chunk == 0 || next_plan == 0 ||
      std::any_of(prompt.begin(), prompt.end(),
                  [](auto token) { return token >= vocabulary; }))
    return Status::InvalidArgument("generation token/context limits invalid");
  if (std::chrono::steady_clock::now() >= deadline)
    return Status::DeadlineExceeded("generation deadline elapsed");
  if (callbacks.cancelled && callbacks.cancelled())
    return Status::Unavailable("generation cancelled before submission");
  auto status = engine.submit_request(request_id, request_generation);
  if (!status.ok()) return status;
  Request request(engine, request_id, request_generation);
  auto stop = [&](Status reason) -> Status {
    const auto cleanup = request.Abort();
    if (!cleanup.ok()) return Status::Internal("generation cancellation could not retire ownership");
    return reason;
  };
  auto interrupted = [&]() -> Status {
    if (std::chrono::steady_clock::now() >= deadline)
      return stop(Status::DeadlineExceeded("generation deadline elapsed"));
    if (callbacks.cancelled && callbacks.cancelled())
      return stop(Status::Unavailable("generation cancelled by caller"));
    return Status::Ok();
  };
  status = engine.configure_ledger(request_id, request_generation,
      prompt.size(), maximum_completion, minimum_completion);
  if (!status.ok()) return status;
  sampling.config_id = request_generation;
  status = engine.configure_sampling(request_id, request_generation, sampling);
  if (!status.ok()) return status;
  std::uint32_t processed = 0;
  DeepSeekAcceptedTokenSnapshot accepted;
  for (;;) {
    status = interrupted();
    if (!status.ok()) return status;
    if (next_plan == std::numeric_limits<std::uint64_t>::max())
      return Status::ResourceExhausted("generation plan identity exhausted");
    const auto plan = next_plan++;
    const bool prefill = processed < prompt.size();
    DeepSeekProductionRankPlanSeed seed;
    if (prefill) {
      // The fixed recent-KV ring has 128 rows. Finish each token's attention
      // before a later token can overwrite a row still visible to it.
      // prefill_chunk remains the admitted capacity, not a batching promise.
      constexpr std::uint32_t count = 1;
      seed.token_ids.assign(prompt.begin() + processed, prompt.begin() + processed + count);
      seed.positions.resize(count);
      std::iota(seed.positions.begin(), seed.positions.end(), processed);
      processed += count;
    } else {
      if (!accepted.pending_input_token || accepted.model_processed_length >= context_capacity)
        return Status::Internal("generation decode frontier invalid");
      seed.token_ids = {*accepted.pending_input_token};
      seed.positions = {static_cast<std::uint32_t>(accepted.model_processed_length)};
    }
    seed.table_position_count = seed.positions.back() + 1;
    const auto count = static_cast<std::uint32_t>(seed.token_ids.size());
    const bool sample = processed == prompt.size();
    std::vector<DeepSeekProductionRankPlanSeed> seeds(engine.world_size(), seed);
    status = engine.prepare_production_pipeline(
        {engine.epoch(), plan, prefill ? DeepSeekPlanPhase::kPrefill : DeepSeekPlanPhase::kDecode, count, 1},
        {{request_id, request_generation, false}}, std::move(seeds), 1, count, vocabulary, sample);
    if (!status.ok()) return status;
    status = engine.commit_pipeline();
    if (!status.ok()) return status;
    request.Committed(plan);
    while (engine.execution_live()) {
      status = interrupted();
      if (!status.ok()) return status;
      status = engine.advance_bound_pipeline();
      if (!status.ok() && status.code() != StatusCode::kUnavailable) return status;
      if (engine.execution_live()) std::this_thread::yield();
    }
    status = request.ReleaseOutput();
    if (!status.ok()) return status;
    // Recheck after device work too: a late final token must not turn an
    // expired/disconnected request into successful output.
    status = interrupted();
    if (!status.ok()) return status;
    if (!sample) continue;
    auto snapshot = engine.accepted_token_snapshot(request_id, request_generation);
    if (!snapshot.ok()) return snapshot.status();
    if (snapshot->accepted_completion_count != accepted.accepted_completion_count + 1 ||
        snapshot->token_ids.size() != snapshot->accepted_completion_count)
      return Status::Internal("generation accepted-token frontier invalid");
    accepted = std::move(*snapshot);
    if (callbacks.on_token && !callbacks.on_token(accepted))
      return stop(Status::Unavailable("generation output sink cancelled"));
    status = interrupted();
    if (!status.ok()) return status;
    if (accepted.finish_reason != DeepSeekFinishReason::kNone) break;
  }
  status = request.Finish();
  if (!status.ok()) return status;
  return accepted;
}
}  // namespace pih::deepseek_plugin
