#include "rank_worker_loop.h"
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<RankWorkerLoop>> RankWorkerLoop::Create(const FlashConfig& config, const RankWorkerResources& r,
    SamplingIdentity identity, SamplingParameters sampling, std::uint32_t prompt, Clock::time_point deadline) {
  const auto valid = ValidateSamplingParameters(sampling); if (!valid.ok()) return valid;
  if (!identity.epoch || identity.plan_seq || !identity.sequence_generation || !identity.sampling_config_id ||
      sampling.ordinal || sampling.suppressed_count || !prompt || prompt > 4096 || Clock::now() >= deadline ||
      r.sequence.failed() || r.sequence.next_position() || r.sequence.next_layer() ||
      r.memory.state() != InferenceMemoryState::kReady || !r.requests.receiver_ready() || !r.receipts.sender_ready() ||
      r.requests.rank() != r.receipts.rank() || r.requests.world() != r.receipts.world() ||
      r.weights.catalog().rank() != r.requests.rank() || r.weights.catalog().world_size() != r.requests.world() ||
      r.weights.catalog().config_sha256() != config.config_sha256() || !r.communicator || !r.completion_event)
    return Status::InvalidArgument("Worker sequence, identity, resources or deadline invalid");
  try {
    auto worker = std::unique_ptr<RankWorkerLoop>(new RankWorkerLoop(config, r));
    worker->identity_ = identity; worker->sampling_ = sampling; worker->prompt_ = prompt; worker->deadline_ = deadline;
    return worker;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Worker loop allocation failed"); }
}
RankWorkerLoop::~RankWorkerLoop() { if (state_ != RankWorkerState::kReceiving && state_ != RankWorkerState::kComplete) Fail(); }
void RankWorkerLoop::Fail() noexcept {
  state_ = RankWorkerState::kFailed; resources_.sequence.Fail(); resources_.memory.Quarantine();
}
Result<RankWorkerState> RankWorkerLoop::Poll() {
  try {
    auto result = PollImpl(); if (!result.ok()) Fail(); return result;
  } catch (const std::bad_alloc&) { Fail(); return Status::ResourceExhausted("Worker continuation allocation failed"); }
}
Result<RankWorkerState> RankWorkerLoop::PollImpl() {
  if (state_ == RankWorkerState::kFailed) return Status::FailedPrecondition("Worker loop failed");
  if (state_ == RankWorkerState::kComplete) return state_;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Worker sequence deadline expired");
  auto& r = resources_;
  if (state_ == RankWorkerState::kReceiving) {
    auto received = r.requests.PollReceive(deadline_); if (!received.ok()) return received.status();
    if (!*received) return state_;
    const auto& request = **received; const auto& id = request.sampling.identity; const auto& p = request.sampling.parameters;
    if (id.epoch != identity_.epoch || id.sequence_generation != identity_.sequence_generation ||
        id.sampling_config_id != identity_.sampling_config_id || p.ordinal != ordinal_ ||
        p.temperature != sampling_.temperature || p.top_p != sampling_.top_p || p.top_k != sampling_.top_k ||
        p.seed != sampling_.seed || p.logprobs != sampling_.logprobs || p.top_count != sampling_.top_count ||
        (!request.finish && (id.plan_seq <= last_plan_ || request.count != (last_plan_ ? 1U : prompt_))))
      return Status::FailedPrecondition("Worker request identity, sampling or token shape drifted");
    if (request.finish) {
      if (!last_plan_ || id.plan_seq != last_plan_ || request.sampling.processed_length != r.sequence.next_position() ||
          r.memory.state() != InferenceMemoryState::kReady)
        return Status::FailedPrecondition("Worker terminal request does not match retired step");
      RankStepReceipt terminal; terminal.identity_ = id; terminal.ordinal_ = ordinal_;
      terminal.processed_ = r.sequence.next_position(); terminal.rank_ = r.receipts.rank(); terminal.world_ = r.receipts.world();
      terminal.terminal_ = true;
      const auto queued = r.receipts.Queue(terminal); if (!queued.ok()) return queued;
      state_ = RankWorkerState::kSendingTerminal; return state_;
    }
    last_plan_ = id.plan_seq;
    auto& workspace = r.sequence.next_position() ? r.decode_workspace : r.prefill_workspace;
    const auto started = r.memory.StartRequest(config_, r.sequence, r.hashes, request, r.weights, workspace,
        r.communicator, r.completion_event, deadline_);
    if (!started.ok()) return started;
    state_ = RankWorkerState::kExecuting; return state_;
  }
  if (state_ == RankWorkerState::kExecuting) {
    auto ready = r.memory.Advance(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    auto receipt = r.memory.Receipt(); if (!receipt.ok()) return receipt.status();
    const auto queued = r.receipts.Queue(*receipt); if (!queued.ok()) return queued;
    state_ = RankWorkerState::kSending; return state_;
  }
  auto sent = r.receipts.PollSend(deadline_); if (!sent.ok()) return sent.status();
  if (!*sent) return state_;
  if (state_ == RankWorkerState::kSendingTerminal) { state_ = RankWorkerState::kComplete; return state_; }
  if (sampling_.temperature != 0) {
    if (ordinal_ == std::numeric_limits<std::uint64_t>::max()) return Status::ResourceExhausted("Worker sampling ordinal exhausted");
    ++ordinal_;
  }
  state_ = RankWorkerState::kReceiving; return state_;
}
}  // namespace pih::deepseek_v41
