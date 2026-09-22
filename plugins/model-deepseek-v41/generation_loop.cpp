#include "generation_loop.h"
#include <algorithm>
#include <limits>
#include <new>

namespace pih::deepseek_v41 {
Result<std::unique_ptr<GenerationLoop>> GenerationLoop::Create(TokenLedger& ledger, std::span<const std::uint32_t> prompt,
    std::span<const std::string_view> bytes, std::span<RankRequestChannel* const> requests,
    std::span<RankReceiptChannel* const> receipts, RankProcessWatch& processes, std::uint64_t plan, Clock::time_point deadline) {
  if (ledger.failed() || !ledger.records().empty() || ledger.pending_output() || !plan || prompt.empty() || prompt.size() > 4096 ||
      prompt.size() != ledger.prompt_tokens() || bytes.size() != FlashConfig::kVocabularySize ||
      (requests.size() != 2 && requests.size() != 4 && requests.size() != 8) ||
      receipts.size() != requests.size() || processes.world() != requests.size() || Clock::now() >= deadline)
    return Status::InvalidArgument("Generation admission, channel count or deadline invalid");
  for (const auto token : prompt) if (token >= bytes.size()) return Status::InvalidArgument("Generation prompt token invalid");
  for (const auto token : bytes) if (token.size() > 8192 || (token.size() && !token.data()))
    return Status::InvalidArgument("Generation token byte table exceeds per-token bound");
  for (std::size_t i = 0; i < requests.size(); ++i)
    if (!requests[i] || !receipts[i] || !requests[i]->sender_ready() || !receipts[i]->receiver_ready() ||
        requests[i]->rank() != i || receipts[i]->rank() != i ||
        requests[i]->world() != requests.size() || receipts[i]->world() != requests.size() ||
        requests[i]->peer_pid() != processes.pid(static_cast<unsigned>(i)) || receipts[i]->peer_pid() != processes.pid(static_cast<unsigned>(i)))
      return Status::InvalidArgument("Generation channels must match rank order");
  try {
    auto loop = std::unique_ptr<GenerationLoop>(new GenerationLoop(ledger));
    loop->processes_ = &processes;
    loop->world_ = static_cast<std::uint32_t>(requests.size()); loop->plan_ = plan; loop->deadline_ = deadline; loop->token_bytes_ = bytes;
    loop->request_.count = static_cast<std::uint32_t>(prompt.size());
    std::copy(prompt.begin(), prompt.end(), loop->request_.tokens.begin());
    std::copy(requests.begin(), requests.end(), loop->requests_.begin()); std::copy(receipts.begin(), receipts.end(), loop->receipts_.begin());
    return loop;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Generation admission allocation failed"); }
}
GenerationLoop::~GenerationLoop() { if (state_ != GenerationState::kComplete) Fail(); }
void GenerationLoop::Fail() noexcept { state_ = GenerationState::kFailed; ledger_.Fail(); if (collector_) collector_->Fail(); }
Result<GenerationState> GenerationLoop::Poll() {
  try {
    auto result = PollImpl(); if (!result.ok()) Fail(); return result;
  } catch (const std::bad_alloc&) { Fail(); return Status::ResourceExhausted("Generation continuation allocation failed"); }
}
Result<GenerationState> GenerationLoop::PollImpl() {
  if (state_ == GenerationState::kFailed || ledger_.failed()) return Status::FailedPrecondition("Generation failed");
  if (state_ == GenerationState::kComplete || state_ == GenerationState::kOutputReady) return state_;
  if (Clock::now() >= deadline_) return Status::DeadlineExceeded("Generation deadline expired");
  const auto live = processes_->CheckLive(); if (!live.ok()) return live;
  if (state_ == GenerationState::kFinishQueue) {
    request_.finish = true; request_.count = 0; request_.tokens.fill(0);
    request_.sampling.parameters.ordinal = ledger_.ordinal();
    request_.sampling.processed_length = ledger_.model_processed_length();
    for (unsigned rank = 0; rank < world_; ++rank) {
      const auto queued = requests_[rank]->Queue(request_); if (!queued.ok()) return queued;
    }
    sent_ = 0; state_ = GenerationState::kFinishSend; return state_;
  }
  if (state_ == GenerationState::kFinishReceive) {
    for (unsigned rank = 0; rank < world_; ++rank) {
      if (sent_ & (1U << rank)) continue;
      auto receipt = receipts_[rank]->PollTerminal(request_.sampling.identity, request_.sampling.parameters.ordinal,
          request_.sampling.processed_length, deadline_);
      if (!receipt.ok()) return receipt.status();
      if (*receipt) sent_ |= 1U << rank;
    }
    if (sent_ == ((1U << world_) - 1)) state_ = GenerationState::kComplete;
    return state_;
  }
  if (state_ == GenerationState::kPreparing) {
    auto collector = RankReceiptCollector::Begin(ledger_, plan_, {receipts_.data(), world_}, deadline_);
    if (!collector.ok()) {
      if (collector.status().code() == StatusCode::kResourceExhausted && !ledger_.failed()) return state_;
      return collector.status();
    }
    collector_ = std::move(*collector); request_.sampling = collector_->request();
    const auto valid = request_.Validate(); if (!valid.ok()) return valid;
    for (unsigned rank = 0; rank < world_; ++rank) {
      const auto queued = requests_[rank]->Queue(request_); if (!queued.ok()) return queued;
    }
    sent_ = 0; state_ = GenerationState::kSending; return state_;
  }
  if (state_ == GenerationState::kSending || state_ == GenerationState::kFinishSend) {
    for (unsigned rank = 0; rank < world_; ++rank) {
      if (sent_ & (1U << rank)) continue;
      auto sent = requests_[rank]->PollSend(deadline_); if (!sent.ok()) return sent.status();
      if (*sent) sent_ |= 1U << rank;
    }
    if (sent_ == ((1U << world_) - 1)) {
      state_ = state_ == GenerationState::kFinishSend ? GenerationState::kFinishReceive : GenerationState::kCollecting;
      sent_ = 0;
    }
    return state_;
  }
  auto ready = collector_->Poll(); if (!ready.ok()) return ready.status();
  if (!*ready) return state_;
  auto candidate = collector_->Candidate(); if (!candidate.ok()) return candidate.status();
  auto publication = collector_->Commit(token_bytes_[candidate->candidate.token_id], *processes_); if (!publication.ok()) return publication.status();
  output_ = *publication; collector_.reset(); state_ = GenerationState::kOutputReady; return state_;
}
Result<TokenOutputLease> GenerationLoop::TakeOutput() {
  if (state_ != GenerationState::kOutputReady || !output_) return Status::FailedPrecondition("Generation has no pending output lease");
  const auto output = *output_;
  if (ledger_.finish() != TokenFinish::kNone) state_ = GenerationState::kFinishQueue;
  else {
    const auto next = ledger_.pending_input();
    if (!next || plan_ == std::numeric_limits<std::uint64_t>::max()) {
      // The accepted publication must still reach its consumer even when no
      // next plan can be issued. A subsequent Poll reports failed generation.
      Fail(); output_.reset(); return output;
    }
    request_.tokens.fill(0); request_.tokens[0] = *next; request_.count = 1; ++plan_; state_ = GenerationState::kPreparing;
  }
  output_.reset(); return output;
}
}  // namespace pih::deepseek_v41
