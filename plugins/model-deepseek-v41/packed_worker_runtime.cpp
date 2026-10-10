#include "packed_worker_runtime.h"
#include <new>

namespace pih::deepseek_v41 {
Status PackedWorkerRuntime::Start(std::span<const std::byte> id, std::uint32_t major,
    std::uint32_t minor, PackedWorkerMemoryBudget budget, SamplingIdentity identity,
    SamplingParameters sampling, Clock::time_point startup, Clock::time_point execution,
    std::chrono::milliseconds retirement_timeout) {
  if (state_ != PackedWorkerRuntimeState::kEmpty) return Status::FailedPrecondition("Worker runtime is single-use");
  if (major != 10 || minor != 3)
    return Status::FailedPrecondition("V4.1 worker runtime requires native SM103");
  const auto valid = ValidateSamplingParameters(sampling); if (!valid.ok()) return valid;
  if (id.size() != WorkerCommunicator::BootstrapId{}.size() || !identity.epoch || identity.plan_seq ||
      !identity.sequence_generation || !identity.sampling_config_id || sampling.ordinal || sampling.suppressed_count ||
      hashes_.position() || startup <= Clock::now() || execution <= startup ||
      retirement_timeout.count() <= 0 || retirement_timeout > std::chrono::minutes(5) ||
      !commands_.receiver_ready() || !notices_.sender_ready() ||
      commands_.rank() != requests_.rank() || notices_.rank() != requests_.rank() ||
      commands_.world() != requests_.world() || notices_.world() != requests_.world() ||
      commands_.peer_pid() != requests_.peer_pid() || notices_.peer_pid() != requests_.peer_pid() ||
      !requests_.receiver_ready() || !receipts_.sender_ready() || requests_.rank() != receipts_.rank() ||
      requests_.world() != receipts_.world() || requests_.peer_pid() != receipts_.peer_pid() ||
      requests_.rank() != plan_.boundary().rank() || requests_.world() != plan_.boundary().world_size() ||
      plan_.boundary().config_sha256() != config_.config_sha256())
    return Status::InvalidArgument("Worker bootstrap identity, channels, configuration or deadlines invalid");
  budget_ = budget; identity_ = identity; sampling_ = sampling;
  deadline_ = startup; sequence_deadline_ = execution;
  retirement_timeout_ = retirement_timeout;
  state_ = PackedWorkerRuntimeState::kFailed;
  auto status = handles_.Create(major, minor); if (!status.ok()) return status;
  status = communicator_.Start(id, requests_.world(), requests_.rank(), handles_.device_ordinal(), startup);
  if (!status.ok()) return status;
  state_ = PackedWorkerRuntimeState::kConnecting;
  return Status::Ok();
}
void PackedWorkerRuntime::Fail() noexcept {
  state_ = PackedWorkerRuntimeState::kFailed;
  if (loop_) loop_->Fail();
}
Result<PackedWorkerRuntimeState> PackedWorkerRuntime::Poll() {
  try {
    auto result = PollImpl();
    if (!result.ok()) Fail();
    return result;
  } catch (const std::bad_alloc&) {
    Fail(); return Status::ResourceExhausted("Worker runtime continuation allocation failed");
  }
}
Result<PackedWorkerRuntimeState> PackedWorkerRuntime::PollImpl() {
  if (state_ == PackedWorkerRuntimeState::kReleased) return state_;
  if (state_ == PackedWorkerRuntimeState::kEmpty || state_ == PackedWorkerRuntimeState::kFailed)
    return Status::FailedPrecondition("Worker runtime is not active");
  if (state_ == PackedWorkerRuntimeState::kAwaitingRetirement) {
    const auto authorized = commands_.PollReceive(RankLifecycleKind::kRetire, identity_, sequence_deadline_);
    if (!authorized.ok()) return authorized.status();
    if (!*authorized) return state_;
    const auto status = BeginRetirement(Clock::now() + retirement_timeout_);
    if (!status.ok()) return status;
    return state_;
  }
  if (state_ != PackedWorkerRuntimeState::kRunning && Clock::now() >= deadline_)
    return Status::DeadlineExceeded("Worker startup or retirement deadline expired");
  if (state_ == PackedWorkerRuntimeState::kConnecting) {
    const auto ready = communicator_.Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    const auto started = memory_.Start(budget_, deadline_); if (!started.ok()) return started;
    state_ = PackedWorkerRuntimeState::kLoading;
    return state_;
  }
  if (state_ == PackedWorkerRuntimeState::kLoading) {
    const auto ready = memory_.Advance(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    auto views = memory_.Views(); if (!views.ok()) return views.status();
    const auto comm = communicator_.Borrow(); if (!comm.ok()) return comm.status();
    PackedRankWorkerResources resources{views->inference, views->weights,
        views->prefill, views->decode, requests_, receipts_, *comm, handles_.ledger().events[0]};
    auto loop = PackedRankWorkerLoop::Create(config_, hashes_, resources, identity_.epoch,
        plan_.maximum_sequences(), plan_.maximum_tokens(), sequence_deadline_);
    if (!loop.ok()) return loop.status();
    loop_ = std::move(*loop);
    const auto queued = notices_.Queue(RankLifecycleKind::kReady, identity_); if (!queued.ok()) return queued;
    state_ = PackedWorkerRuntimeState::kPublishingReady;
    return state_;
  }
  if (state_ == PackedWorkerRuntimeState::kPublishingReady || state_ == PackedWorkerRuntimeState::kPublishingReleased) {
    const auto sent = notices_.PollSend(deadline_); if (!sent.ok()) return sent.status();
    if (*sent) state_ = state_ == PackedWorkerRuntimeState::kPublishingReady
        ? PackedWorkerRuntimeState::kRunning : PackedWorkerRuntimeState::kReleased;
    return state_;
  }
  if (state_ == PackedWorkerRuntimeState::kRunning) {
    if (loop_->idle()) {
      const auto authorized = commands_.PollReceive(RankLifecycleKind::kRetire, identity_, sequence_deadline_);
      if (!authorized.ok()) return authorized.status();
      if (*authorized) {
        state_ = PackedWorkerRuntimeState::kAwaitingRetirement;
        const auto retired = BeginRetirement(Clock::now() + retirement_timeout_);
        if (!retired.ok()) return retired; return state_;
      }
    }
    auto result = loop_->Poll(); if (!result.ok()) return result.status();
    if (*result == RankWorkerState::kComplete) state_ = PackedWorkerRuntimeState::kAwaitingRetirement;
    return state_;
  }
  if (state_ == PackedWorkerRuntimeState::kFinalizing) {
    auto ready = communicator_.Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    auto status = communicator_.Release(); if (!status.ok()) return status;
    status = memory_.BeginRetirement(deadline_); if (!status.ok()) return status;
    state_ = PackedWorkerRuntimeState::kReleasingMemory;
    return state_;
  }
  if (state_ == PackedWorkerRuntimeState::kReleasingMemory) {
    auto ready = memory_.Advance(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    const auto status = handles_.BeginRetirement(); if (!status.ok()) return status;
    state_ = PackedWorkerRuntimeState::kReleasingHandles;
    return state_;
  }
  auto ready = handles_.PollRetirement(); if (!ready.ok()) return ready.status();
  if (!*ready) return state_;
  const auto status = handles_.Release(); if (!status.ok()) return status;
  const auto queued = notices_.Queue(RankLifecycleKind::kReleased, identity_); if (!queued.ok()) return queued;
  state_ = PackedWorkerRuntimeState::kPublishingReleased;
  return state_;
}
Status PackedWorkerRuntime::BeginRetirement(Clock::time_point deadline) {
  if (state_ != PackedWorkerRuntimeState::kAwaitingRetirement || deadline <= Clock::now())
    return Status::FailedPrecondition("Worker requires global terminal retirement authorization");
  // Drop the completed loop before invalidating any borrowed memory or NCCL
  // handle. Its destructor does not poison a terminally completed sequence.
  loop_.reset();
  const auto status = communicator_.BeginFinalize(deadline);
  if (!status.ok()) { Fail(); return status; }
  deadline_ = deadline;
  state_ = PackedWorkerRuntimeState::kFinalizing;
  return Status::Ok();
}
Status PackedWorkerRuntime::AbortCommunicator() {
  if (state_ != PackedWorkerRuntimeState::kFailed)
    return Status::FailedPrecondition("Communicator abort is only allowed after worker failure");
  return communicator_.Abort();
}
}  // namespace pih::deepseek_v41
