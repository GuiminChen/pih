#include "worker_runtime.h"
#include <new>

namespace pih::deepseek_v41 {
Status WorkerRuntime::Start(std::span<const std::byte> id, std::uint32_t major,
    std::uint32_t minor, WorkerMemoryBudget budget, SamplingIdentity identity,
    SamplingParameters sampling, Clock::time_point startup, Clock::time_point execution,
    std::chrono::milliseconds retirement_timeout) {
  if (state_ != WorkerRuntimeState::kEmpty) return Status::FailedPrecondition("Worker runtime is single-use");
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
  state_ = WorkerRuntimeState::kFailed;
  auto status = handles_.Create(major, minor); if (!status.ok()) return status;
  status = communicator_.Start(id, requests_.world(), requests_.rank(), handles_.device_ordinal(), startup);
  if (!status.ok()) return status;
  state_ = WorkerRuntimeState::kConnecting;
  return Status::Ok();
}
void WorkerRuntime::Fail() noexcept {
  state_ = WorkerRuntimeState::kFailed;
  if (loop_) loop_->Fail();
}
Result<WorkerRuntimeState> WorkerRuntime::Poll() {
  try {
    auto result = PollImpl();
    if (!result.ok()) Fail();
    return result;
  } catch (const std::bad_alloc&) {
    Fail(); return Status::ResourceExhausted("Worker runtime continuation allocation failed");
  }
}
Result<WorkerRuntimeState> WorkerRuntime::PollImpl() {
  if (state_ == WorkerRuntimeState::kReleased) return state_;
  if (state_ == WorkerRuntimeState::kEmpty || state_ == WorkerRuntimeState::kFailed)
    return Status::FailedPrecondition("Worker runtime is not active");
  if (state_ == WorkerRuntimeState::kAwaitingRetirement) {
    const auto authorized = commands_.PollReceive(RankLifecycleKind::kRetire, identity_, sequence_deadline_);
    if (!authorized.ok()) return authorized.status();
    if (!*authorized) return state_;
    const auto status = BeginRetirement(Clock::now() + retirement_timeout_);
    if (!status.ok()) return status;
    return state_;
  }
  if (state_ != WorkerRuntimeState::kRunning && Clock::now() >= deadline_)
    return Status::DeadlineExceeded("Worker startup or retirement deadline expired");
  if (state_ == WorkerRuntimeState::kConnecting) {
    const auto ready = communicator_.Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    const auto started = memory_.Start(budget_, deadline_); if (!started.ok()) return started;
    state_ = WorkerRuntimeState::kLoading;
    return state_;
  }
  if (state_ == WorkerRuntimeState::kLoading) {
    const auto ready = memory_.Advance(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    auto views = memory_.Views(); if (!views.ok()) return views.status();
    const auto comm = communicator_.Borrow(); if (!comm.ok()) return comm.status();
    RankWorkerResources resources{sequence_, hashes_, views->inference, views->weights,
        views->prefill, views->decode, requests_, receipts_, *comm, handles_.ledger().events[0]};
    auto loop = RankWorkerLoop::Create(config_, resources, identity_, sampling_,
        plan_.boundary().token_capacity(), sequence_deadline_);
    if (!loop.ok()) return loop.status();
    loop_ = std::move(*loop);
    const auto queued = notices_.Queue(RankLifecycleKind::kReady, identity_); if (!queued.ok()) return queued;
    state_ = WorkerRuntimeState::kPublishingReady;
    return state_;
  }
  if (state_ == WorkerRuntimeState::kPublishingReady || state_ == WorkerRuntimeState::kPublishingReleased) {
    const auto sent = notices_.PollSend(deadline_); if (!sent.ok()) return sent.status();
    if (*sent) state_ = state_ == WorkerRuntimeState::kPublishingReady
        ? WorkerRuntimeState::kRunning : WorkerRuntimeState::kReleased;
    return state_;
  }
  if (state_ == WorkerRuntimeState::kRunning) {
    auto result = loop_->Poll(); if (!result.ok()) return result.status();
    if (*result == RankWorkerState::kComplete) state_ = WorkerRuntimeState::kAwaitingRetirement;
    return state_;
  }
  if (state_ == WorkerRuntimeState::kFinalizing) {
    auto ready = communicator_.Poll(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    auto status = communicator_.Release(); if (!status.ok()) return status;
    status = memory_.BeginRetirement(deadline_); if (!status.ok()) return status;
    state_ = WorkerRuntimeState::kReleasingMemory;
    return state_;
  }
  if (state_ == WorkerRuntimeState::kReleasingMemory) {
    auto ready = memory_.Advance(); if (!ready.ok()) return ready.status();
    if (!*ready) return state_;
    const auto status = handles_.BeginRetirement(); if (!status.ok()) return status;
    state_ = WorkerRuntimeState::kReleasingHandles;
    return state_;
  }
  auto ready = handles_.PollRetirement(); if (!ready.ok()) return ready.status();
  if (!*ready) return state_;
  const auto status = handles_.Release(); if (!status.ok()) return status;
  const auto queued = notices_.Queue(RankLifecycleKind::kReleased, identity_); if (!queued.ok()) return queued;
  state_ = WorkerRuntimeState::kPublishingReleased;
  return state_;
}
Status WorkerRuntime::BeginRetirement(Clock::time_point deadline) {
  if (state_ != WorkerRuntimeState::kAwaitingRetirement || deadline <= Clock::now())
    return Status::FailedPrecondition("Worker requires global terminal retirement authorization");
  // Drop the completed loop before invalidating any borrowed memory or NCCL
  // handle. Its destructor does not poison a terminally completed sequence.
  loop_.reset();
  const auto status = communicator_.BeginFinalize(deadline);
  if (!status.ok()) { Fail(); return status; }
  deadline_ = deadline;
  state_ = WorkerRuntimeState::kFinalizing;
  return Status::Ok();
}
Status WorkerRuntime::AbortCommunicator() {
  if (state_ != WorkerRuntimeState::kFailed)
    return Status::FailedPrecondition("Communicator abort is only allowed after worker failure");
  return communicator_.Abort();
}
}  // namespace pih::deepseek_v41
