#include "supervisor_execution.h"
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
using Clock = SupervisorOperation::Clock;
void Require(const Status& s) { if (!s.ok()) throw s; }
template<class T> T Take(Result<T> value) { Require(value.status()); return std::move(*value); }
void Yield() { const timespec delay{0, 1000000}; ::clock_nanosleep(CLOCK_MONOTONIC, 0, &delay, nullptr); }
bool Terminal(SupervisorOperationState s) {
  return s == SupervisorOperationState::kComplete || s == SupervisorOperationState::kFailedRetired ||
      s == SupervisorOperationState::kFailedUnreconciled;
}
struct Fd { int value{-1}; ~Fd() { if (value >= 0) ::close(value); } };
int OpenParent(const std::string& path) {
  Fd root{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  if (root.value < 0) throw Status::Unavailable("Cannot open cgroup root");
  for (const auto& part : std::filesystem::path(path).relative_path()) {
    Fd next{::openat(root.value, part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    if (next.value < 0) throw Status::Unavailable("Cannot open delegated cgroup without links");
    std::swap(root.value, next.value);
  }
  const auto fd = root.value; root.value = -1; return fd;
}
std::string Nonce() {
  std::array<unsigned char,16> bytes{}; size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = ::getrandom(bytes.data()+offset, bytes.size()-offset, GRND_NONBLOCK);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) throw Status::Unavailable("Kernel endpoint nonce unavailable");
    offset += static_cast<size_t>(count);
  }
  constexpr char hex[] = "0123456789abcdef"; std::string value; value.reserve(32);
  for (auto b : bytes) { value += hex[b >> 4]; value += hex[b & 15]; } return value;
}
}
Status AdmitSupervisorProcess() {
  try {
    size_t threads = 0;
    for (const auto& unused : std::filesystem::directory_iterator("/proc/self/task")) ++threads;
    if (threads != 1) return Status::FailedPrecondition("V4.1 controller requires a dedicated single-threaded process");
    struct sigaction action{};
    if (::sigaction(SIGCHLD, nullptr, &action) || action.sa_handler != SIG_DFL || (action.sa_flags & SA_NOCLDWAIT))
      return Status::FailedPrecondition("V4.1 controller requires default SIGCHLD reaping");
    siginfo_t info{};
    if (::waitid(P_ALL, 0, &info, WEXITED | WNOHANG | WNOWAIT) == 0 || errno != ECHILD)
      return Status::FailedPrecondition("V4.1 controller process already has children");
    return Status::Ok();
  } catch (...) { return Status::Unavailable("Cannot inspect dedicated supervisor process"); }
}
SupervisorExecution::SupervisorExecution(std::unique_ptr<SupervisorRequest> request)
    : request_(std::move(request)), output_(request_->output()) {}
bool SupervisorExecution::Cleanup(std::chrono::milliseconds timeout) noexcept {
  try {
    const auto deadline = Clock::now() + timeout;
    while (Clock::now() < deadline) {
      bool clean = true;
      for (auto& group : groups_) {
        if (!group.created() || group.removed()) continue;
        auto empty = group.Empty();
        if (!empty.ok() || !*empty) { (void)group.Kill(); clean = false; continue; }
        if (!group.Remove().ok()) clean = false;
      }
      // Safe only under AdmitSupervisorProcess: all children and adopted
      // descendants belong to this execution, and operation polling has ended.
      for (unsigned i = 0; i < 64; ++i) {
        const auto pid = ::waitpid(-1, nullptr, WNOHANG);
        if (pid > 0) { clean = false; continue; }
        if (pid == 0) clean = false;
        else if (errno == EINTR) continue;
        else if (errno != ECHILD) clean = false;
        break;
      }
      if (clean) { retired_ = true; return true; }
      Yield();
    }
  } catch (...) {}
  return false;
}
Result<SupervisorCompletion> SupervisorExecution::Run(const SupervisorConfig& config,
    void* context, SupervisorOutput::Writer writer, Cancelled cancelled) noexcept {
  Status failure = Status::Internal("V4.1 supervisor execution failed");
  try { return RunImpl(config, context, writer, cancelled); }
  catch (const Status& s) { failure = s; }
  catch (const std::bad_alloc&) { failure = Status::ResourceExhausted("V4.1 supervisor allocation failed"); }
  catch (...) {}
  try {
    if (!Terminal(operation_.state()) && operation_.state() != SupervisorOperationState::kEmpty) {
      (void)operation_.Cancel(failure);
      const auto deadline = Clock::now() + std::chrono::milliseconds(config.retirement_ms);
      while (!Terminal(operation_.state()) && Clock::now() < deadline) {
        const auto poll = operation_.Poll(); if (!poll.ok()) break; Yield();
      }
    }
    if (output_.pending()) (void)output_.Discard();
  } catch (...) {}
  if (!retired_ && !Cleanup(std::chrono::milliseconds(config.retirement_ms)))
    return Status::Internal("V4.1 retirement unresolved; retain controller owners and restart required");
  return failure;
}
Result<SupervisorCompletion> SupervisorExecution::RunImpl(const SupervisorConfig& config,
    void* context, SupervisorOutput::Writer writer, Cancelled cancelled) {
  if (!writer || !cancelled || operation_.state() != SupervisorOperationState::kEmpty)
    throw Status::InvalidArgument("V4.1 supervisor callbacks or state invalid");
  Require(AdmitSupervisorProcess());
  if (::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0))
    throw Status::Unavailable("Cannot enable supervisor child subreaper");
  const auto now = Clock::now();
  const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
  const auto duration = static_cast<uint64_t>(config.sequence_ms) * 1000000;
  if (now_ns <= 0 || duration > static_cast<uint64_t>(INT64_MAX - now_ns))
    throw Status::InvalidArgument("Supervisor absolute deadline overflow");
  const auto deadline = now + std::chrono::milliseconds(config.sequence_ms);
  const auto startup = now + std::chrono::milliseconds(config.startup_ms);
  environment_ = Take(WorkerEnvironment::Create(config.library_directories));
  worker_ = Take(WorkerExecutable::Open(config.worker.path, config.worker.sha256, config.worker.byte_budget, startup));
  helper_ = Take(WorkerExecutable::Open(config.helper.path, config.helper.sha256, config.helper.byte_budget, startup));
  auto placements = config.placements;
  const auto nonce = Nonce();
  Fd parent{OpenParent(config.delegated_cgroup)};
  std::array<WorkerCgroup*,8> ranks{};
  std::fprintf(stderr, "native V4.1 cgroup namespace: pih-v41-%s\n", nonce.c_str());
  retired_ = false;
  Require(groups_[8].Create(parent.value, "pih-v41-" + nonce + "-helper", config.helper_limits));
  for (size_t rank = 0; rank < placements.size(); ++rank) {
    ranks[rank] = &groups_[rank];
    Require(groups_[rank].Create(parent.value, "pih-v41-" + nonce + "-rank-" + std::to_string(rank), config.rank_limits));
    auto& p = placements[rank]; p.supervisor_pid = ::getpid(); p.supervisor_uid = ::geteuid();
    p.startup_ns = now_ns + static_cast<uint64_t>(config.startup_ms) * 1000000;
    p.sequence_ns = now_ns + duration;
    for (unsigned channel = 0; channel < 4; ++channel)
      p.endpoints[channel] = "pih-v41-" + nonce + "-" + std::to_string(rank) + "-" + std::to_string(channel);
  }
  if (cancelled(context)) throw Status::Unavailable("V4.1 request cancelled before launch");
  const auto started = operation_.Start(*worker_, *helper_, *environment_, *request_, placements,
      {ranks.data(), placements.size()}, groups_[8], config.first_plan, std::chrono::milliseconds(config.grace_ms));
  if (!started.ok() && operation_.state() == SupervisorOperationState::kEmpty) throw started;
  Status output_failure = Status::Ok();
  for (;;) {
    if (output_failure.ok() && (cancelled(context) || Clock::now() >= deadline))
      output_failure = Clock::now() >= deadline ? Status::DeadlineExceeded("V4.1 generation deadline elapsed")
          : Status::Unavailable("V4.1 client cancelled");
    if (!output_failure.ok() && !Terminal(operation_.state())) Require(operation_.Cancel(output_failure));
    if (!Terminal(operation_.state())) (void)Take(operation_.Poll());
    if (!output_.pending() && (operation_.state() == SupervisorOperationState::kOutputReady || Terminal(operation_.state()))) {
      const bool promised = operation_.state() == SupervisorOperationState::kOutputReady;
      auto lease = operation_.TakeOutput();
      if (!lease.ok() && promised) throw lease.status();
      if (lease.ok()) Require(output_.Adopt(*lease));
      else if (Terminal(operation_.state())) break;
    }
    if (output_.pending()) {
      if (output_failure.ok() && Clock::now() >= deadline)
        output_failure = Status::DeadlineExceeded("V4.1 output deadline elapsed");
      if (output_failure.ok()) {
        auto sent = output_.Write(context, writer);
        if (!sent.ok()) output_failure = sent.status();
        if (Clock::now() >= deadline) output_failure = Status::DeadlineExceeded("V4.1 output deadline elapsed");
      }
      if (!output_failure.ok()) Require(output_.Discard());
    }
    if (!Terminal(operation_.state()) || output_.pending()) Yield();
  }
  if (!Cleanup(std::chrono::milliseconds(config.retirement_ms)))
    throw Status::Internal("V4.1 process retirement unresolved");
  Require(output_failure);
  if (operation_.state() != SupervisorOperationState::kComplete)
    throw operation_.failure().ok() ? Status::Internal("V4.1 generation did not complete") : operation_.failure();
  Require(output_.Finish(request_->ledger()));
  const auto& ledger = request_->ledger();
  return SupervisorCompletion{ledger.identity().sequence_generation, ledger.prompt_tokens(),
      ledger.records().size(), output_.visible_bytes(), ledger.finish()};
}
}
