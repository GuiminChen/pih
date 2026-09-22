#include "supervisor_config.h"
#include "supervisor_deployment.h"
#include "supervisor_operation.h"
#include "supervisor_output.h"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <fcntl.h>
#include <limits>
#include <optional>
#include <stdexcept>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

namespace {
using namespace pih;
using namespace pih::deepseek_v41;
using Clock = SupervisorOperation::Clock;
volatile sig_atomic_t cancelled = 0;
void Signal(int) { cancelled = 1; }
void Require(const Status& s) { if (!s.ok()) throw std::runtime_error(std::string(s.message())); }
template<class T> T Take(Result<T> r) { Require(r.status()); return std::move(*r); }
void Report(const char* stage, const Status& s) {
  const auto text = s.message();
  std::fprintf(stderr, "native supervisor %s: %.*s\n", stage, static_cast<int>(text.size()), text.data());
}
void Yield() { const timespec delay{0, 1000000}; ::clock_nanosleep(CLOCK_MONOTONIC, 0, &delay, nullptr); }
Result<std::size_t> WriteStdout(void*, std::string_view bytes) {
  const auto count = ::write(STDOUT_FILENO, bytes.data(), bytes.size());
  if (count > 0) return static_cast<std::size_t>(count);
  if (count < 0 && (errno == EAGAIN || errno == EINTR)) return std::size_t{0};
  return Status::Unavailable("Supervisor stdout closed or failed");
}
struct Fd { int value = -1; ~Fd() { if (value >= 0) ::close(value); } };
struct OutputFlags {
  int original = -1;
  ~OutputFlags() { if (original >= 0) ::fcntl(STDOUT_FILENO, F_SETFL, original); }
};
int OpenParent(const std::string& path) {
  Fd root{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  if (root.value < 0) throw std::runtime_error("cannot open cgroup root");
  for (const auto& part : std::filesystem::path(path).relative_path()) {
    const auto name = part.string();
    Fd next{::openat(root.value, name.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    if (next.value < 0) throw std::runtime_error("cannot open delegated cgroup path without links");
    std::swap(root.value, next.value);
  }
  const auto fd = root.value; root.value = -1; return fd;
}
std::string Nonce() {
  std::array<unsigned char, 16> bytes{};
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto n = ::getrandom(bytes.data() + offset, bytes.size() - offset, GRND_NONBLOCK);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw std::runtime_error("kernel random endpoint identity unavailable");
    offset += static_cast<std::size_t>(n);
  }
  constexpr char hex[] = "0123456789abcdef";
  std::string text; text.reserve(32);
  for (auto b : bytes) { text += hex[b >> 4]; text += hex[b & 15]; }
  return text;
}
bool Terminal(SupervisorOperationState s) {
  return s == SupervisorOperationState::kComplete || s == SupervisorOperationState::kFailedRetired ||
      s == SupervisorOperationState::kFailedUnreconciled;
}
// Dedicated CLI owns every child it creates. Called only after operation polling
// has stopped, so waitpid cannot steal receipts from active pidfd retirement.
bool FinalCleanup(std::array<WorkerCgroup, 9>& groups, Clock::time_point deadline) {
  while (Clock::now() < deadline) {
    bool clean = true;
    for (auto& group : groups) {
      if (!group.created() || group.removed()) continue;
      auto empty = group.Empty();
      if (!empty.ok()) {
        // A missing population observation is not evidence that children are
        // gone. Kill independently revalidates the owned directory identity;
        // it will refuse a replaced/missing member rather than target a name.
        (void)group.Kill(); clean = false; continue;
      }
      if (!*empty) { (void)group.Kill(); clean = false; continue; }
      if (!group.Remove().ok()) clean = false;
    }
    // Includes orphaned descendants adopted by this dedicated subreaper.
    for (unsigned reaped = 0; reaped < 64; ++reaped) {
      const auto pid = ::waitpid(-1, nullptr, WNOHANG);
      if (pid > 0) { clean = false; continue; }
      if (pid == 0) clean = false;
      else if (errno == EINTR) continue;
      else if (errno != ECHILD) clean = false;
      break;
    }
    if (clean) return true;
    Yield();
  }
  return false;
}
int Run(SupervisorConfig config) {
  std::array<WorkerCgroup, 9> groups;
  std::array<WorkerCgroup*, 8> rank_groups{};
  std::unique_ptr<WorkerEnvironment> environment;
  std::unique_ptr<WorkerExecutable> worker, helper;
  std::unique_ptr<SupervisorDeployment> deployment;
  std::unique_ptr<SupervisorRequest> request;
  SupervisorOperation operation; // Destroy before every borrowed admission.
  OutputFlags output_flags;
  std::unique_ptr<SupervisorOutput> output;
  bool output_failed = false;
  int result = 2;
  try {
    if (::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)) throw std::runtime_error("cannot become dedicated child subreaper");
    struct sigaction stop{}; stop.sa_handler = Signal; ::sigemptyset(&stop.sa_mask);
    struct sigaction ignore{}; ignore.sa_handler = SIG_IGN; ::sigemptyset(&ignore.sa_mask);
    struct sigaction child{}; child.sa_handler = SIG_DFL; ::sigemptyset(&child.sa_mask);
    if (::sigaction(SIGINT, &stop, nullptr) || ::sigaction(SIGTERM, &stop, nullptr) ||
        ::sigaction(SIGPIPE, &ignore, nullptr) || ::sigaction(SIGCHLD, &child, nullptr))
      throw std::runtime_error("cannot configure supervisor signal handling");
    output_flags.original = ::fcntl(STDOUT_FILENO, F_GETFL);
    if (output_flags.original < 0 || ::fcntl(STDOUT_FILENO, F_SETFL, output_flags.original | O_NONBLOCK))
      throw std::runtime_error("stdout must support nonblocking output");
    const auto now = Clock::now();
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    const auto duration_ns = static_cast<std::uint64_t>(config.sequence_ms) * 1000000;
    if (now_ns <= 0 || duration_ns > static_cast<std::uint64_t>(INT64_MAX - now_ns))
      throw std::runtime_error("absolute sequence deadline overflow");
    const auto startup = now + std::chrono::milliseconds(config.startup_ms);
    const auto nonce = Nonce();
    environment = Take(WorkerEnvironment::Create(config.library_directories));
    worker = Take(WorkerExecutable::Open(config.worker.path, config.worker.sha256, config.worker.byte_budget, startup));
    helper = Take(WorkerExecutable::Open(config.helper.path, config.helper.sha256, config.helper.byte_budget, startup));
    deployment = Take(SupervisorDeployment::Create(config));
    request = Take(deployment->Admit(config.rendered_prompt, config.sampling, config.stopping));
    output = std::make_unique<SupervisorOutput>(request->output());
    Fd parent{OpenParent(config.delegated_cgroup)};
    // The configuration grants only fresh direct children under this parent.
    // Log their namespace so an unresolved cleanup can be inspected by name.
    std::fprintf(stderr, "native supervisor cgroup namespace: pih-v41-%s\n", nonce.c_str());
    Require(groups[8].Create(parent.value, "pih-v41-" + nonce + "-helper", config.helper_limits));
    for (unsigned rank = 0; rank < config.placements.size(); ++rank) {
      rank_groups[rank] = &groups[rank];
      Require(groups[rank].Create(parent.value, "pih-v41-" + nonce + "-rank-" + std::to_string(rank), config.rank_limits));
      auto& p = config.placements[rank];
      p.supervisor_pid = ::getpid(); p.supervisor_uid = ::geteuid();
      p.startup_ns = now_ns + static_cast<std::uint64_t>(config.startup_ms) * 1000000;
      p.sequence_ns = now_ns + duration_ns;
      for (unsigned channel = 0; channel < 4; ++channel)
        p.endpoints[channel] = "pih-v41-" + nonce + "-" + std::to_string(rank) + "-" + std::to_string(channel);
    }
    if (cancelled) throw std::runtime_error("startup cancelled before process launch");
    auto started = operation.Start(*worker, *helper, *environment, *request, config.placements,
        {rank_groups.data(), config.placements.size()}, groups[8], config.first_plan, std::chrono::milliseconds(config.grace_ms));
    if (!started.ok()) Report("startup", started);
    if (!started.ok() && operation.state() == SupervisorOperationState::kEmpty) throw std::runtime_error("startup rejected before process custody");
    // Terminal process state does not imply that all queued publications have
    // been consumed. Drain until the terminal queue has no further lease.
    for (;;) {
      if (cancelled) {
        cancelled = 0; output_failed = true;
        if (!Terminal(operation.state())) Require(operation.Cancel(Status::FailedPrecondition("supervisor received cancellation signal")));
      }
      if (!Terminal(operation.state())) (void)Take(operation.Poll());
      if (!output->pending() && (operation.state() == SupervisorOperationState::kOutputReady || Terminal(operation.state()))) {
        const bool promised = operation.state() == SupervisorOperationState::kOutputReady;
        auto lease = operation.TakeOutput();
        if (!lease.ok() && promised) Require(lease.status());
        if (lease.ok()) {
          Require(output->Adopt(*lease));
        }
        else if (Terminal(operation.state())) break;
      }
      if (output->pending()) {
        if (Clock::now() >= now + std::chrono::milliseconds(config.sequence_ms))
          output_failed = true;
        if (!output_failed) {
          const auto written = output->Write(nullptr, WriteStdout);
          if (!written.ok()) output_failed = true;
          // Output cannot outlive the admitted sequence deadline indefinitely.
          if (Clock::now() >= now + std::chrono::milliseconds(config.sequence_ms)) output_failed = true;
        }
        if (output_failed && !Terminal(operation.state()))
          Require(operation.Cancel(Status::Unavailable("supervisor output closed or exceeded deadline")));
        if (output_failed) Require(output->Discard());
      }
      if (!Terminal(operation.state()) || output->pending()) Yield();
    }
    if (!operation.failure().ok()) Report("operation", operation.failure());
    if (!operation.cleanup_failure().ok()) Report("cleanup", operation.cleanup_failure());
    if (output_failed) std::fputs("native supervisor output incomplete or cancelled\n", stderr);
    result = operation.state() == SupervisorOperationState::kComplete && !output_failed ? 0 :
        operation.state() == SupervisorOperationState::kFailedUnreconciled ? 3 : 2;
  } catch (...) {
    // Even a nonstandard exception must retain process custody and reach final
    // cgroup reconciliation; do not unwind borrowed owners directly to main.
    try { throw; }
    catch (const std::exception& e) {
      std::fprintf(stderr, "native supervisor failure: %s\n", e.what());
    }
    catch (...) { std::fputs("native supervisor failure: nonstandard exception\n", stderr); }
    // Retain owners while driving existing retirement; never unwind a running
    // group and then attempt to recreate its lost process custody.
    if (operation.state() != SupervisorOperationState::kEmpty && !Terminal(operation.state())) {
      try {
        Require(operation.Cancel(Status::Internal("supervisor continuation failed")));
        const auto deadline = Clock::now() + std::chrono::milliseconds(config.retirement_ms);
        while (!Terminal(operation.state()) && Clock::now() < deadline) { (void)Take(operation.Poll()); Yield(); }
      } catch (...) { result = 3; }
      if (!Terminal(operation.state()) || operation.state() == SupervisorOperationState::kFailedUnreconciled) result = 3;
    }
  }
  bool clean = false;
  try { clean = FinalCleanup(groups, Clock::now() + std::chrono::milliseconds(config.retirement_ms)); }
  catch (...) { clean = false; }
  if (!clean) {
    std::fputs("native supervisor cleanup unresolved; inspect the logged cgroup namespace\n", stderr); result = 3;
  }
  if (result == 0) {
    if (!request || !output || !output->Finish(request->ledger()).ok()) {
      std::fputs("native supervisor completion ledger is not terminal\n", stderr);
      return 2;
    }
    const char* finish = request->ledger().finish() == TokenFinish::kLength ? "length" : "stop";
    // A machine-readable final record on stderr preserves raw token bytes on
    // stdout. Emit only after successful operation, output and final cleanup.
    const auto written = std::fprintf(stderr,
        "{\"schema\":\"pih.deepseek-v41.supervisor-completion.v1\","
        "\"epoch\":%llu,\"sequence_generation\":%llu,\"world_size\":%zu,"
        "\"prompt_tokens\":%u,\"completion_tokens\":%zu,"
        "\"visible_bytes_written\":%llu,\"finish_reason\":\"%s\",\"cleanup\":\"reconciled\"}\n",
        static_cast<unsigned long long>(config.identity.epoch),
        static_cast<unsigned long long>(config.identity.sequence_generation), config.placements.size(),
        request->ledger().prompt_tokens(), request->ledger().records().size(),
        static_cast<unsigned long long>(output->visible_bytes()), finish);
    if (written < 0 || std::fflush(stderr) != 0) return 2;
  }
  return result;
}
}
int main(int argc, char** argv) {
  const bool check = argc == 5 && std::string_view(argv[1]) == "--check-config";
  const bool check_request = argc == 5 && std::string_view(argv[1]) == "--check-request";
  const int shift = (check || check_request) ? 1 : 0;
  if (argc != 4 + shift || std::string_view(argv[2 + shift]) != "--sha256") {
    std::fputs("usage: pih-v41-supervisor [--check-config|--check-request] /absolute/config.json --sha256 TRUSTED_CONFIG_SHA256\n", stderr); return 2;
  }
  try {
    const auto digest = Take(pih::Sha256Digest::ParseHex(argv[3 + shift]));
    auto config = Take(SupervisorConfig::Load(argv[1 + shift], digest));
    if (check) {
      std::fputs("configuration digest, schema and static request budgets accepted; artifacts, libraries, devices and runtime not checked\n", stderr);
      return 0;
    }
    if (check_request) {
      auto deployment = Take(SupervisorDeployment::Create(std::move(config)));
      const auto& sealed = deployment->configuration();
      auto request = Take(deployment->Admit(sealed.rendered_prompt, sealed.sampling, sealed.stopping));
      const auto written = std::fprintf(stdout,
          "{\"schema\":\"pih.deepseek-v41.request-admission.v1\",\"prompt_tokens\":%u,"
          "\"maximum_completion_tokens\":%u,\"maximum_positions\":%u,\"workers_started\":false}\n",
          request->ledger().prompt_tokens(), request->ledger().maximum_completion_tokens(), sealed.maximum_positions);
      return written >= 0 && std::fflush(stdout) == 0 ? 0 : 2;
    }
    return Run(std::move(config));
  } catch (const std::exception& e) { std::fprintf(stderr, "native supervisor admission: %s\n", e.what()); return 2; }
}
