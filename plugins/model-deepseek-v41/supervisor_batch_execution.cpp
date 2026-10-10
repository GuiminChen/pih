#include "supervisor_batch_execution.h"
#include "worker_group.h"
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
using Clock = std::chrono::steady_clock;
using batch_ipc::kMagic; using batch_ipc::kMaximumFrame;
enum Kind : std::uint32_t { kAdmit = 1, kCancel, kShutdown, kReady, kOutput, kComplete, kRetired, kAdmitted, kBegin };
using batch_ipc::Header; using batch_ipc::Frame;
void Require(Status status) { if (!status.ok()) throw status; }
template<class T> T Take(Result<T> result) { Require(result.status()); return std::move(*result); }
void Yield() { ::poll(nullptr, 0, 1); }
Status DecodeStatus(std::uint32_t code, std::string text) {
  switch (static_cast<StatusCode>(code)) {
    case StatusCode::kOk: return Status::Ok();
    case StatusCode::kInvalidArgument: return Status::InvalidArgument(std::move(text));
    case StatusCode::kResourceExhausted: return Status::ResourceExhausted(std::move(text));
    case StatusCode::kFailedPrecondition: return Status::FailedPrecondition(std::move(text));
    case StatusCode::kUnavailable: return Status::Unavailable(std::move(text));
    case StatusCode::kDeadlineExceeded: return Status::DeadlineExceeded(std::move(text));
    default: return Status::Internal(std::move(text));
  }
}
template<class T> void Append(std::string& bytes, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  bytes.append(reinterpret_cast<const char*>(&value), sizeof(value));
}
template<class T> T Read(std::string_view& bytes) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (bytes.size() < sizeof(T)) throw Status::InvalidArgument("V4.1 IPC truncated field");
  T value; std::memcpy(&value, bytes.data(), sizeof(value)); bytes.remove_prefix(sizeof(value)); return value;
}
void AppendString(std::string& out, std::string_view value) {
  const auto size = static_cast<std::uint32_t>(value.size()); Append(out, size); out.append(value);
}
std::string ReadString(std::string_view& in) {
  auto size = Read<std::uint32_t>(in);
  if (size > in.size()) throw Status::InvalidArgument("V4.1 IPC invalid string extent");
  std::string out(in.substr(0, size)); in.remove_prefix(size); return out;
}
std::string RequestBytes(std::string_view prompt, const SamplingParameters& sampling, const TokenStopConfig& stop, Clock::time_point deadline) {
  if (prompt.size() > (1U << 20) || stop.pattern_count > 16 || stop.token_count > 17)
    throw Status::InvalidArgument("V4.1 IPC request bounds invalid");
  const auto deadline_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(deadline.time_since_epoch()).count();
  if (deadline_ns <= 0) throw Status::InvalidArgument("V4.1 absolute request deadline invalid");
  std::string bytes; Append(bytes, static_cast<std::uint64_t>(deadline_ns));
  Append(bytes, sampling); Append(bytes, stop.minimum); Append(bytes, stop.maximum);
  Append(bytes, stop.token_count); for (auto id : stop.tokens) Append(bytes, id);
  Append(bytes, stop.pattern_count); for (std::uint32_t i = 0; i < stop.pattern_count; ++i) AppendString(bytes, stop.patterns[i]);
  AppendString(bytes, prompt);
  if (bytes.size() > kMaximumFrame) throw Status::ResourceExhausted("V4.1 IPC request frame too large");
  return bytes;
}
// One reader and writer per endpoint; partial stream I/O is bounded and never
// invokes model or user callbacks. Protocol is private to the forked binary.
class Socket final {
 public:
  explicit Socket(int fd) : fd_(fd) {}
  ~Socket() { if (fd_ >= 0) ::close(fd_); }
  void Enqueue(Kind kind, std::uint64_t id, std::string body = {}, Status status = Status::Ok()) {
    if (body.size() > kMaximumFrame || queued_ > (32U << 20) - sizeof(Header) - body.size())
      throw Status::ResourceExhausted("V4.1 IPC output backpressure limit");
    Header header{kMagic, kind, id, static_cast<std::uint32_t>(body.size()), static_cast<std::uint32_t>(status.code())};
    std::string frame; Append(frame, header); frame += body; queued_ += frame.size(); tx_.push_back(std::move(frame));
  }
  void StatusFrame(Kind kind, std::uint64_t id, Status status) {
    Enqueue(kind, id, std::string(status.message()), status);
  }
  void Flush() {
    while (!tx_.empty()) {
      const auto& frame = tx_.front();
      auto count = ::send(fd_, frame.data() + sent_, frame.size() - sent_, MSG_NOSIGNAL | MSG_DONTWAIT);
      if (count < 0 && errno == EINTR) continue;
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return;
      if (count <= 0) throw Status::Unavailable("V4.1 IPC peer disconnected");
      sent_ += count;
      if (sent_ == frame.size()) { queued_ -= frame.size(); tx_.pop_front(); sent_ = 0; }
    }
  }
  std::optional<Frame> Receive() {
    for (;;) {
      if (auto frame = Take(decoder_.Take())) return frame;
      char bytes[16384]; auto count = ::recv(fd_, bytes, sizeof(bytes), MSG_DONTWAIT);
      if (count < 0 && errno == EINTR) continue;
      if (count < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) return {};
      if (count <= 0) throw Status::Unavailable("V4.1 IPC peer disconnected");
      Require(decoder_.Feed({bytes, static_cast<std::size_t>(count)}));
    }
  }
  bool empty() const { return tx_.empty(); }
  bool request_credit(std::size_t bytes) const {
    // Reserve control credit so queued prompt traffic cannot starve cancellation
    // and shutdown messages. Unsent requests stay in the bounded parent arena.
    return bytes <= kMaximumFrame && queued_ <= (16U << 20) - sizeof(Header) - bytes;
  }
  bool model_credit(std::uint32_t sequences, std::uint32_t pending) const {
    const auto reserve = static_cast<std::size_t>(sequences) * (8448 + sizeof(Header)) +
        static_cast<std::size_t>(sequences + pending) * (Status::kMaxMessageBytes + sizeof(Header) + sizeof(SupervisorCompletion)) +
        Status::kMaxMessageBytes + sizeof(Header);
    return reserve < (32U << 20) && queued_ <= (32U << 20) - reserve;
  }
 private:
  int fd_; std::deque<std::string> tx_; std::size_t sent_ = 0, queued_ = 0; batch_ipc::Decoder decoder_;
};
struct Fd { int fd = -1; ~Fd() { if (fd >= 0) ::close(fd); } };
int OpenParent(const std::string& path) {
  Fd root{::open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)};
  if (root.fd < 0) throw Status::Unavailable("V4.1 cgroup root unavailable");
  for (const auto& part : std::filesystem::path(path).relative_path()) {
    Fd next{::openat(root.fd, part.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)};
    if (next.fd < 0) throw Status::Unavailable("V4.1 delegated cgroup unavailable"); std::swap(root.fd, next.fd);
  }
  const auto fd = root.fd; root.fd = -1; return fd;
}
std::string Nonce() {
  std::array<unsigned char, 16> bytes{}; std::size_t offset = 0;
  while (offset < bytes.size()) {
    auto n = ::getrandom(bytes.data() + offset, bytes.size() - offset, GRND_NONBLOCK);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) throw Status::Unavailable("V4.1 endpoint nonce unavailable"); offset += n;
  }
  constexpr char hex[] = "0123456789abcdef"; std::string out;
  for (auto b : bytes) { out += hex[b >> 4]; out += hex[b & 15]; } return out;
}
struct ChildOwners {
  std::array<WorkerCgroup, 9> cgroups;
  std::unique_ptr<WorkerEnvironment> environment;
  std::unique_ptr<WorkerExecutable> worker, helper;
  NcclBootstrapBroker broker;
  WorkerGroup group;
  std::unique_ptr<PackedGenerationSession> session;
  struct Request { std::unique_ptr<SupervisorRequest> request; bool cancelled = false; std::uint64_t visible = 0; bool activated = false; Clock::time_point deadline{}; };
  std::map<std::uint64_t, Request> requests;
  std::map<SupervisorRequest*, std::uint64_t> identities;
  bool Cleanup(std::uint32_t timeout) noexcept {
    try {
      const auto deadline = Clock::now() + std::chrono::milliseconds(timeout);
      if (session) session->Fail();
      if (group.state() == WorkerGroupState::kRetiring) (void)group.ResumeStartupRetirement(deadline);
      if (broker.state() == NcclBrokerState::kRetiring) (void)broker.ResumeRetirement(deadline);
      if (group.has_startup_custody() && group.state() != WorkerGroupState::kRetired && group.state() != WorkerGroupState::kRetiring)
        (void)group.BeginStartupRetirement(deadline);
      while (Clock::now() < deadline) {
        if (group.state() == WorkerGroupState::kRetiring) (void)group.PollStartupRetirement();
        if (broker.state() != NcclBrokerState::kEmpty && broker.state() != NcclBrokerState::kRetired) {
          if (broker.state() != NcclBrokerState::kRetiring) (void)broker.BeginRetirement(deadline);
          if (broker.state() == NcclBrokerState::kRetiring) (void)broker.PollRetirement();
        }
        bool clean = (!group.has_startup_custody() || group.state() == WorkerGroupState::kRetired) &&
            (broker.state() == NcclBrokerState::kEmpty || broker.state() == NcclBrokerState::kRetired);
        for (auto& cgroup : cgroups) {
          if (!cgroup.created() || cgroup.removed()) continue;
          auto empty = cgroup.Empty();
          if (!empty.ok() || !*empty) { (void)cgroup.Kill(); clean = false; }
          else if (!cgroup.Remove().ok()) clean = false;
        }
        // This dedicated subreaper owns every child; no service caller can fork
        // or consume wait statuses here. cgroup.kill contains all descendants.
        for (unsigned i = 0; clean && i < 64; ++i) {
          auto pid = ::waitpid(-1, nullptr, WNOHANG);
          if (pid > 0) { clean = false; continue; }
          if (!pid || (pid < 0 && errno != ECHILD)) clean = false; break;
        }
        if (clean) {
          if (session) while (auto output = Take(session->TakeOutput())) Require(output->request->output().Release(output->lease));
          for (auto& [_, request] : requests) {
            if (auto lease = request.request->ledger().pending_output())
              Require(request.request->ledger().AbortRetired(lease->credit.plan_sequence));
          }
          session.reset(); requests.clear(); identities.clear(); return true;
        }
        Yield();
      }
    } catch (...) {}
    return false;
  }
  void Start(SupervisorDeployment& deployment) {
    const auto& config = deployment.configuration();
    if (::prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)) throw Status::Unavailable("V4.1 batch subreaper admission failed");
    auto now = Clock::now(), startup = now + std::chrono::milliseconds(config.startup_ms);
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    environment = Take(WorkerEnvironment::Create(config.library_directories));
    worker = Take(WorkerExecutable::Open(config.worker.path, config.worker.sha256, config.worker.byte_budget, startup));
    helper = Take(WorkerExecutable::Open(config.helper.path, config.helper.sha256, config.helper.byte_budget, startup));
    auto seed = Take(deployment.Admit(config.rendered_prompt, config.sampling, config.stopping));
    auto placements = config.placements; const auto nonce = Nonce();
    Fd parent{OpenParent(config.delegated_cgroup)};
    Require(cgroups[8].Create(parent.fd, "pih-v41-" + nonce + "-helper", config.helper_limits));
    std::array<WorkerCgroup*, 8> ranks{};
    for (std::size_t rank = 0; rank < placements.size(); ++rank) {
      ranks[rank] = &cgroups[rank];
      Require(cgroups[rank].Create(parent.fd, "pih-v41-" + nonce + "-rank-" + std::to_string(rank), config.rank_limits));
      auto p = placements[rank];
      p.supervisor_pid = ::getpid(); p.supervisor_uid = ::geteuid();
      p.startup_ns = ns + static_cast<std::uint64_t>(config.startup_ms) * 1000000; p.sequence_ns = INT64_MAX;
      p.packed_requests = true; p.maximum_batch_sequences = config.maximum_batch_sequences;
      p.maximum_batch_tokens = config.maximum_batch_tokens; p.maximum_pending_requests = config.maximum_pending_requests;
      for (unsigned channel = 0; channel < 4; ++channel)
        p.endpoints[channel] = "pih-v41-" + nonce + "-" + std::to_string(rank) + "-" + std::to_string(channel);
      placements[rank] = std::move(p);
    }
    Require(broker.Start(*helper, cgroups[8], *environment, startup, Clock::time_point::max()));
    while (!Take(broker.Poll())) { if (Clock::now() >= startup) throw Status::DeadlineExceeded("V4.1 broker startup deadline"); Yield(); }
    auto id = Take(broker.BorrowId());
    for (auto& placement : placements) { placement.nccl_id = id; placement = Take(seed->BindBootstrap(placement)); }
    Require(group.Start(*worker, {ranks.data(), placements.size()}, placements, *environment, broker));
    while (!Take(group.Poll())) { if (Clock::now() >= startup) throw Status::DeadlineExceeded("V4.1 rank connection deadline"); Yield(); }
    while (!Take(group.PollPackedReady())) { if (Clock::now() >= startup) throw Status::DeadlineExceeded("V4.1 rank startup deadline"); Yield(); }
    session = Take(group.StartPackedGeneration(config.identity.epoch, config.maximum_batch_sequences,
        config.maximum_batch_tokens, config.maximum_pending_requests, std::chrono::milliseconds(config.retirement_ms), config.first_plan));
  }
};
[[noreturn]] void Child(int fd, SupervisorDeployment& deployment) noexcept {
  Socket socket(fd); ChildOwners* owners = nullptr;
  const auto& config = deployment.configuration(); bool connected = true;
  Status failure = Status::Ok();
  try {
    owners = new ChildOwners;
    owners->Start(deployment); socket.StatusFrame(kReady, 0, Status::Ok()); socket.Flush();
    for (;;) {
      const auto input_slice = Clock::now(); std::uint32_t commands = 0;
      while (commands++ < 32 && Clock::now() - input_slice < std::chrono::milliseconds(5) &&
          socket.request_credit(Status::kMaxMessageBytes)) {
        auto frame = socket.Receive(); if (!frame) break;
        const auto id = frame->header.id;
        if (frame->header.kind == kShutdown) {
          failure = Status::Unavailable("V4.1 batch service retiring");
          const bool retired = owners->Cleanup(config.retirement_ms);
          socket.StatusFrame(kRetired, 0, retired ? Status::Ok() : Status::Internal("V4.1 retirement unresolved; owner retained"));
          while (!socket.empty()) { socket.Flush(); Yield(); }
          if (retired) { delete owners; ::_exit(0); }
          continue;
        }
        if (frame->header.kind == kCancel) {
          auto found = owners->requests.find(id);
          if (found != owners->requests.end()) {
            found->second.cancelled = true;
            if (found->second.activated) Require(owners->session->Cancel(found->second.request->ledger().identity().sequence_generation));
            else {
              owners->identities.erase(found->second.request.get()); owners->requests.erase(found);
              socket.StatusFrame(kComplete, id, Status::Unavailable("V4.1 request cancelled before model submission"));
            }
          }
          continue;
        }
        if (frame->header.kind == kBegin) {
          auto found = owners->requests.find(id);
          if (found == owners->requests.end()) continue; // A staged expiry/cancel can race caller start.
          if (found->second.activated) throw Status::Internal("V4.1 IPC duplicate begin identity");
          auto status = Clock::now() >= found->second.deadline ? Status::DeadlineExceeded("V4.1 staged request deadline elapsed") :
              failure.ok() ? owners->session->Admit(*found->second.request, found->second.deadline) : failure;
          if (!status.ok()) {
            owners->identities.erase(found->second.request.get()); owners->requests.erase(found);
            socket.StatusFrame(kComplete, id, status);
          } else found->second.activated = true;
          continue;
        }
        if (frame->header.kind != kAdmit) throw Status::Internal("V4.1 IPC input kind invalid");
        if (!failure.ok()) { socket.StatusFrame(kComplete, id, failure); continue; }
        if (owners->requests.size() >= config.maximum_batch_sequences + config.maximum_pending_requests) {
          socket.StatusFrame(kComplete, id, Status::ResourceExhausted("V4.1 child request admission full")); continue;
        }
        std::string_view body = frame->body;
        const auto deadline_ns = Read<std::uint64_t>(body);
        if (!deadline_ns || deadline_ns > INT64_MAX) throw Status::Internal("V4.1 IPC absolute deadline invalid");
        const auto deadline = Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(deadline_ns)));
        auto sampling = Read<SamplingParameters>(body); TokenStopConfig stop;
        stop.minimum = Read<std::uint32_t>(body); stop.maximum = Read<std::uint32_t>(body);
        stop.token_count = Read<std::uint32_t>(body); for (auto& token : stop.tokens) token = Read<std::uint32_t>(body);
        stop.pattern_count = Read<std::uint32_t>(body);
        if (stop.pattern_count > 16 || stop.token_count > 17) throw Status::Internal("V4.1 IPC stop extent invalid");
        for (std::uint32_t i = 0; i < stop.pattern_count; ++i) stop.patterns[i] = ReadString(body);
        auto prompt = ReadString(body); if (!body.empty()) throw Status::Internal("V4.1 IPC trailing request bytes");
        if (Clock::now() >= deadline) { socket.StatusFrame(kComplete, id, Status::DeadlineExceeded("V4.1 admission deadline elapsed")); continue; }
        auto admitted = deployment.Admit(prompt, sampling, std::move(stop));
        if (!admitted.ok()) { socket.StatusFrame(kComplete, id, admitted.status()); continue; }
        if (Clock::now() >= deadline) { socket.StatusFrame(kComplete, id, Status::DeadlineExceeded("V4.1 tokenization deadline elapsed")); continue; }
        auto request = std::move(*admitted); auto* pointer = request.get();
        if (request->prompt().size() > config.maximum_batch_tokens) {
          socket.StatusFrame(kComplete, id, Status::InvalidArgument("V4.1 prompt exceeds packed token budget")); continue;
        }
        if (owners->requests.contains(id)) throw Status::Internal("V4.1 duplicate request identity");
        owners->identities.emplace(pointer, id); owners->requests.emplace(id, ChildOwners::Request{std::move(request), false, 0, false, deadline});
        socket.Enqueue(kAdmitted, id);
      }
      for (auto it = owners->requests.begin(); it != owners->requests.end();) {
        if (!it->second.activated && Clock::now() >= it->second.deadline) {
          socket.StatusFrame(kComplete, it->first, Status::DeadlineExceeded("V4.1 staged request deadline elapsed"));
          owners->identities.erase(it->second.request.get()); it = owners->requests.erase(it);
        } else ++it;
      }
      if (owners->session && failure.ok() && socket.model_credit(config.maximum_batch_sequences, config.maximum_pending_requests)) {
        auto status = owners->session->Poll();
        if (!status.ok()) { failure = status; owners->session->Fail(); }
        if (failure.ok()) {
          while (auto output = Take(owners->session->TakeOutput())) {
            const auto id = owners->identities.at(output->request); auto& request = owners->requests.at(id);
            auto publication = Take(output->request->output().Read(output->lease));
            if (!request.cancelled) {
              socket.Enqueue(kOutput, id, {publication.visible.data(), publication.visible_size}); request.visible += publication.visible_size;
            }
            Require(output->request->output().Release(output->lease));
          }
          while (auto completed = Take(owners->session->TakeCompleted())) {
            const auto id = owners->identities.at(completed->request); auto& request = owners->requests.at(id);
            const auto& ledger = completed->request->ledger();
            auto status = completed->status;
            if (request.cancelled && status.ok()) status = Status::Unavailable("V4.1 request cancelled");
            if (status.ok()) {
              SupervisorCompletion result{ledger.identity().sequence_generation, ledger.prompt_tokens(), ledger.records().size(), request.visible, ledger.finish()};
              std::string bytes; Append(bytes, result); socket.Enqueue(kComplete, id, std::move(bytes));
            } else socket.StatusFrame(kComplete, id, status);
            owners->identities.erase(completed->request); owners->requests.erase(id);
          }
        } else {
          // Retain inflight ledger/output borrowers until ranks are proven dead.
          std::vector<std::uint64_t> failed;
          for (const auto& [id, _] : owners->requests) failed.push_back(id);
          socket.StatusFrame(kReady, 0, failure); socket.Flush();
          (void)owners->Cleanup(config.retirement_ms);
          for (auto id : failed) socket.StatusFrame(kComplete, id, failure);
        }
      }
      socket.Flush(); Yield();
    }
  } catch (const Status& status) { failure = status; }
  catch (...) { failure = Status::Internal("V4.1 batch child exception"); }
  const bool clean = !owners || owners->Cleanup(config.retirement_ms);
  try { socket.StatusFrame(kReady, 0, failure); socket.Flush(); } catch (...) { connected = false; }
  if (clean) { delete owners; ::_exit(1); }
  // The failed process remains the sole custody owner. It may retry retirement
  // through Close; EOF also triggers cleanup without orphaning rank descendants.
  for (;;) {
    try {
      if (connected) {
        auto frame = socket.Receive();
        if (frame && frame->header.kind == kShutdown) {
          const bool retired = owners->Cleanup(config.retirement_ms);
          socket.StatusFrame(kRetired, 0, retired ? Status::Ok() : Status::Internal("V4.1 retirement unresolved")); socket.Flush();
          if (retired) { delete owners; ::_exit(1); }
        }
        socket.Flush();
      } else if (owners->Cleanup(config.retirement_ms)) { delete owners; ::_exit(1); }
    } catch (...) { connected = false; }
    Yield();
  }
}
}

struct SupervisorBatchExecution::Impl {
  struct Request { std::string body; std::deque<std::string> output; std::size_t output_bytes = 0;
    SupervisorCompletion completion{}; Status status = Status::Ok(); bool sent = false, cancel = false, cancel_sent = false, done = false;
    bool admitted = false, start_called = false, begin = false, begin_sent = false; };
  Impl(int fd, pid_t pid, const SupervisorConfig& config) : socket(fd), pid(pid), limit(config.maximum_batch_sequences + config.maximum_pending_requests),
      startup_ms(config.startup_ms), sequence_ms(config.sequence_ms), retirement_ms(config.retirement_ms) {}
  Socket socket; pid_t pid; std::uint32_t limit, startup_ms, sequence_ms, retirement_ms;
  std::mutex mutex; std::condition_variable cv; std::thread thread;
  std::map<std::uint64_t, std::shared_ptr<Request>> requests;
  std::uint64_t next = 1; Status startup = Status::Unavailable("V4.1 batch child starting"), failure = Status::Ok(), close_status = Status::Ok();
  bool ready = false, close_requested = false, close_sent = false, close_received = false, retired = false, exit = false;
  bool ReapChild(Clock::time_point deadline) {
    while (Clock::now() < deadline) {
      int status = 0; auto reaped = ::waitpid(pid, &status, WNOHANG);
      if (reaped == pid) return true;
      if (reaped < 0 && errno != EINTR) return false;
      Yield();
    }
    return false;
  }
  Status CloseSynchronously() {
    const auto deadline = Clock::now() + std::chrono::milliseconds(retirement_ms);
    try {
      socket.Enqueue(kShutdown, 0);
      while (Clock::now() < deadline) {
        socket.Flush();
        if (auto frame = socket.Receive(); frame && frame->header.kind == kRetired) {
          auto status = DecodeStatus(frame->header.code, frame->body); if (!status.ok()) return status;
          if (ReapChild(deadline)) { retired = true; return Status::Ok(); }
          return Status::Internal("V4.1 child reaping unresolved");
        }
        Yield();
      }
    } catch (...) {
      // A clean startup-error child can close before receiving shutdown.
      while (Clock::now() < deadline) {
        int status = 0; auto reaped = ::waitpid(pid, &status, WNOHANG);
        if (reaped == pid && WIFEXITED(status) && (WEXITSTATUS(status) == 0 || WEXITSTATUS(status) == 1)) {
          retired = true; return Status::Ok();
        }
        if (reaped < 0 && errno != EINTR) break; Yield();
      }
    }
    return Status::Internal("V4.1 synchronous child retirement unresolved; retain owner");
  }
  void Fail(Status status) {
    failure = status;
    for (auto& [_, request] : requests) { request->status = status; request->done = true; }
    requests.clear(); cv.notify_all();
  }
  void Loop() noexcept {
    std::unique_lock lock(mutex);
    try {
      std::uint32_t drained = 0;
      for (;;) {
        if (exit) return;
        if (ready && startup.ok()) {
          for (auto it = requests.begin(); it != requests.end();) {
            const auto id = it->first; auto r = it->second;
            if (r->cancel && !r->sent) { r->done = true; it = requests.erase(it); cv.notify_all(); continue; }
            if (!r->sent && socket.request_credit(r->body.size())) { socket.Enqueue(kAdmit, id, std::move(r->body)); r->sent = true; }
            if (r->begin && !r->cancel && !r->begin_sent) { socket.Enqueue(kBegin, id); r->begin_sent = true; }
            if (r->sent && r->cancel && !r->cancel_sent) { socket.Enqueue(kCancel, id); r->cancel_sent = true; }
            ++it;
          }
        }
        if (close_requested && !close_sent) { socket.Enqueue(kShutdown, 0); close_sent = true; }
        lock.unlock(); socket.Flush(); auto frame = socket.Receive(); lock.lock();
        if (frame) {
          const auto& h = frame->header;
          if (h.kind == kReady) {
            const bool previously_ready = ready;
            startup = DecodeStatus(h.code, frame->body); ready = true;
            if (previously_ready && !startup.ok()) Fail(startup);
            cv.notify_all();
          }
          else if (h.kind == kRetired) {
            close_status = DecodeStatus(h.code, frame->body); close_received = true;
            if (close_status.ok()) {
              lock.unlock(); const bool reaped = ReapChild(Clock::now() + std::chrono::seconds(1));
              lock.lock();
              if (reaped) { retired = true; exit = true; Fail(Status::Unavailable("V4.1 batch service closed")); }
              else close_status = Status::Internal("V4.1 child reaping unresolved");
            }
            cv.notify_all();
          } else {
            auto found = requests.find(h.id);
            if (found == requests.end()) {
              if (!failure.ok()) continue;
              throw Status::Internal("V4.1 IPC unknown request generation");
            }
            auto r = found->second;
            if (h.kind == kAdmitted) {
              if (r->admitted || !frame->body.empty() || h.code) throw Status::Internal("V4.1 duplicate or invalid admission handshake");
              r->admitted = true;
            } else if (h.kind == kOutput) {
              if (!r->begin_sent) throw Status::Internal("V4.1 output preceded model submission handshake");
              if (!r->cancel) {
                if (frame->body.size() > 65536 - r->output_bytes) { r->cancel = true; r->status = Status::ResourceExhausted("V4.1 slow output consumer queue full"); }
                else { r->output_bytes += frame->body.size(); r->output.push_back(std::move(frame->body)); }
              }
            } else if (h.kind == kComplete) {
              if (h.code) { if (r->status.ok()) r->status = DecodeStatus(h.code, frame->body); }
              else { std::string_view bytes = frame->body; r->completion = Read<SupervisorCompletion>(bytes); if (!bytes.empty()) throw Status::Internal("V4.1 IPC completion extent"); }
              r->done = true; requests.erase(found);
            } else throw Status::Internal("V4.1 IPC response kind invalid");
            cv.notify_all();
          }
        }
        if (frame && ++drained < 256) continue;
        drained = 0;
        if (!exit) cv.wait_for(lock, std::chrono::milliseconds(1));
      }
    } catch (const Status& status) { if (!lock.owns_lock()) lock.lock(); Fail(status); }
    catch (...) { if (!lock.owns_lock()) lock.lock(); Fail(Status::Internal("V4.1 IPC owner exception")); }
    // EOF alone cannot prove rank retirement. Preserve owner and report failure;
    // the child performs custody cleanup before exiting, then waitpid proves it.
    int status = 0; auto reaped = ::waitpid(pid, &status, WNOHANG);
    if (reaped == pid && WIFEXITED(status) && (WEXITSTATUS(status) == 0 || WEXITSTATUS(status) == 1)) {
      retired = true; close_received = true; close_status = Status::Ok();
    }
    if (!ready) { startup = failure; ready = true; }
    exit = true;
    cv.notify_all();
  }
};
SupervisorBatchExecution::SupervisorBatchExecution(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SupervisorBatchExecution::~SupervisorBatchExecution() {
  if (!impl_->retired) (void)Close();
  // An unresolved provider must retain the owner instead of destroying it.
  if (!impl_->retired) std::terminate();
  if (impl_->thread.joinable()) impl_->thread.join();
}
Result<std::unique_ptr<SupervisorBatchExecution>> SupervisorBatchExecution::Create(SupervisorDeployment& deployment) {
  auto admitted = AdmitSupervisorProcess(); if (!admitted.ok()) return admitted;
  int fds[2]; if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0, fds)) return Status::Unavailable("V4.1 IPC socketpair failed");
  std::unique_ptr<SupervisorBatchExecution> owner;
  try { owner.reset(new SupervisorBatchExecution(std::make_unique<Impl>(fds[0], 0, deployment.configuration()))); }
  catch (...) { ::close(fds[0]); ::close(fds[1]); return Status::ResourceExhausted("V4.1 supervisor owner allocation failed"); }
  const auto pid = ::fork();
  if (pid < 0) { ::close(fds[1]); owner->impl_->retired = true; return Status::Unavailable("V4.1 batch supervisor fork failed"); }
  if (!pid) { ::close(fds[0]); Child(fds[1], deployment); }
  ::close(fds[1]);
  owner->impl_->pid = pid;
  try { owner->impl_->thread = std::thread([impl = owner->impl_.get()] { impl->Loop(); }); }
  catch (...) {
    owner->impl_->startup = Status::ResourceExhausted("V4.1 IPC owner thread unavailable");
    owner->impl_->ready = true; (void)owner->impl_->CloseSynchronously(); return owner;
  }
  std::unique_lock lock(owner->impl_->mutex);
  if (!owner->impl_->cv.wait_for(lock, std::chrono::milliseconds(owner->impl_->startup_ms + owner->impl_->retirement_ms), [&] { return owner->impl_->ready; }))
    owner->impl_->startup = Status::DeadlineExceeded("V4.1 child startup handshake elapsed; owner retained");
  return owner;
}
Status SupervisorBatchExecution::startup_status() const noexcept { std::lock_guard lock(impl_->mutex); return impl_->startup; }
bool SupervisorBatchExecution::retired() const noexcept { std::lock_guard lock(impl_->mutex); return impl_->retired; }
bool SupervisorBatchExecution::healthy() const noexcept {
  std::lock_guard lock(impl_->mutex); return impl_->startup.ok() && impl_->failure.ok() && !impl_->close_requested;
}
Result<SupervisorCompletion> SupervisorBatchExecution::Run(std::string_view prompt, SamplingParameters sampling,
    TokenStopConfig stopping, void* context, SupervisorOutput::Writer writer, SupervisorExecution::Cancelled cancelled, Started started, Clock::time_point deadline) noexcept {
  try {
    if (!writer || !cancelled || !started) return Status::InvalidArgument("V4.1 batch callbacks missing");
    if (deadline > Clock::time_point::max() - std::chrono::milliseconds(impl_->retirement_ms))
      return Status::InvalidArgument("V4.1 request deadline leaves no retirement interval");
    if (Clock::now() >= deadline) return Status::DeadlineExceeded("V4.1 request deadline elapsed before admission");
    auto r = std::make_shared<Impl::Request>(); r->body = RequestBytes(prompt, sampling, stopping, deadline);
    std::unique_lock lock(impl_->mutex);
    if (!impl_->startup.ok() || !impl_->failure.ok() || impl_->close_requested) return Status::Unavailable("V4.1 batch owner unavailable");
    if (impl_->requests.size() >= impl_->limit) return Status::ResourceExhausted("V4.1 pending request queue full");
    if (impl_->next == UINT64_MAX) return Status::ResourceExhausted("V4.1 request identity exhausted");
    impl_->requests.emplace(impl_->next++, r); impl_->cv.notify_all();
    for (;;) {
      lock.unlock(); bool cancel = false; Status callback_failure = Status::Ok();
      try { cancel = cancelled(context); } catch (...) { cancel = true; callback_failure = Status::Internal("V4.1 cancellation callback exception"); }
      lock.lock();
      if (!callback_failure.ok()) r->status = callback_failure;
      if (cancel || Clock::now() >= deadline) {
        r->cancel = true;
        if (r->status.ok()) r->status = Clock::now() >= deadline ? Status::DeadlineExceeded("V4.1 request deadline elapsed") : Status::Unavailable("V4.1 request cancelled");
        impl_->cv.notify_all();
      }
      if (r->admitted && !r->start_called && !r->cancel && !r->done) {
        r->start_called = true;
        lock.unlock(); Status status = Status::Ok();
        try { status = started(context); } catch (...) { status = Status::Internal("V4.1 output start callback exception"); }
        lock.lock();
        if (Clock::now() >= deadline && status.ok()) status = Status::DeadlineExceeded("V4.1 start callback exceeded request deadline");
        if (status.ok()) r->begin = true;
        else { r->status = status; r->cancel = true; }
        impl_->cv.notify_all(); continue;
      }
      if (!r->output.empty()) {
        auto bytes = std::move(r->output.front()); r->output.pop_front(); r->output_bytes -= bytes.size();
        const bool deliver = !r->cancel;
        lock.unlock();
        Status failure = Status::Ok();
        if (deliver) {
          try {
            std::size_t offset = 0;
            while (offset < bytes.size()) {
              auto sent = writer(context, std::string_view(bytes).substr(offset));
              if (!sent.ok()) { failure = sent.status(); break; }
              if (!*sent || *sent > bytes.size() - offset) { failure = Status::Unavailable("V4.1 callback failed to consume output"); break; }
              offset += *sent;
            }
          } catch (...) { failure = Status::Internal("V4.1 output callback exception"); }
        }
        lock.lock(); if (!failure.ok()) { r->status = failure; r->cancel = true; impl_->cv.notify_all(); } continue;
      }
      if (r->done) return r->status.ok() ? Result<SupervisorCompletion>(r->completion) : Result<SupervisorCompletion>(r->status);
      if (Clock::now() >= deadline + std::chrono::milliseconds(impl_->retirement_ms)) {
        auto status = Status::Internal("V4.1 request cancellation not retired; restart required");
        impl_->Fail(status); return status;
      }
      impl_->cv.wait_for(lock, std::chrono::milliseconds(1));
    }
  } catch (...) { return Status::Internal("V4.1 batch request exception"); }
}
Status SupervisorBatchExecution::Close() noexcept {
  try {
    std::unique_lock lock(impl_->mutex);
    if (impl_->retired) return Status::Ok();
    if (!impl_->thread.joinable() || impl_->exit) {
      lock.unlock(); return impl_->CloseSynchronously();
    }
    impl_->close_requested = true; impl_->close_sent = false; impl_->close_received = false; impl_->cv.notify_all();
    if (!impl_->cv.wait_for(lock, std::chrono::milliseconds(impl_->retirement_ms + 1000), [&] { return impl_->close_received || impl_->retired; }))
      return Status::Internal("V4.1 supervisor retirement not proven; retain owner");
    return impl_->close_status;
  } catch (...) { return Status::Internal("V4.1 supervisor Close exception; retain owner"); }
}
}
