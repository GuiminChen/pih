#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "worker_bootstrap.h"
#include <algorithm>
#include <bit>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <new>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pih::deepseek_v41 {
namespace {
bool Path(std::string_view path) {
  if (path.size() < 2 || path.size() > 511 || path.front() != '/' || path.back() == '/') return false;
  for (const unsigned char c : path) if (c < 32 || c == 127) return false;
  path.remove_prefix(1);
  while (!path.empty()) {
    const auto end = path.find('/'); const auto part = path.substr(0, end);
    if (part.empty() || part == "." || part == "..") return false;
    if (end == std::string_view::npos) break;
    path.remove_prefix(end + 1);
  }
  return true;
}
constexpr int kSeals = F_SEAL_WRITE | F_SEAL_GROW | F_SEAL_SHRINK | F_SEAL_SEAL;
}
Status WorkerBootstrap::Validate() const {
  if (!supervisor_pid || supervisor_pid > INT32_MAX || (world != 2 && world != 4 && world != 8) || rank >= world ||
      device > INT32_MAX || sm_major != 10 || sm_minor != 3 || !prompt_tokens || prompt_tokens > 4096 ||
      maximum_positions < prompt_tokens || maximum_positions > FlashConfig::kMaximumPositions ||
      !retirement_ms || retirement_ms > 300000 || !identity.epoch || identity.plan_seq ||
      !identity.sequence_generation || !identity.sampling_config_id || !device_budget || !host_budget ||
      !staging_bytes || staging_bytes > (1U << 20) || staging_bytes % 256 || staging_bytes > host_budget ||
      !startup_ns || sequence_ns <= startup_ns || sequence_ns > INT64_MAX ||
      sampling.ordinal || sampling.suppressed_count || !Path(artifact_directory) || !Path(plugin_lock))
    return Status::InvalidArgument("Worker bootstrap identity, topology, budgets, paths or deadlines invalid");
  const auto sampled = ValidateSamplingParameters(sampling); if (!sampled.ok()) return sampled;
  for (const auto token : sampling.suppressed) if (token) return Status::InvalidArgument("Bootstrap suppression slots must be zero");
  for (const auto* hash : {&config_sha256, &map_sha256, &weight_manifest_sha256, &plugin_lock_sha256})
    if (std::all_of(hash->bytes.begin(), hash->bytes.end(), [](std::byte b) { return b == std::byte{}; }))
      return Status::InvalidArgument("Worker bootstrap artifact identity is absent");
  if (std::all_of(nccl_id.begin(), nccl_id.end(), [](std::byte b) { return b == std::byte{}; }))
    return Status::InvalidArgument("Worker NCCL bootstrap ID is absent");
  for (unsigned i = 0; i < 4; ++i) {
    const auto& name = endpoints[i];
    if (name.size() < 32 || name.size() > 96) return Status::InvalidArgument("Worker endpoint length invalid");
    for (const char c : name)
      if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '-' || c == '.'))
        return Status::InvalidArgument("Worker endpoint character invalid");
    for (unsigned j = 0; j < i; ++j) if (name == endpoints[j]) return Status::InvalidArgument("Worker endpoints repeat");
  }
  return Status::Ok();
}
Result<std::array<std::uint8_t, WorkerBootstrap::kWireBytes>> WorkerBootstrap::Encode() const {
  const auto valid = Validate(); if (!valid.ok()) return valid;
  std::array<std::uint8_t, kWireBytes> frame{};
  const auto put = [&](unsigned at, std::uint64_t value, unsigned count) {
    for (unsigned i = 0; i < count; ++i) frame[at + i] = static_cast<std::uint8_t>(value >> (8 * i));
  };
  put(0, 0x31544f4f42484950ULL, 8); put(8, 1, 4); put(12, kWireBytes, 4); // PIHBOOT1
  unsigned at = 16;
  for (auto value : {supervisor_pid, supervisor_uid, world, rank, device, sm_major, sm_minor, prompt_tokens, maximum_positions, retirement_ms}) {
    put(at, value, 4); at += 4;
  }
  at = 56;
  for (auto value : {identity.epoch, identity.sequence_generation, identity.sampling_config_id,
      device_budget, host_budget, staging_bytes, startup_ns, sequence_ns, sampling.seed}) { put(at, value, 8); at += 8; }
  put(128, std::bit_cast<std::uint32_t>(sampling.temperature), 4); put(132, std::bit_cast<std::uint32_t>(sampling.top_p), 4);
  put(136, sampling.top_k, 4); put(140, sampling.logprobs, 4); put(144, sampling.top_count, 4);
  at = 160;
  for (const auto* hash : {&config_sha256, &map_sha256, &weight_manifest_sha256, &plugin_lock_sha256})
    for (const auto b : hash->bytes) frame[at++] = std::to_integer<std::uint8_t>(b);
  for (const auto b : nccl_id) frame[at++] = std::to_integer<std::uint8_t>(b);
  for (unsigned i = 0; i < 4; ++i) {
    put(416 + i * 100, endpoints[i].size(), 4);
    std::copy(endpoints[i].begin(), endpoints[i].end(), frame.begin() + 420 + i * 100);
  }
  std::copy(artifact_directory.begin(), artifact_directory.end(), frame.begin() + 816);
  std::copy(plugin_lock.begin(), plugin_lock.end(), frame.begin() + 1328);
  return frame;
}
Result<WorkerBootstrap> WorkerBootstrap::Decode(std::span<const std::uint8_t> frame) {
  if (frame.size() != kWireBytes) return Status::InvalidArgument("Worker bootstrap extent invalid");
  const auto get = [&](unsigned at, unsigned count) {
    std::uint64_t value = 0; for (unsigned i = 0; i < count; ++i) value |= std::uint64_t(frame[at + i]) << (8 * i); return value;
  };
  if (get(0, 8) != 0x31544f4f42484950ULL || get(8, 4) != 1 || get(12, 4) != kWireBytes || get(140, 4) > 1)
    return Status::InvalidArgument("Worker bootstrap header invalid");
  try {
    WorkerBootstrap out;
    unsigned at = 16;
    for (auto* field : {&out.supervisor_pid, &out.supervisor_uid, &out.world, &out.rank, &out.device,
        &out.sm_major, &out.sm_minor, &out.prompt_tokens, &out.maximum_positions, &out.retirement_ms}) { *field = get(at, 4); at += 4; }
    at = 56;
    for (auto* field : {&out.identity.epoch, &out.identity.sequence_generation, &out.identity.sampling_config_id,
        &out.device_budget, &out.host_budget, &out.staging_bytes, &out.startup_ns, &out.sequence_ns, &out.sampling.seed}) { *field = get(at, 8); at += 8; }
    out.sampling.temperature = std::bit_cast<float>(static_cast<std::uint32_t>(get(128, 4)));
    out.sampling.top_p = std::bit_cast<float>(static_cast<std::uint32_t>(get(132, 4)));
    out.sampling.top_k = get(136, 4); out.sampling.logprobs = get(140, 4); out.sampling.top_count = get(144, 4);
    at = 160;
    for (auto* hash : {&out.config_sha256, &out.map_sha256, &out.weight_manifest_sha256, &out.plugin_lock_sha256})
      for (auto& b : hash->bytes) b = static_cast<std::byte>(frame[at++]);
    for (auto& b : out.nccl_id) b = static_cast<std::byte>(frame[at++]);
    for (unsigned i = 0; i < 4; ++i) {
      const auto length = get(416 + i * 100, 4);
      if (length > 96) return Status::InvalidArgument("Worker endpoint extent invalid");
      out.endpoints[i].assign(reinterpret_cast<const char*>(frame.data() + 420 + i * 100), length);
    }
    const auto path = [&](unsigned start) {
      const auto* first = frame.data() + start;
      const auto* end = std::find(first, first + 512, std::uint8_t{});
      return std::string(reinterpret_cast<const char*>(first), static_cast<std::size_t>(end - first));
    };
    out.artifact_directory = path(816); out.plugin_lock = path(1328);
    const auto canonical = out.Encode(); if (!canonical.ok()) return canonical.status();
    if (!std::equal(frame.begin(), frame.end(), canonical->begin()))
      return Status::InvalidArgument("Worker bootstrap unused bytes or padding are noncanonical");
    return out;
  } catch (const std::bad_alloc&) { return Status::ResourceExhausted("Worker bootstrap decode allocation failed"); }
}
SealedWorkerBootstrap::~SealedWorkerBootstrap() { if (fd_ >= 0) ::close(fd_); }
Result<SealedWorkerBootstrap> SealedWorkerBootstrap::Create(const WorkerBootstrap& bootstrap) {
  const auto frame = bootstrap.Encode(); if (!frame.ok()) return frame.status();
  SealedWorkerBootstrap out;
  out.fd_ = ::memfd_create("pih-worker-bootstrap", MFD_CLOEXEC | MFD_ALLOW_SEALING);
  if (out.fd_ < 0) return Status::Unavailable("Cannot create worker bootstrap memfd");
  const auto written = ::pwrite(out.fd_, frame->data(), frame->size(), 0);
  if (written != static_cast<ssize_t>(frame->size()) || ::fcntl(out.fd_, F_ADD_SEALS, kSeals))
    return Status::Unavailable("Cannot write/seal complete worker bootstrap");
  return out;
}
Result<WorkerBootstrap> SealedWorkerBootstrap::Read(int fd) {
  SealedWorkerBootstrap held;
  held.fd_ = ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
  if (held.fd_ < 0) return Status::Unavailable("Cannot retain bootstrap descriptor");
  struct stat info{};
  const int seals = ::fcntl(held.fd_, F_GET_SEALS);
  if (seals < 0 || (seals & kSeals) != kSeals || ::fstat(held.fd_, &info) || !S_ISREG(info.st_mode) ||
      info.st_size != WorkerBootstrap::kWireBytes)
    return Status::FailedPrecondition("Worker bootstrap is not exact-size sealed storage");
  std::array<std::uint8_t, WorkerBootstrap::kWireBytes> frame{};
  if (::pread(held.fd_, frame.data(), frame.size(), 0) != static_cast<ssize_t>(frame.size()))
    return Status::Unavailable("Cannot read complete worker bootstrap");
  return WorkerBootstrap::Decode(frame);
}
}  // namespace pih::deepseek_v41
