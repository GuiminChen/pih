#pragma once
#include "sampling.h"
#include <string>

namespace pih::deepseek_v41 {
// Supervisor-admitted startup envelope, not a substitute for hashing artifacts.
struct WorkerBootstrap final {
  static constexpr std::size_t kWireBytes = 2048;
  std::uint32_t supervisor_pid = 0, supervisor_uid = 0, world = 0, rank = 0, device = 0;
  std::uint32_t sm_major = 0, sm_minor = 0, prompt_tokens = 0, maximum_positions = 0, retirement_ms = 0;
  SamplingIdentity identity;
  SamplingParameters sampling;
  std::uint64_t device_budget = 0, host_budget = 0, staging_bytes = 0;
  // Absolute CLOCK_MONOTONIC nanoseconds in the same Linux time namespace.
  std::uint64_t startup_ns = 0, sequence_ns = 0;
  Sha256Digest config_sha256, map_sha256, weight_manifest_sha256, plugin_lock_sha256;
  std::array<std::byte, 128> nccl_id{};
  std::array<std::string, 4> endpoints;
  std::string artifact_directory, plugin_lock;
  Status Validate() const;
  Result<std::array<std::uint8_t, kWireBytes>> Encode() const;
  static Result<WorkerBootstrap> Decode(std::span<const std::uint8_t> bytes);
};
// Linux sealed memfd, inherited as FD 3 by WorkerSpawn. Read duplicates the
// input and validates exact size/full seals before parsing. No path fallback.
class SealedWorkerBootstrap final {
 public:
  static Result<SealedWorkerBootstrap> Create(const WorkerBootstrap& bootstrap);
  static Result<WorkerBootstrap> Read(int fd);
  SealedWorkerBootstrap(const SealedWorkerBootstrap&) = delete;
  SealedWorkerBootstrap& operator=(const SealedWorkerBootstrap&) = delete;
  SealedWorkerBootstrap(SealedWorkerBootstrap&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  ~SealedWorkerBootstrap();
  int descriptor() const noexcept { return fd_; }
 private:
  SealedWorkerBootstrap() = default;
  int fd_ = -1;
};
}  // namespace pih::deepseek_v41
