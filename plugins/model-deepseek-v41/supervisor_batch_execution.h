#pragma once
#include "supervisor_execution.h"
#include <memory>
#include <cstring>
#include <optional>

namespace pih::deepseek_v41::batch_ipc {
inline constexpr std::uint32_t kMagic = 0x31423456, kMaximumFrame = (1U << 20) + 8192;
struct Header { std::uint32_t magic = kMagic, kind = 0; std::uint64_t id = 0; std::uint32_t bytes = 0, code = 0; };
static_assert(sizeof(Header) == 24);
struct Frame { Header header; std::string body; };
// Portable bounded stream framing. The forked endpoints share the same native
// representation; this is not a remotely callable or persisted ABI.
class Decoder final {
 public:
  Status Feed(std::string_view bytes) {
    if (bytes.size() > kMaximumFrame + sizeof(Header) + 16384 - buffered_.size())
      return Status::ResourceExhausted("V4.1 IPC receive limit");
    buffered_.append(bytes); return Status::Ok();
  }
  Result<std::optional<Frame>> Take() {
    if (buffered_.size() < sizeof(Header)) return std::optional<Frame>{};
    Header header; std::memcpy(&header, buffered_.data(), sizeof(header));
    if (header.magic != kMagic || header.bytes > kMaximumFrame || header.kind < 1 || header.kind > 9 || header.code > 6)
      return Status::Internal("V4.1 IPC frame contract invalid");
    if (buffered_.size() < sizeof(Header) + header.bytes) return std::optional<Frame>{};
    Frame frame{header, buffered_.substr(sizeof(Header), header.bytes)};
    buffered_.erase(0, sizeof(Header) + header.bytes); return std::optional<Frame>{std::move(frame)};
  }
 private:
  std::string buffered_;
};
}

namespace pih::deepseek_v41 {
// Load forks one dedicated CPU supervisor before service threads exist. The
// child alone owns rank processes, packed dispatch and token publication leases.
// A failed Close retains this owner; callers must refuse provider unload.
class SupervisorBatchExecution final {
 public:
  using Started = Status (*)(void*);
  static Result<std::unique_ptr<SupervisorBatchExecution>> Create(SupervisorDeployment& deployment);
  ~SupervisorBatchExecution();
  SupervisorBatchExecution(const SupervisorBatchExecution&) = delete;
  Status startup_status() const noexcept;
  Result<SupervisorCompletion> Run(std::string_view prompt, SamplingParameters sampling,
      TokenStopConfig stopping, void* context, SupervisorOutput::Writer writer,
      SupervisorExecution::Cancelled cancelled, Started started,
      std::chrono::steady_clock::time_point deadline) noexcept;
  Status Close() noexcept;
  bool retired() const noexcept;
  bool healthy() const noexcept;
 private:
  struct Impl;
  explicit SupervisorBatchExecution(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}
