#pragma once
#include "pih/core/sha256.h"
#include <chrono>
#include <filesystem>
#include <memory>

namespace pih::deepseek_v41 {
// Owns a digest-admitted immutable ELF snapshot, not a mutable source path.
// The supplied digest must come from trusted deployment admission.
class WorkerExecutable final {
 public:
  using Clock = std::chrono::steady_clock;
  static Result<std::unique_ptr<WorkerExecutable>> Open(const std::filesystem::path& path,
      const Sha256Digest& expected, std::uint64_t byte_budget, Clock::time_point deadline);
  WorkerExecutable(const WorkerExecutable&) = delete;
  WorkerExecutable& operator=(const WorkerExecutable&) = delete;
  ~WorkerExecutable();
  Result<int> Descriptor() const;
 private:
  WorkerExecutable() = default;
  int fd_ = -1;
  std::uint64_t bytes_ = 0;
};
}  // namespace pih::deepseek_v41
