#pragma once
#include "pih/core/result.h"
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace pih::deepseek_v41 {
// Explicit native-worker environment, never copied from supervisor environ.
// Directory checks are not authentication of the shared libraries inside them.
class WorkerEnvironment final {
 public:
  static Result<std::unique_ptr<WorkerEnvironment>> Create(
      std::span<const std::string> library_directories);
  WorkerEnvironment(const WorkerEnvironment&) = delete;
  WorkerEnvironment& operator=(const WorkerEnvironment&) = delete;
  std::span<const std::string> entries() const noexcept { return entries_; }
 private:
  WorkerEnvironment() = default;
  std::vector<std::string> entries_;
};
}  // namespace pih::deepseek_v41
