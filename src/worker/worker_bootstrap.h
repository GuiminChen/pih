#pragma once

#include <string>
#include <cstdint>
#include <vector>
#include <memory>
#include "pih/core/sha256.h"

namespace pih::worker {

// Backend-only rank activation from an authenticated lock snapshot. Single
// thread; keep alive until all CUDA/NCCL/resource users have retired. Methods
// throw on invalid locks/provider errors, like the underlying native loaders.
class WorkerPluginStack final {
 public:
  WorkerPluginStack();
  WorkerPluginStack(const WorkerPluginStack&) = delete;
  WorkerPluginStack& operator=(const WorkerPluginStack&) = delete;
  ~WorkerPluginStack();
  void Start(const std::string& snapshot, const std::string& absolute_lock_path,
      const Sha256Digest& expected_digest, std::uint64_t epoch);
  const void* Resolve(const std::string& contract, std::uint32_t scope, std::uint32_t cardinality) const;
  // Close registration/resolution and shut down the admitted stack exactly once.
  // Failure is terminal; destruction preserves host contexts for process exit.
  void Shutdown();
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

int RunDevelopmentLock(const std::string& lock_path, std::uint16_t serve_port = 0,
    const std::vector<std::uint32_t>& prompt_tokens = {},
    std::uint32_t maximum_completion_tokens = 128);

}  // namespace pih::worker
