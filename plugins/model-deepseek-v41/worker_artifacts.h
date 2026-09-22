#pragma once
#include "worker_bootstrap.h"
#include "weight_files.h"
#include "engram_hash.h"

namespace pih::deepseek_v41 {
// CPU admission before CUDA/NCCL. Owned immutable config, fresh sequence hash
// state, authenticated weight FDs and exact plugin-lock byte snapshot.
class WorkerArtifacts final {
 public:
  static Result<std::unique_ptr<WorkerArtifacts>> Open(const WorkerBootstrap& bootstrap);
  WorkerArtifacts(const WorkerArtifacts&) = delete;
  WorkerArtifacts& operator=(const WorkerArtifacts&) = delete;
  const FlashConfig& config() const noexcept { return config_; }
  EngramHashState& hashes() noexcept { return hashes_; }
  const BackboneWeightFiles& weights() const noexcept { return *weights_; }
  // Parse this authenticated snapshot. Reopening the lock path would lose the
  // digest-to-use binding; plugin binaries still require their own verification.
  std::string_view plugin_lock_json() const noexcept { return lock_; }
 private:
  WorkerArtifacts(FlashConfig config, EngramHashState hashes)
      : config_(std::move(config)), hashes_(std::move(hashes)) {}
  FlashConfig config_;
  EngramHashState hashes_;
  std::unique_ptr<BackboneWeightFiles> weights_;
  std::string lock_;
};
}  // namespace pih::deepseek_v41
