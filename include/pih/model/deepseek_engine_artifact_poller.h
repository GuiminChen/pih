#pragma once
#include <cstdint>
#include "pih/core/status.h"

namespace pih {
// Model-local adapter over storage capabilities; no controller file descriptors
// or platform-specific verity receipts cross this interface.
class DeepSeekEngineArtifactPoller {
 public:
  virtual ~DeepSeekEngineArtifactPoller() = default;
  virtual std::uint32_t world_size() const noexcept = 0;
  virtual Status poll() = 0;
};
}  // namespace pih
