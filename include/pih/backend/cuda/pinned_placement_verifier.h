#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

class PinnedPlacementVerifier {
 public:
  virtual ~PinnedPlacementVerifier() = default;
  virtual Status verify(const void* data, std::uint64_t bytes,
                        std::int32_t numa_node) = 0;
};

}  // namespace pih
