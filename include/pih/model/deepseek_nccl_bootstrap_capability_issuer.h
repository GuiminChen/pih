#pragma once

#include <array>
#include <memory>
#include <vector>

#include "pih/model/deepseek_nccl_bootstrap_lease.h"

namespace pih {

class DeepSeekNcclUniqueIdSource {
 public:
  virtual ~DeepSeekNcclUniqueIdSource() = default;
  virtual Result<std::array<std::byte,
                            DeepSeekNcclBootstrapLease::kUniqueIdBytes>>
  get_unique_id() = 0;
};

struct DeepSeekNcclIssuedEdgeCapability final {
  std::uint32_t edge_id = 0;
  std::uint64_t lease_id = 0;
  std::uint64_t commitment_id = 0;
  std::unique_ptr<DeepSeekNcclBootstrapLease> lower;
  std::unique_ptr<DeepSeekNcclBootstrapLease> upper;
};

class DeepSeekNcclBootstrapCapabilityIssuer final {
 public:
  static Result<std::vector<DeepSeekNcclIssuedEdgeCapability>> Issue(
      std::uint64_t engine_epoch, std::uint32_t world_size,
      std::uint64_t first_lease_id, DeepSeekNcclUniqueIdSource& source,
      RegisteredPinnedAllocator& allocator);
};

}  // namespace pih
