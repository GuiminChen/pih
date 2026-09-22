#pragma once

#include <memory>
#include <span>

#include "pih/backend/cuda/nvidia_deepseek_rank_runtime.h"
#include "pih/model/deepseek_nccl_boundary_bootstrap.h"

namespace pih {

class NvidiaDeepSeekNcclBoundaryBootstrap final {
 public:
  static Result<std::unique_ptr<DeepSeekNcclGenerationBoundaryAssembly>> Build(
      DeepSeekNcclBoundaryBootstrapConfig config,
      std::span<const std::uint64_t> device_identities,
      std::span<NvidiaDeepSeekRankRuntimeView* const> runtimes,
      std::span<Allocator* const> device_allocators,
      std::span<const DeepSeekNcclReleaseConfig> release_configs,
      RegisteredPinnedAllocator& pinned_allocator,
      std::unique_ptr<DeepSeekNcclCApi> api_owner,
      DeepSeekNcclUniqueIdSource& unique_id_source);
};

}  // namespace pih
