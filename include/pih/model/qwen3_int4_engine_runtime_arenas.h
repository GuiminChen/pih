#pragma once
#include "pih/model/qwen3_bf16_engine_arenas.h"
#include "pih/model/qwen3_bf16_pinned_arenas.h"
#include "pih/model/qwen3_int4_engine_bootstrap_plan.h"
#include "pih/model/qwen3_int4_synchronous_backend.h"
namespace pih{
struct QwenInt4BackendArenaOwnerIds final{
 std::uint64_t pinned_staging,device_staging,sampled_token,device_error,pinned_result;
};
class QwenInt4EngineRuntimeArenas final{public:
 static Result<QwenInt4EngineRuntimeArenas>Allocate(const QwenInt4EngineBootstrapPlan& plan,
  Allocator& device_allocator,RegisteredPinnedAllocator& pinned_allocator,
  PinnedPlacementVerifier& verifier,std::int32_t numa_node,std::int32_t owning_rank);
 static Result<QwenInt4EngineRuntimeArenas>Allocate(const QwenInt4EngineResourcePlan& plan,
  Allocator& device_allocator,RegisteredPinnedAllocator& pinned_allocator,
  PinnedPlacementVerifier& verifier,std::int32_t numa_node,std::int32_t owning_rank);
 QwenInt4EngineRuntimeArenas(const QwenInt4EngineRuntimeArenas&)=delete;
 QwenInt4EngineRuntimeArenas& operator=(const QwenInt4EngineRuntimeArenas&)=delete;
 QwenInt4EngineRuntimeArenas(QwenInt4EngineRuntimeArenas&&) noexcept=default;
 QwenInt4EngineRuntimeArenas& operator=(QwenInt4EngineRuntimeArenas&&) noexcept=default;
 [[nodiscard]]const QwenBf16EngineDeviceArenas& device()const noexcept{return device_;}
 [[nodiscard]]QwenBf16EngineDeviceArenas& device()noexcept{return device_;}
 [[nodiscard]]QwenBf16PinnedHostArenas& pinned()noexcept{return pinned_;}
 Result<QwenInt4SynchronousBackendArenas> backend_arenas(
  QwenInt4BackendArenaOwnerIds owner_ids,std::int32_t owning_rank);
 private:QwenInt4EngineRuntimeArenas(QwenBf16EngineDeviceArenas d,QwenBf16PinnedHostArenas p)
  :device_(std::move(d)),pinned_(std::move(p)){}
 QwenBf16EngineDeviceArenas device_;QwenBf16PinnedHostArenas pinned_;
};}
