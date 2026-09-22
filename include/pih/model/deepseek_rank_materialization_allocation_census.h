#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/backend/cuda/registered_pinned_allocator.h"
#include "pih/core/allocator.h"
#include "pih/model/engine_cuda_owner_ledger_manifest_compiler.h"
#include "pih/model/engine_manifest_tracking_allocator.h"
#include "pih/model/deepseek_expert_bundle_layout.h"
#include "pih/model/deepseek_weight_materialization_plan.h"
#include "pih/model/runtime_profile_payload.h"

namespace pih {

// Controller-authorized identities for the two backing allocations created by
// a rank during materialization. The canonical authority root is signed into
// type-13 v2 and recomputed by the census builder. Full-resident ranks have no
// pinned backing and must use an entirely zero pinned authority.
struct DeepSeekRankMaterializationAllocationAuthority final {
  std::int32_t device_ordinal = -1;
  RuntimeProfileResidency residency = RuntimeProfileResidency::kFullResident;
  Sha256Digest physical_gpu_identity{};
  std::uint64_t cuda_allocation_identity = 0;
  Sha256Digest cuda_owner_identity{};
  Sha256Digest cuda_resource_identity{};
  std::uint64_t pinned_registration_identity = 0;
  Sha256Digest pinned_owner_identity{};
  Sha256Digest pinned_resource_identity{};
  std::int32_t pinned_numa_node = -1;
};

// Canonical identity of the allocation authority supplied by the controller.
// It covers the NUMA and pinned-registration coordinates even though the
// platform probe that supplies those coordinates is Linux-only today.  A
// worker must never choose an authority whose root differs from its signed
// materialization grant.
Result<Sha256Digest>
compile_deepseek_rank_materialization_allocation_authority_root(
    const DeepSeekRankMaterializationAllocationAuthority& authority);

struct DeepSeekRankMaterializationAllocationCensusObservation final {
  std::uint64_t sample_identity = 0;
  std::uint64_t sample_started_ns = 0;
  std::uint64_t sample_completed_ns = 0;
  std::uint64_t cuda_resident_weight_allocation_bytes = 0;
  std::uint64_t pinned_staging_allocation_bytes = 0;
  Sha256Digest cuda_allocation_root{};
  Sha256Digest pinned_allocation_root{};
};

// Owns the tracked allocator wrappers for the entire lifetime of the buffers
// they allocate.  It is intentionally a fixed two-owner census: a resident
// weight backing always exists; host-spill additionally has one pinned staging
// backing.  Additional materialization allocations require a new ABI.
class DeepSeekRankMaterializationAllocationCensus final {
 public:
  static Result<std::unique_ptr<DeepSeekRankMaterializationAllocationCensus>>
  Create(const DeepSeekRankMaterializationAllocationAuthority& authority,
         std::uint64_t resident_weight_backing_bytes,
         std::uint64_t pinned_staging_backing_bytes,
         Allocator& device_allocator,
         RegisteredPinnedAllocator& pinned_allocator);

  ~DeepSeekRankMaterializationAllocationCensus();

  DeepSeekRankMaterializationAllocationCensus(
      const DeepSeekRankMaterializationAllocationCensus&) = delete;
  DeepSeekRankMaterializationAllocationCensus& operator=(
      const DeepSeekRankMaterializationAllocationCensus&) = delete;
  DeepSeekRankMaterializationAllocationCensus(
      DeepSeekRankMaterializationAllocationCensus&&) = delete;
  DeepSeekRankMaterializationAllocationCensus& operator=(
      DeepSeekRankMaterializationAllocationCensus&&) = delete;

  [[nodiscard]] Allocator& device_allocator() noexcept {
    return *device_allocator_;
  }
  [[nodiscard]] RegisteredPinnedAllocator* pinned_allocator() noexcept {
    return pinned_allocator_.get();
  }
  [[nodiscard]] bool host_spill() const noexcept {
    return pinned_allocator_ != nullptr;
  }
  [[nodiscard]] std::uint64_t resident_weight_backing_bytes() const noexcept {
    return resident_weight_backing_bytes_;
  }
  [[nodiscard]] std::uint64_t pinned_staging_backing_bytes() const noexcept {
    return pinned_staging_backing_bytes_;
  }
  Result<DeepSeekRankMaterializationAllocationCensusObservation> capture()
      const;

 private:
  class Clock;
  DeepSeekRankMaterializationAllocationCensus(
      std::unique_ptr<Clock> clock,
      std::unique_ptr<EngineCudaOwnerLedgerRegistry> registry,
      std::unique_ptr<EngineTrackedGpuAllocator> device_allocator,
      std::unique_ptr<EngineTrackedPinnedAllocator> pinned_allocator,
      std::vector<std::uint64_t> pinned_identities,
      std::vector<std::uint64_t> device_identities,
      std::vector<EnginePinnedMemoryBinding> pinned_bindings,
      std::vector<EngineGpuAllocationBinding> device_bindings,
      std::uint64_t resident_weight_backing_bytes,
      std::uint64_t pinned_staging_backing_bytes) noexcept;

  std::unique_ptr<Clock> clock_;
  std::unique_ptr<EngineCudaOwnerLedgerRegistry> registry_;
  std::unique_ptr<EngineTrackedGpuAllocator> device_allocator_;
  std::unique_ptr<EngineTrackedPinnedAllocator> pinned_allocator_;
  std::vector<std::uint64_t> pinned_identities_;
  std::vector<std::uint64_t> device_identities_;
  std::vector<EnginePinnedMemoryBinding> pinned_bindings_;
  std::vector<EngineGpuAllocationBinding> device_bindings_;
  std::uint64_t resident_weight_backing_bytes_ = 0;
  std::uint64_t pinned_staging_backing_bytes_ = 0;
};

}  // namespace pih
