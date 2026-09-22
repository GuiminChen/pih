#include "pih/model/deepseek_rank_materialization_allocation_census.h"

#include <array>
#include <chrono>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool nonzero(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return true;
  return false;
}

bool valid_authority(
    const DeepSeekRankMaterializationAllocationAuthority& authority) {
  const bool host_spill =
      authority.residency == RuntimeProfileResidency::kHostSpill;
  const bool cuda_valid = authority.device_ordinal >= 0 &&
                          nonzero(authority.physical_gpu_identity) &&
                          authority.cuda_allocation_identity != 0 &&
                          nonzero(authority.cuda_owner_identity) &&
                          nonzero(authority.cuda_resource_identity);
  const bool pinned_empty = authority.pinned_registration_identity == 0 &&
                            !nonzero(authority.pinned_owner_identity) &&
                            !nonzero(authority.pinned_resource_identity) &&
                            authority.pinned_numa_node == -1;
  const bool pinned_valid = authority.pinned_registration_identity != 0 &&
                            nonzero(authority.pinned_owner_identity) &&
                            nonzero(authority.pinned_resource_identity) &&
                            authority.pinned_numa_node >= 0;
  return cuda_valid && (host_spill ? pinned_valid : pinned_empty);
}

}  // namespace

Result<Sha256Digest>
compile_deepseek_rank_materialization_allocation_authority_root(
    const DeepSeekRankMaterializationAllocationAuthority& authority) {
  if (!valid_authority(authority)) {
    return Status::InvalidArgument(
        "DeepSeek materialization allocation authority is invalid");
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:deepseek-rank-materialization-allocation-authority:v1",
      10);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_u32(
      1, static_cast<std::uint32_t>(authority.device_ordinal));
  if (status.ok()) {
    status = builder->add_u32(2, static_cast<std::uint32_t>(authority.residency));
  }
  // The full-resident canonical shape contains zero pinned digests.  Encode
  // authority digests as fixed-width bytes rather than content hashes so that
  // this explicit absence remains signed instead of being rejected by the
  // generic nonzero-content-digest helper.
  if (status.ok()) {
    status = builder->add_bytes(3, authority.physical_gpu_identity.bytes);
  }
  if (status.ok()) status = builder->add_u64(4, authority.cuda_allocation_identity);
  if (status.ok()) {
    status = builder->add_bytes(5, authority.cuda_owner_identity.bytes);
  }
  if (status.ok()) {
    status = builder->add_bytes(6, authority.cuda_resource_identity.bytes);
  }
  if (status.ok()) {
    status = builder->add_u64(7, authority.pinned_registration_identity);
  }
  if (status.ok()) {
    status = builder->add_bytes(8, authority.pinned_owner_identity.bytes);
  }
  if (status.ok()) {
    status = builder->add_bytes(9, authority.pinned_resource_identity.bytes);
  }
  if (status.ok()) {
    status = builder->add_u32(
        10, static_cast<std::uint32_t>(authority.pinned_numa_node));
  }
  if (!status.ok()) return status;
  return builder->finalize();
}

class DeepSeekRankMaterializationAllocationCensus::Clock final
    : public EngineCudaOwnerLedgerClock {
 public:
  Result<std::uint64_t> monotonic_ns() override {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto value = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           now)
                           .count();
    if (value <= 0) return Status::Internal("materialization census clock failed");
    return static_cast<std::uint64_t>(value);
  }
};

DeepSeekRankMaterializationAllocationCensus::
    ~DeepSeekRankMaterializationAllocationCensus() = default;

DeepSeekRankMaterializationAllocationCensus::
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
        std::uint64_t pinned_staging_backing_bytes) noexcept
    : clock_(std::move(clock)), registry_(std::move(registry)),
      device_allocator_(std::move(device_allocator)),
      pinned_allocator_(std::move(pinned_allocator)),
      pinned_identities_(std::move(pinned_identities)),
      device_identities_(std::move(device_identities)),
      pinned_bindings_(std::move(pinned_bindings)),
      device_bindings_(std::move(device_bindings)),
      resident_weight_backing_bytes_(resident_weight_backing_bytes),
      pinned_staging_backing_bytes_(pinned_staging_backing_bytes) {}

Result<std::unique_ptr<DeepSeekRankMaterializationAllocationCensus>>
DeepSeekRankMaterializationAllocationCensus::Create(
    const DeepSeekRankMaterializationAllocationAuthority& authority,
    std::uint64_t resident_weight_backing_bytes,
    std::uint64_t pinned_staging_backing_bytes, Allocator& device_allocator,
    RegisteredPinnedAllocator& pinned_allocator) {
  const bool host_spill =
      authority.residency == RuntimeProfileResidency::kHostSpill;
  if (!valid_authority(authority) || resident_weight_backing_bytes == 0 ||
      (host_spill ? pinned_staging_backing_bytes == 0
                  : pinned_staging_backing_bytes != 0)) {
    return Status::InvalidArgument(
        "DeepSeek materialization allocation authority is invalid");
  }
  auto clock = std::make_unique<Clock>();
  auto registry = std::make_unique<EngineCudaOwnerLedgerRegistry>(*clock);
  const std::array device_plans{EngineTrackedGpuAllocationPlan{
      authority.cuda_allocation_identity, authority.physical_gpu_identity,
      resident_weight_backing_bytes, DeepSeekWeightMaterializationPlan::kFinalAlignment}};
  auto tracked_device = EngineTrackedGpuAllocator::Create(
      device_plans, device_allocator, *registry);
  if (!tracked_device.ok()) return tracked_device.status();

  std::vector<EngineTrackedPinnedAllocationPlan> pinned_plans;
  std::vector<EnginePinnedMemoryBinding> pinned_bindings;
  std::vector<std::uint64_t> pinned_identities;
  std::unique_ptr<EngineTrackedPinnedAllocator> tracked_pinned;
  if (host_spill) {
    pinned_plans.push_back({authority.pinned_registration_identity,
                            authority.physical_gpu_identity,
                            authority.pinned_numa_node,
                            pinned_staging_backing_bytes,
                            DeepSeekExpertBundleLayout::kAlignment});
    auto created = EngineTrackedPinnedAllocator::Create(
        pinned_plans, pinned_allocator, *registry);
    if (!created.ok()) return created.status();
    tracked_pinned = std::move(*created);
    pinned_identities.push_back(authority.pinned_registration_identity);
    pinned_bindings.push_back(
        {authority.pinned_registration_identity, authority.pinned_owner_identity,
         authority.pinned_resource_identity, authority.physical_gpu_identity,
         authority.pinned_numa_node, pinned_staging_backing_bytes,
         EngineOwnedResourceLifecycleState::kActive});
  }
  std::vector<EngineGpuAllocationBinding> device_bindings{
      {authority.cuda_allocation_identity, authority.cuda_owner_identity,
       authority.cuda_resource_identity, authority.physical_gpu_identity,
       resident_weight_backing_bytes, EngineOwnedResourceLifecycleState::kActive}};
  return std::unique_ptr<DeepSeekRankMaterializationAllocationCensus>(
      new DeepSeekRankMaterializationAllocationCensus(
          std::move(clock), std::move(registry), std::move(*tracked_device),
          std::move(tracked_pinned), std::move(pinned_identities),
          {authority.cuda_allocation_identity}, std::move(pinned_bindings),
          std::move(device_bindings), resident_weight_backing_bytes,
          pinned_staging_backing_bytes));
}

Result<DeepSeekRankMaterializationAllocationCensusObservation>
DeepSeekRankMaterializationAllocationCensus::capture() const {
  auto snapshot = registry_->capture(pinned_identities_, device_identities_);
  if (!snapshot.ok()) return snapshot.status();
  auto manifest = compile_engine_cuda_owner_ledger_manifest(
      *snapshot, pinned_bindings_, device_bindings_);
  if (!manifest.ok()) return manifest.status();
  if (manifest->allocation_records.size() != 1 ||
      manifest->allocation_records.front().backing_bytes !=
          resident_weight_backing_bytes_ ||
      (host_spill() &&
       (manifest->pinned_records.size() != 1 ||
        manifest->pinned_records.front().backing_bytes !=
            pinned_staging_backing_bytes_)) ||
      (!host_spill() && (!manifest->pinned_records.empty() ||
                         nonzero(manifest->pinned_census_digest)))) {
    return Status::FailedPrecondition(
        "DeepSeek materialization allocation census drifted");
  }
  return DeepSeekRankMaterializationAllocationCensusObservation{
      manifest->sample_identity, manifest->sample_started_ns,
      manifest->sample_completed_ns, resident_weight_backing_bytes_,
      pinned_staging_backing_bytes_, manifest->allocation_census_digest,
      manifest->pinned_census_digest};
}

}  // namespace pih
