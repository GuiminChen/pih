#include "pih/model/engine_manifest_tracking_allocator.h"

#include <set>

namespace pih {
namespace {

bool zero_tracking_digest(const Sha256Digest& digest) {
  for (const auto value : digest.bytes)
    if (value != std::byte{0}) return false;
  return true;
}

bool same_allocation(const Allocation& left, const Allocation& right) {
  return left.data == right.data && left.bytes == right.bytes &&
         left.alignment == right.alignment &&
         left.generation == right.generation && left.device == right.device;
}

}  // namespace

Result<std::unique_ptr<EngineTrackedGpuAllocator>>
EngineTrackedGpuAllocator::Create(
    std::span<const EngineTrackedGpuAllocationPlan> plans,
    Allocator& allocator, EngineCudaOwnerLedgerRegistry& registry) {
  if (plans.empty())
    return Status::InvalidArgument("tracked GPU allocation plan is empty");
  std::set<std::uint64_t> identities;
  for (const auto& plan : plans) {
    if (plan.allocation_identity == 0 ||
        zero_tracking_digest(plan.physical_gpu_identity) || plan.bytes == 0 ||
        plan.alignment == 0 ||
        (plan.alignment & (plan.alignment - 1)) != 0 ||
        !identities.insert(plan.allocation_identity).second)
      return Status::InvalidArgument("tracked GPU allocation plan is invalid");
  }
  return std::unique_ptr<EngineTrackedGpuAllocator>(
      new EngineTrackedGpuAllocator(
          std::vector<EngineTrackedGpuAllocationPlan>(plans.begin(), plans.end()),
          allocator, registry));
}

EngineTrackedGpuAllocator::~EngineTrackedGpuAllocator() {
  std::lock_guard lock(mutex_);
  if (!active_.empty()) registry_->poison("tracked GPU owners leaked");
}

Result<Allocation> EngineTrackedGpuAllocator::allocate(
    std::uint64_t bytes, std::uint64_t alignment) {
  std::lock_guard lock(mutex_);
  if (next_plan_ >= plans_.size() || plans_[next_plan_].bytes != bytes ||
      plans_[next_plan_].alignment != alignment) {
    registry_->poison("tracked GPU allocation order drifted");
    return Status::FailedPrecondition("tracked GPU allocation order drifted");
  }
  const auto& plan = plans_[next_plan_];
  auto allocation = allocator_->allocate(bytes, alignment);
  if (!allocation.ok()) return allocation.status();
  if (allocation->generation == 0 || allocation->data == nullptr ||
      allocation->bytes != bytes || allocation->alignment != alignment) {
    allocator_->deallocate(*allocation);
    registry_->poison("tracked GPU allocator returned invalid extent");
    return Status::Internal("tracked GPU allocator returned invalid extent");
  }
  auto status = registry_->register_allocation(
      plan.allocation_identity, plan.physical_gpu_identity, plan.bytes);
  if (!status.ok()) {
    allocator_->deallocate(*allocation);
    return status;
  }
  active_.emplace(allocation->generation,
                  Active{*allocation, plan.allocation_identity});
  ++next_plan_;
  return allocation;
}

void EngineTrackedGpuAllocator::deallocate(Allocation allocation) noexcept {
  std::lock_guard lock(mutex_);
  const auto found = active_.find(allocation.generation);
  if (found == active_.end() ||
      !same_allocation(found->second.allocation, allocation)) {
    registry_->poison("tracked GPU allocation release drifted");
    return;
  }
  allocator_->deallocate(found->second.allocation);
  const auto status = registry_->release_allocation(found->second.identity);
  active_.erase(found);
  if (!status.ok()) registry_->poison("tracked GPU owner release failed");
}

Result<std::unique_ptr<EngineTrackedPinnedAllocator>>
EngineTrackedPinnedAllocator::Create(
    std::span<const EngineTrackedPinnedAllocationPlan> plans,
    RegisteredPinnedAllocator& allocator,
    EngineCudaOwnerLedgerRegistry& registry) {
  if (plans.empty())
    return Status::InvalidArgument("tracked pinned allocation plan is empty");
  std::set<std::uint64_t> identities;
  for (const auto& plan : plans) {
    if (plan.registration_identity == 0 ||
        zero_tracking_digest(plan.physical_gpu_identity) ||
        plan.numa_node < 0 || plan.bytes == 0 || plan.alignment == 0 ||
        (plan.alignment & (plan.alignment - 1)) != 0 ||
        !identities.insert(plan.registration_identity).second)
      return Status::InvalidArgument(
          "tracked pinned allocation plan is invalid");
  }
  return std::unique_ptr<EngineTrackedPinnedAllocator>(
      new EngineTrackedPinnedAllocator(
          std::vector<EngineTrackedPinnedAllocationPlan>(plans.begin(),
                                                          plans.end()),
          allocator, registry));
}

EngineTrackedPinnedAllocator::~EngineTrackedPinnedAllocator() {
  std::lock_guard lock(mutex_);
  if (!active_.empty()) registry_->poison("tracked pinned owners leaked");
}

Result<Allocation> EngineTrackedPinnedAllocator::allocate(
    std::uint64_t bytes, std::uint64_t alignment) {
  std::lock_guard lock(mutex_);
  if (next_plan_ >= plans_.size() || plans_[next_plan_].bytes != bytes ||
      plans_[next_plan_].alignment != alignment) {
    registry_->poison("tracked pinned allocation order drifted");
    return Status::FailedPrecondition(
        "tracked pinned allocation order drifted");
  }
  const auto& plan = plans_[next_plan_];
  auto allocation = allocator_->allocate(bytes, alignment);
  if (!allocation.ok()) return allocation.status();
  if (allocation->generation == 0 || allocation->data == nullptr ||
      allocation->bytes != bytes || allocation->alignment != alignment ||
      allocation->device != Device::Cpu()) {
    allocator_->deallocate(*allocation);
    registry_->poison("tracked pinned allocator returned invalid extent");
    return Status::Internal(
        "tracked pinned allocator returned invalid extent");
  }
  auto status = registry_->register_pinned(
      plan.registration_identity, plan.physical_gpu_identity, plan.numa_node,
      plan.bytes);
  if (!status.ok()) {
    allocator_->deallocate(*allocation);
    return status;
  }
  active_.emplace(allocation->generation,
                  Active{*allocation, plan.registration_identity});
  ++next_plan_;
  return allocation;
}

void EngineTrackedPinnedAllocator::deallocate(Allocation allocation) noexcept {
  std::lock_guard lock(mutex_);
  const auto found = active_.find(allocation.generation);
  if (found == active_.end() ||
      !same_allocation(found->second.allocation, allocation)) {
    registry_->poison("tracked pinned allocation release drifted");
    return;
  }
  allocator_->deallocate(found->second.allocation);
  const auto status = registry_->release_pinned(found->second.identity);
  active_.erase(found);
  if (!status.ok()) registry_->poison("tracked pinned owner release failed");
}

}  // namespace pih
