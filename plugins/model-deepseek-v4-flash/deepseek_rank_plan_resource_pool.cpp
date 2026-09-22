#include "pih/model/deepseek_rank_plan_resource_pool.h"

#include <algorithm>
#include <utility>

namespace pih {

class DeepSeekRankPlanResourcePool::Lease final
    : public DeepSeekRankPlanResourceLease {
 public:
  Lease(DeepSeekRankPlanResourcePool& owner,
        DeepSeekRankPlanResourceKind kind, std::uint64_t units)
      : owner_(&owner), kind_(kind), units_(units) {}
  ~Lease() override {
    if (owner_ != nullptr) owner_->release(kind_, units_);
  }
 private:
  DeepSeekRankPlanResourcePool* owner_ = nullptr;
  DeepSeekRankPlanResourceKind kind_{};
  std::uint64_t units_ = 0;
};

std::size_t DeepSeekRankPlanResourcePool::index(
    DeepSeekRankPlanResourceKind kind) noexcept {
  return static_cast<std::size_t>(kind);
}

Result<std::unique_ptr<DeepSeekRankPlanResourcePool>>
DeepSeekRankPlanResourcePool::Create(
    std::uint64_t engine_epoch, std::uint32_t rank,
    std::uint32_t world_size,
    std::span<const DeepSeekRankPlanResourceCapacity> capacities) {
  if (engine_epoch == 0 || world_size == 0 || world_size > 4 ||
      rank >= world_size || capacities.size() != kKindCount) {
    return Status::InvalidArgument(
        "DeepSeek rank resource pool identity or cardinality is invalid");
  }
  std::array<std::uint64_t, kKindCount> frozen{};
  std::array<bool, kKindCount> present{};
  for (const auto& capacity : capacities) {
    const auto ordinal = index(capacity.kind);
    if (ordinal >= kKindCount || present[ordinal]) {
      return Status::InvalidArgument(
          "DeepSeek rank resource capacity kind is invalid or duplicated");
    }
    present[ordinal] = true;
    frozen[ordinal] = capacity.units;
  }
  if (std::ranges::any_of(present, [](bool value) { return !value; })) {
    return Status::InvalidArgument(
        "DeepSeek rank resource capacity set is incomplete");
  }
  return std::unique_ptr<DeepSeekRankPlanResourcePool>(
      new DeepSeekRankPlanResourcePool(engine_epoch, rank, world_size,
                                       frozen));
}

Result<std::unique_ptr<DeepSeekRankPlanResourceLease>>
DeepSeekRankPlanResourcePool::reserve(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    const DeepSeekStagePlan& stage,
    DeepSeekRankPlanResourceRequirement requirement) {
  const auto ordinal = index(requirement.kind);
  if (descriptor.engine_epoch != engine_epoch_ || descriptor.plan_sequence == 0 ||
      stage.rank != rank_ || rank_ >= world_size_ ||
      ordinal >= kKindCount || requirement.units == 0) {
    return Status::FailedPrecondition(
        "DeepSeek rank resource request identity is invalid");
  }
  if (available_[ordinal] < requirement.units) {
    return Status::ResourceExhausted(
        "DeepSeek rank plan resource capacity is exhausted");
  }
  available_[ordinal] -= requirement.units;
  return std::unique_ptr<DeepSeekRankPlanResourceLease>(
      new Lease(*this, requirement.kind, requirement.units));
}

void DeepSeekRankPlanResourcePool::release(
    DeepSeekRankPlanResourceKind kind, std::uint64_t units) noexcept {
  const auto ordinal = index(kind);
  if (ordinal >= kKindCount || units > capacities_[ordinal] - available_[ordinal]) {
    std::terminate();
  }
  available_[ordinal] += units;
}

std::uint64_t DeepSeekRankPlanResourcePool::available(
    DeepSeekRankPlanResourceKind kind) const noexcept {
  const auto ordinal = index(kind);
  return ordinal < kKindCount ? available_[ordinal] : 0;
}

std::uint64_t DeepSeekRankPlanResourcePool::capacity(
    DeepSeekRankPlanResourceKind kind) const noexcept {
  const auto ordinal = index(kind);
  return ordinal < kKindCount ? capacities_[ordinal] : 0;
}

bool DeepSeekRankPlanResourcePool::fully_released() const noexcept {
  return available_ == capacities_;
}

}  // namespace pih
