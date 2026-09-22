#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <span>

#include "pih/model/deepseek_rank_plan_reservation.h"

namespace pih {

struct DeepSeekRankPlanResourceCapacity final {
  DeepSeekRankPlanResourceKind kind{};
  std::uint64_t units = 0;
  bool operator==(const DeepSeekRankPlanResourceCapacity&) const = default;
};

class DeepSeekRankPlanResourcePool final
    : public DeepSeekRankPlanResourceProvider {
 public:
  static constexpr std::size_t kKindCount = 10;
  static Result<std::unique_ptr<DeepSeekRankPlanResourcePool>> Create(
      std::uint64_t engine_epoch, std::uint32_t rank,
      std::uint32_t world_size,
      std::span<const DeepSeekRankPlanResourceCapacity> capacities);

  DeepSeekRankPlanResourcePool(const DeepSeekRankPlanResourcePool&) = delete;
  DeepSeekRankPlanResourcePool& operator=(
      const DeepSeekRankPlanResourcePool&) = delete;

  Result<std::unique_ptr<DeepSeekRankPlanResourceLease>> reserve(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      const DeepSeekStagePlan& stage,
      DeepSeekRankPlanResourceRequirement requirement) override;
  [[nodiscard]] std::uint64_t available(
      DeepSeekRankPlanResourceKind kind) const noexcept;
  [[nodiscard]] std::uint64_t capacity(
      DeepSeekRankPlanResourceKind kind) const noexcept;
  [[nodiscard]] bool fully_released() const noexcept;

 private:
  class Lease;
  DeepSeekRankPlanResourcePool(
      std::uint64_t engine_epoch, std::uint32_t rank,
      std::uint32_t world_size,
      std::array<std::uint64_t, kKindCount> capacities)
      : engine_epoch_(engine_epoch), rank_(rank), world_size_(world_size),
        capacities_(capacities), available_(capacities) {}
  static std::size_t index(DeepSeekRankPlanResourceKind kind) noexcept;
  void release(DeepSeekRankPlanResourceKind kind,
               std::uint64_t units) noexcept;

  std::uint64_t engine_epoch_ = 0;
  std::uint32_t rank_ = 0;
  std::uint32_t world_size_ = 0;
  std::array<std::uint64_t, kKindCount> capacities_{};
  std::array<std::uint64_t, kKindCount> available_{};
};

}  // namespace pih
