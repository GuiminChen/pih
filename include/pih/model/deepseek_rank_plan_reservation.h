#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "pih/model/deepseek_pipeline_transaction.h"
#include "pih/model/deepseek_rank_weight_disposition_manifest.h"

namespace pih {

enum class DeepSeekRankPlanResourceKind : std::uint8_t {
  kIncomingBoundary,
  kOutgoingActivation,
  kControlSlot,
  kAttentionStateTransaction,
  kVerifyPrefixWorkspaceBytes,
  kOperatorWorkspaceBytes,
  kExpertWorkspaceBytes,
  kPagerForwardProgressSlot,
  kPinnedStagingCredit,
  kOutputSamplingCredit,
};

struct DeepSeekRankPlanResourceRequirement final {
  DeepSeekRankPlanResourceKind kind{};
  std::uint64_t units = 0;
  bool operator==(const DeepSeekRankPlanResourceRequirement&) const = default;
};

class DeepSeekRankPlanResourceLease {
 public:
  virtual ~DeepSeekRankPlanResourceLease() = default;
};

class DeepSeekRankPlanResourceProvider {
 public:
  virtual ~DeepSeekRankPlanResourceProvider() = default;
  virtual Result<std::unique_ptr<DeepSeekRankPlanResourceLease>> reserve(
      const DeepSeekPipelinePlanDescriptor& descriptor,
      const DeepSeekStagePlan& stage,
      DeepSeekRankPlanResourceRequirement requirement) = 0;
};

enum class DeepSeekRankPlanReservationState : std::uint8_t {
  kPrepared,
  kCommitted,
  kReleased,
};

class DeepSeekRankPlanReservation final {
 public:
  static Result<DeepSeekRankPlanReservation> Prepare(
      DeepSeekPipelinePlanDescriptor descriptor,
      DeepSeekStagePlan stage,
      std::uint32_t world_size,
      DeepSeekRoutedExpertResidency expert_residency,
      std::uint64_t verify_prefix_workspace_bytes,
      std::uint64_t operator_workspace_bytes,
      std::uint64_t expert_workspace_bytes,
      DeepSeekRankPlanResourceProvider& provider);

  DeepSeekRankPlanReservation(const DeepSeekRankPlanReservation&) = delete;
  DeepSeekRankPlanReservation& operator=(const DeepSeekRankPlanReservation&) = delete;
  DeepSeekRankPlanReservation(DeepSeekRankPlanReservation&& other) noexcept;
  DeepSeekRankPlanReservation& operator=(
      DeepSeekRankPlanReservation&& other) noexcept;
  ~DeepSeekRankPlanReservation();

  [[nodiscard]] Status validate_commit() const;
  Status commit();
  void release() noexcept;
  [[nodiscard]] DeepSeekRankPlanReservationState state() const noexcept {
    return state_;
  }
  [[nodiscard]] const std::vector<DeepSeekRankPlanResourceRequirement>&
  requirements() const noexcept { return requirements_; }

 private:
  DeepSeekRankPlanReservation(
      std::vector<DeepSeekRankPlanResourceRequirement> requirements,
      std::vector<std::unique_ptr<DeepSeekRankPlanResourceLease>> leases)
      : requirements_(std::move(requirements)), leases_(std::move(leases)) {}

  std::vector<DeepSeekRankPlanResourceRequirement> requirements_;
  std::vector<std::unique_ptr<DeepSeekRankPlanResourceLease>> leases_;
  DeepSeekRankPlanReservationState state_ =
      DeepSeekRankPlanReservationState::kPrepared;
};

}  // namespace pih
