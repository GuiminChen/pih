#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "deepseek_rank_capacity_test_fixture.h"
#include "pih/model/deepseek_rank_model_startup_plan.h"

namespace pih::test_fixture {

inline std::vector<DeepSeekRankProcessManifest> startup_manifests(
    std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    result.push_back(
        {7, 8, count, rank, 10U + rank, 20U + rank,
         rank_capacity_digest(static_cast<std::uint8_t>(rank + 1)),
         static_cast<std::int32_t>(rank), 500});
  }
  return result;
}

inline DeepSeekRankSpawnResourcePlan startup_spawn_plan(
    std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

inline DeepSeekRankSpawnResourceObservation startup_spawn_observation() {
  return {10,
          100,
          true,
          {{rank_capacity_digest(50), 5, 100},
           {rank_capacity_digest(51), 10, 200}},
          10,
          100,
          200,
          1000,
          100,
          1000,
          1000};
}

inline DeepSeekRankPostExecResourcePlan startup_resource_plan() {
  return {8, 20, 200, 2, 4, 10, rank_capacity_digest(30)};
}

class StartupProcessDriver final : public DeepSeekRankProcessDriver {
 public:
  Result<DeepSeekRankProcessHandle> spawn(
      const DeepSeekRankProcessManifest& manifest) override {
    return DeepSeekRankProcessHandle{100U + manifest.rank,
                                     200U + manifest.rank,
                                     300U + manifest.rank};
  }
  Result<DeepSeekRankProcessObservation> observe(
      const DeepSeekRankProcessHandle&) override {
    return DeepSeekRankProcessObservation::kRunning;
  }
  Status terminate(const DeepSeekRankProcessHandle&) override {
    return Status::Ok();
  }
  Status send_challenge(
      const DeepSeekRankProcessHandle&,
      const DeepSeekRankExecChallenge& challenge) override {
    challenges.push_back(challenge);
    return Status::Ok();
  }
  Result<std::optional<DeepSeekRankExecReady>> poll_ready(
      const DeepSeekRankProcessHandle& handle) override {
    for (const auto& challenge : challenges) {
      if (challenge.handle.process_identity != handle.process_identity) {
        continue;
      }
      const auto& manifest = challenge.manifest;
      return std::optional<DeepSeekRankExecReady>{{
          {manifest.engine_epoch, manifest.worker_generation, manifest.rank,
           manifest.physical_device_identity,
           manifest.process_manifest_identity, handle.process_identity,
           handle.pidfd_identity, handle.control_identity,
           manifest.physical_device_uuid_commitment,
           manifest.startup_device_ordinal, manifest.startup_deadline_ns},
          challenge.challenge_identity}};
    }
    return std::optional<DeepSeekRankExecReady>{};
  }

  std::vector<DeepSeekRankExecChallenge> challenges;
};

class DeepSeekRankStartupFixture final {
 public:
  static std::unique_ptr<DeepSeekRankStartupFixture> Create(
      std::uint32_t count,
      const RuntimeEvidenceProjection* supplied_evidence = nullptr,
      bool dspark_enabled = false,
      std::uint64_t deployment_generation = 17) {
    auto result = std::unique_ptr<DeepSeekRankStartupFixture>(
        new DeepSeekRankStartupFixture);
    result->manifests_ = startup_manifests(count);
    result->spawn_ = startup_spawn_plan(count);
    result->admission_.emplace(rank_capacity_admission(
        static_cast<std::uint8_t>(count), true, true,
        std::span<const std::int32_t>{}, supplied_evidence, dspark_enabled));
    result->bootstrap_.emplace(rank_capacity_bootstrap(
        *result->admission_, deployment_generation));
    auto capacity_instance = DeepSeekRankCapacityPlanInstance::Compile(
        *result->admission_, *result->bootstrap_, result->manifests_,
        result->spawn_, 90)
                                 .value();
    auto preflight = DeepSeekRankSpawnPreflightReceipt::Compile(
        result->manifests_, result->spawn_, startup_spawn_observation())
                         .value();
    auto authorization = DeepSeekRankSpawnAuthorization::Create(
        result->manifests_, std::move(capacity_instance), preflight)
                             .value();
    result->supervisor_.emplace(
        DeepSeekRankProcessSupervisor::Create(
            result->manifests_, std::move(authorization), result->driver_)
            .value());
    if (!result->supervisor_->launch().ok()) return {};
    std::vector<std::uint64_t> challenges;
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      challenges.push_back(400U + rank);
    }
    if (!result->supervisor_->dispatch_challenges(90, challenges).ok()) {
      return {};
    }
    if (!result->supervisor_->advance_exec_startup(499).ok()) return {};

    result->resource_plans_.assign(count, startup_resource_plan());
    std::vector<DeepSeekRankPostExecResourceReceipt> receipts;
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      const auto& manifest = result->manifests_[rank];
      DeepSeekRankPostExecResourceObservation observation{
          manifest.engine_epoch,
          manifest.worker_generation,
          rank,
          100U + rank,
          400U + rank,
          result->supervisor_->capacity_plan_instance_root(),
          rank_capacity_digest(30),
          8,
          20,
          0,
          200,
          24,
          100,
          1000,
          210,
          true};
      receipts.push_back(DeepSeekRankPostExecResourceReceipt::Compile(
                             result->manifests_, result->spawn_,
                             result->resource_plans_[rank],
                             *result->supervisor_->exec_ready(rank), observation)
                             .value());
    }
    result->seal_.emplace(DeepSeekRankPostExecResourceSeal::Compile(
                              *result->supervisor_, result->manifests_,
                              result->spawn_, result->resource_plans_, receipts)
                              .value());
    std::array<std::byte, kRuntimeReadinessNonceBytes> nonce{};
    nonce.fill(std::byte{0x33});
    result->readiness_.emplace(issue_runtime_profile_readiness_receipt(
        *result->admission_, *result->bootstrap_, nonce,
        result->bootstrap_->authority_snapshot_root(),
        result->supervisor_->capacity_plan_instance_root())
                                   .value());
    return result;
  }

  Result<DeepSeekRankModelStartupPlan> compile_model_startup(
      std::uint64_t deadline = 900) const {
    auto pipeline = DeepSeekPipelinePlan::Create(
        static_cast<std::uint32_t>(manifests_.size()),
        admission_->dspark_enabled());
    if (!pipeline.ok()) return pipeline.status();
    auto capacity = DeepSeekPipelineCapacity::Create(
        static_cast<std::uint32_t>(manifests_.size()), 17, 9, 3,
        admission_->dspark_enabled());
    if (!capacity.ok()) return capacity.status();
    return DeepSeekRankModelStartupPlan::Compile(
        *bootstrap_, *readiness_, *supervisor_, manifests_, *seal_, *pipeline,
        *capacity, deadline);
  }

  [[nodiscard]] const RuntimeEngineAdmission& admission() const {
    return *admission_;
  }
  [[nodiscard]] const std::vector<DeepSeekRankProcessManifest>& manifests()
      const noexcept {
    return manifests_;
  }
  [[nodiscard]] const DeepSeekRankSpawnResourcePlan& spawn_plan()
      const noexcept {
    return spawn_;
  }
  [[nodiscard]] const DeepSeekRankProcessSupervisor& supervisor()
      const noexcept {
    return *supervisor_;
  }
  [[nodiscard]] DeepSeekRankProcessSupervisor& mutable_supervisor()
      noexcept {
    return *supervisor_;
  }
  [[nodiscard]] const std::vector<DeepSeekRankPostExecResourcePlan>&
  resource_plans() const noexcept {
    return resource_plans_;
  }
  [[nodiscard]] const DeepSeekRankPostExecResourceSeal& resource_seal()
      const noexcept {
    return *seal_;
  }

 private:
  DeepSeekRankStartupFixture() = default;
  std::vector<DeepSeekRankProcessManifest> manifests_;
  DeepSeekRankSpawnResourcePlan spawn_;
  StartupProcessDriver driver_;
  std::optional<RuntimeEngineAdmission> admission_;
  std::optional<RuntimeProfileSupervisorBootstrapManifest> bootstrap_;
  std::optional<DeepSeekRankProcessSupervisor> supervisor_;
  std::vector<DeepSeekRankPostExecResourcePlan> resource_plans_;
  std::optional<DeepSeekRankPostExecResourceSeal> seal_;
  std::optional<RuntimeProfileReadinessReceipt> readiness_;
};

}  // namespace pih::test_fixture
