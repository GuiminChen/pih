#include "pih/model/deepseek_rank_post_exec_resource_receipt.h"
#include "deepseek_rank_capacity_test_fixture.h"

#include <gtest/gtest.h>

#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace pih {
namespace {

Sha256Digest post_exec_digest(std::uint8_t seed) {
  Sha256Digest result{};
  result.bytes.fill(static_cast<std::byte>(seed));
  return result;
}

std::vector<DeepSeekRankProcessManifest> post_exec_manifests(
    std::uint32_t count) {
  std::vector<DeepSeekRankProcessManifest> result;
  for (std::uint32_t rank = 0; rank < count; ++rank) {
    result.push_back({7, 8, count, rank, 10U + rank, 20U + rank,
                      post_exec_digest(static_cast<std::uint8_t>(rank + 1)),
                      static_cast<std::int32_t>(rank), 500});
  }
  return result;
}

DeepSeekRankSpawnResourcePlan post_exec_spawn_plan(std::uint32_t count) {
  return {count, 3, 20, 4, 200, 10, 41};
}

DeepSeekRankPostExecResourcePlan post_exec_plan() {
  return {8, 20, 200, 2, 4, 10, post_exec_digest(30)};
}

DeepSeekRankExecReady post_exec_ready(
    const DeepSeekRankProcessManifest& manifest) {
  return {{manifest.engine_epoch, manifest.worker_generation, manifest.rank,
           manifest.physical_device_identity,
           manifest.process_manifest_identity, 100U + manifest.rank,
           200U + manifest.rank, 300U + manifest.rank,
           manifest.physical_device_uuid_commitment,
           manifest.startup_device_ordinal, manifest.startup_deadline_ns},
          400U + manifest.rank};
}

DeepSeekRankPostExecResourceObservation post_exec_observation(
    const DeepSeekRankProcessManifest& manifest,
    Sha256Digest capacity_root = post_exec_digest(31),
    Sha256Digest envelope_root = post_exec_digest(30)) {
  return {manifest.engine_epoch,
          manifest.worker_generation,
          manifest.rank,
          100U + manifest.rank,
          400U + manifest.rank,
          capacity_root,
          envelope_root,
          8,
          20,
          0,
          200,
          24,
          100,
          1000,
          210,
          true};
}

DeepSeekRankSpawnResourceObservation post_exec_spawn_observation() {
  return {10,
          100,
          true,
          {{post_exec_digest(50), 5, 100},
           {post_exec_digest(51), 10, 200}},
          10,
          100,
          200,
          1000,
          100,
          1000,
          1000};
}

class PostExecProcessDriver final : public DeepSeekRankProcessDriver {
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

DeepSeekRankSpawnAuthorization post_exec_authorization(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankSpawnResourcePlan& spawn) {
  auto preflight = DeepSeekRankSpawnPreflightReceipt::Compile(
      manifests, spawn, post_exec_spawn_observation()).value();
  auto capacity = test_fixture::rank_capacity_instance(
      manifests, spawn, 90);
  return DeepSeekRankSpawnAuthorization::Create(
      manifests, std::move(capacity), preflight).value();
}

DeepSeekRankProcessSupervisor post_exec_supervisor(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankSpawnResourcePlan& spawn,
    PostExecProcessDriver& driver, bool make_ready = true) {
  auto supervisor = DeepSeekRankProcessSupervisor::Create(
      manifests, post_exec_authorization(manifests, spawn), driver).value();
  (void)supervisor.launch();
  if (make_ready) {
    std::vector<std::uint64_t> challenges;
    challenges.reserve(manifests.size());
    for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
      challenges.push_back(400U + rank);
    }
    (void)supervisor.dispatch_challenges(90, challenges);
    (void)supervisor.advance_exec_startup(499);
  }
  return supervisor;
}

std::vector<DeepSeekRankPostExecResourceReceipt> post_exec_receipts(
    const std::vector<DeepSeekRankProcessManifest>& manifests,
    const DeepSeekRankSpawnResourcePlan& spawn,
    const std::vector<DeepSeekRankPostExecResourcePlan>& plans,
    const DeepSeekRankProcessSupervisor& supervisor) {
  std::vector<DeepSeekRankPostExecResourceReceipt> receipts;
  receipts.reserve(manifests.size());
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    auto observation = post_exec_observation(
        manifests[rank], supervisor.capacity_plan_instance_root(),
        plans[rank].os_resource_envelope_root);
    receipts.push_back(DeepSeekRankPostExecResourceReceipt::Compile(
        manifests, spawn, plans[rank], *supervisor.exec_ready(rank),
        observation).value());
  }
  return receipts;
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     CompilesExactOneToFourRankReceipts) {
  EXPECT_EQ(kDeepSeekRankPostExecResourcePlanAbi,
            "pih_deepseek_rank_post_exec_resource_plan_v1");
  EXPECT_EQ(kDeepSeekRankPostExecResourceReceiptAbi,
            "pih_deepseek_rank_post_exec_resource_receipt_v1");
  for (std::uint32_t count = 1; count <= 4; ++count) {
    const auto manifests = post_exec_manifests(count);
    const auto spawn = post_exec_spawn_plan(count);
    for (std::uint32_t rank = 0; rank < count; ++rank) {
      auto receipt = DeepSeekRankPostExecResourceReceipt::Compile(
          manifests, spawn, post_exec_plan(), post_exec_ready(manifests[rank]),
          post_exec_observation(manifests[rank]));
      ASSERT_TRUE(receipt.ok()) << receipt.status().message();
      EXPECT_EQ(receipt->engine_epoch(), 7U);
      EXPECT_EQ(receipt->worker_generation(), 8U);
      EXPECT_EQ(receipt->rank(), rank);
      EXPECT_EQ(receipt->world_size(), count);
      EXPECT_EQ(receipt->process_identity(), 100U + rank);
      EXPECT_EQ(receipt->capacity_plan_instance_root(), post_exec_digest(31));
      EXPECT_EQ(receipt->os_resource_envelope_root(), post_exec_digest(30));
      EXPECT_NE(receipt->resource_plan_root(), Sha256Digest{});
      EXPECT_NE(receipt->receipt_root(), Sha256Digest{});
    }
  }
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     ExactHeadroomPassesAndEveryOneShortBoundaryRejects) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const auto ready = post_exec_ready(manifests[1]);
  auto observation = post_exec_observation(manifests[1]);
  ASSERT_TRUE(DeepSeekRankPostExecResourceReceipt::Compile(
                  manifests, spawn, post_exec_plan(), ready, observation)
                  .ok());

  observation.rlimit_nofile_soft = 23;
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, post_exec_plan(), ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
  observation = post_exec_observation(manifests[1]);
  observation.vm_max_map_count = 209;
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, post_exec_plan(), ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
  observation = post_exec_observation(manifests[1]);
  observation.task_count = 9;
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, post_exec_plan(), ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
  observation = post_exec_observation(manifests[1]);
  observation.open_fd_count = 21;
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, post_exec_plan(), ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
  observation = post_exec_observation(manifests[1]);
  observation.vma_count = 201;
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, post_exec_plan(), ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     RejectsIdentityCapacityEnvelopeAndSteadyStateDrift) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const auto ready = post_exec_ready(manifests[0]);
  auto observation = post_exec_observation(manifests[0]);

  observation.process_identity++;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
  observation = post_exec_observation(manifests[0]);
  observation.challenge_identity++;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
  observation = post_exec_observation(manifests[0]);
  observation.acknowledged_capacity_plan_instance_root = {};
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
  observation = post_exec_observation(manifests[0]);
  observation.acknowledged_os_resource_envelope_root = post_exec_digest(99);
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
  observation = post_exec_observation(manifests[0]);
  observation.scm_rights_inflight_fd_count = 1;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
  observation = post_exec_observation(manifests[0]);
  observation.non_dumpable = false;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, post_exec_plan(), ready, observation)
                   .ok());
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     RejectsSpawnProjectionDriftAndCheckedOverflow) {
  const auto manifests = post_exec_manifests(1);
  const auto ready = post_exec_ready(manifests[0]);
  const auto observation = post_exec_observation(manifests[0]);
  auto spawn = post_exec_spawn_plan(1);
  auto plan = post_exec_plan();

  plan.worker_fd_peak++;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, plan, ready, observation)
                   .ok());
  plan = post_exec_plan();
  plan.vma_emergency_reserve++;
  EXPECT_FALSE(DeepSeekRankPostExecResourceReceipt::Compile(
                   manifests, spawn, plan, ready, observation)
                   .ok());
  plan = post_exec_plan();
  plan.worker_task_peak = std::numeric_limits<std::uint64_t>::max();
  EXPECT_EQ(DeepSeekRankPostExecResourceReceipt::Compile(
                manifests, spawn, plan, ready, observation)
                .status()
                .code(),
            StatusCode::kResourceExhausted);
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     RootBindsRankPlanObservationAndCapacityAcknowledgement) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const auto ready = post_exec_ready(manifests[0]);
  const auto observation = post_exec_observation(manifests[0]);
  const auto first = DeepSeekRankPostExecResourceReceipt::Compile(
      manifests, spawn, post_exec_plan(), ready, observation).value();
  const auto replay = DeepSeekRankPostExecResourceReceipt::Compile(
      manifests, spawn, post_exec_plan(), ready, observation).value();
  auto changed_observation = observation;
  changed_observation.open_fd_count--;
  const auto observed = DeepSeekRankPostExecResourceReceipt::Compile(
      manifests, spawn, post_exec_plan(), ready, changed_observation).value();
  changed_observation = observation;
  changed_observation.acknowledged_capacity_plan_instance_root =
      post_exec_digest(32);
  const auto capacity = DeepSeekRankPostExecResourceReceipt::Compile(
      manifests, spawn, post_exec_plan(), ready, changed_observation).value();

  EXPECT_EQ(first.receipt_root(), replay.receipt_root());
  EXPECT_NE(first.receipt_root(), observed.receipt_root());
  EXPECT_NE(first.receipt_root(), capacity.receipt_root());
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     SealsExactReadyOneToFourRankReceiptSets) {
  EXPECT_EQ(kDeepSeekRankPostExecResourceSealAbi,
            "pih_deepseek_rank_post_exec_resource_seal_v1");
  for (std::uint32_t count = 1; count <= 4; ++count) {
    const auto manifests = post_exec_manifests(count);
    const auto spawn = post_exec_spawn_plan(count);
    const std::vector<DeepSeekRankPostExecResourcePlan> plans(
        count, post_exec_plan());
    PostExecProcessDriver driver;
    auto supervisor = post_exec_supervisor(manifests, spawn, driver);
    ASSERT_TRUE(supervisor.ready());
    const auto receipts = post_exec_receipts(
        manifests, spawn, plans, supervisor);

    auto seal = DeepSeekRankPostExecResourceSeal::Compile(
        supervisor, manifests, spawn, plans, receipts);
    ASSERT_TRUE(seal.ok()) << seal.status().message();
    auto replay = DeepSeekRankPostExecResourceSeal::Compile(
        supervisor, manifests, spawn, plans, receipts);
    ASSERT_TRUE(replay.ok()) << replay.status().message();
    EXPECT_EQ(seal->engine_epoch(), 7U);
    EXPECT_EQ(seal->worker_generation(), 8U);
    EXPECT_EQ(seal->world_size(), count);
    EXPECT_EQ(seal->capacity_plan_instance_root(),
              supervisor.capacity_plan_instance_root());
    EXPECT_EQ(seal->os_resource_envelope_root(), post_exec_digest(30));
    EXPECT_NE(seal->receipt_set_root(), Sha256Digest{});
    EXPECT_EQ(seal->seal_root(), replay->seal_root());
    if (count == 2) {
      EXPECT_EQ(seal->seal_root().hex(),
                "b526d17d21e42a4577194226cc9da10c4941e5508eef0587fb935c983befb1d8");
    }
  }
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     SealRejectsIncompleteReorderedAndForeignSpawnSets) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, post_exec_plan());
  PostExecProcessDriver driver;
  auto supervisor = post_exec_supervisor(manifests, spawn, driver);
  auto receipts = post_exec_receipts(manifests, spawn, plans, supervisor);

  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, spawn, plans,
                   std::span(receipts).first(1))
                   .ok());
  std::swap(receipts[0], receipts[1]);
  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, spawn, plans, receipts)
                   .ok());
  std::swap(receipts[0], receipts[1]);
  auto foreign_spawn = spawn;
  foreign_spawn.node_file_handle_reserve++;
  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, foreign_spawn, plans, receipts)
                   .ok());
  auto foreign_manifests = manifests;
  foreign_manifests[1].physical_device_uuid_commitment.bytes[0] =
      std::byte{0x7f};
  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, foreign_manifests, spawn, plans, receipts)
                   .ok());
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     SealRejectsForeignCapacityAcknowledgementAcrossRanks) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, post_exec_plan());
  PostExecProcessDriver driver;
  auto supervisor = post_exec_supervisor(manifests, spawn, driver);
  auto receipts = post_exec_receipts(manifests, spawn, plans, supervisor);
  auto foreign_observation = post_exec_observation(
      manifests[1], post_exec_digest(99));
  auto foreign_receipt = DeepSeekRankPostExecResourceReceipt::Compile(
      manifests, spawn, plans[1], *supervisor.exec_ready(1),
      foreign_observation);
  ASSERT_TRUE(foreign_receipt.ok()) << foreign_receipt.status().message();
  receipts[1] = *foreign_receipt;

  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, spawn, plans, receipts)
                   .ok());
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     SealRejectsMixedOsResourceEnvelopesAcrossRanks) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, post_exec_plan());
  plans[1].os_resource_envelope_root = post_exec_digest(98);
  PostExecProcessDriver driver;
  auto supervisor = post_exec_supervisor(manifests, spawn, driver);
  const auto receipts = post_exec_receipts(
      manifests, spawn, plans, supervisor);

  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, spawn, plans, receipts)
                   .ok());
}

TEST(DeepSeekRankPostExecResourceReceiptTest,
     SealRejectsSupervisorBeforeAllRanksAreExecReady) {
  const auto manifests = post_exec_manifests(2);
  const auto spawn = post_exec_spawn_plan(2);
  const std::vector<DeepSeekRankPostExecResourcePlan> plans(
      2, post_exec_plan());
  PostExecProcessDriver driver;
  auto supervisor = post_exec_supervisor(
      manifests, spawn, driver, false);
  std::vector<DeepSeekRankPostExecResourceReceipt> receipts;
  for (std::uint32_t rank = 0; rank < manifests.size(); ++rank) {
    receipts.push_back(DeepSeekRankPostExecResourceReceipt::Compile(
        manifests, spawn, plans[rank], post_exec_ready(manifests[rank]),
        post_exec_observation(
            manifests[rank], supervisor.capacity_plan_instance_root()))
                           .value());
  }

  EXPECT_FALSE(DeepSeekRankPostExecResourceSeal::Compile(
                   supervisor, manifests, spawn, plans, receipts)
                   .ok());
}

}  // namespace
}  // namespace pih
