#include "pih/model/deepseek_rank_artifact_transfer_transaction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_rank_startup_test_fixture.h"
#include "deepseek_target_artifact_test_fixture.h"
#include "pih/model/deepseek_controller_artifact_catalog.h"
#include "pih/model/runtime_evidence_projection.h"

namespace pih {
namespace {

DeepSeekRuntimeArtifactEvidenceManifest transfer_evidence(
    const Sha256Digest& artifact_root) {
  return DeepSeekRuntimeArtifactEvidenceManifest::Create(
             RuntimeProfileGpuFamily::kH100Pcie80GiB, 1,
             RuntimeProfileResidency::kHostSpill,
             {test_fixture::rank_capacity_digest(1),
              test_fixture::rank_capacity_digest(2),
              test_fixture::rank_capacity_digest(3),
              test_fixture::rank_capacity_digest(5),
              test_fixture::rank_capacity_digest(6),
              test_fixture::rank_capacity_digest(7)},
             artifact_root,
             {test_fixture::rank_capacity_digest(61),
              test_fixture::rank_capacity_digest(62),
              test_fixture::rank_capacity_digest(63)})
      .value();
}

RuntimeEvidenceProjection transfer_projection(
    const DeepSeekRuntimeArtifactEvidenceManifest& evidence) {
  return RuntimeEvidenceProjection::Create(
             RuntimeEvidenceBundleState::kComplete,
             RuntimeEvidenceReplayState::kPassed,
             RuntimeHardwareQualificationState::kSupported,
             {test_fixture::rank_capacity_digest(30),
              test_fixture::rank_capacity_digest(31),
              test_fixture::rank_capacity_digest(32),
              evidence.domain_decision_roots_digest(),
              test_fixture::rank_capacity_digest(34)})
      .value();
}

class FakeTransferOperations final
    : public DeepSeekRankArtifactTransferOperations {
 public:
  Status send_manifest(
      std::uint32_t,
      std::span<const std::byte> frame) override {
    ++manifest_attempts;
    if (manifest_unavailable_once && manifest_attempts == 1) {
      return Status::Unavailable("manifest backpressure");
    }
    if (frame.size() != kDeepSeekRankArtifactTransferManifestBytes) {
      return Status::InvalidArgument("manifest frame size");
    }
    ++manifest_accepts;
    return Status::Ok();
  }

  Status send_descriptor_batch(
      const DeepSeekRankArtifactDescriptorBatchView& batch) override {
    ++batch_attempts;
    if (batch_unavailable_once && batch_attempts == 1) {
      first_attempt_batch_root = batch.descriptor_batch_root;
      return Status::Unavailable("descriptor backpressure");
    }
    if (fatal_batch_send) return Status::Internal("descriptor send failed");
    if (first_attempt_batch_root &&
        *first_attempt_batch_root != batch.descriptor_batch_root) {
      return Status::FailedPrecondition("retry batch identity changed");
    }
    auto recomputed_batch =
        compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
            batch.rank, batch.batch_index, batch.expectations);
    if (!recomputed_batch.ok()) return recomputed_batch.status();
    std::vector<DeepSeekRankArtifactDescriptorExpectation> adopted_candidate =
        adopted_expectations;
    adopted_candidate.insert(adopted_candidate.end(),
                             batch.expectations.begin(),
                             batch.expectations.end());
    auto recomputed_adopted =
        compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
            batch.rank, adopted_candidate);
    if (!recomputed_adopted.ok()) return recomputed_adopted.status();
    if (*recomputed_batch != batch.descriptor_batch_root ||
        *recomputed_adopted != batch.adopted_descriptor_set_root) {
      return Status::FailedPrecondition(
          "worker descriptor canonicalization differs");
    }
    adopted_expectations = std::move(adopted_candidate);
    ++batch_accepts;
    DeepSeekRankArtifactTransferAckFields fields{
        batch.engine_epoch,
        batch.worker_generation,
        batch.world_size,
        batch.rank,
        batch.batch_index,
        batch.batch_count,
        batch.first_descriptor_ordinal,
        static_cast<std::uint32_t>(batch.descriptors.size()),
        batch.cumulative_descriptor_count,
        batch.final_batch,
        batch.process_manifest_identity,
        batch.process_identity,
        batch.pidfd_identity,
        batch.control_identity,
        batch.challenge_identity,
        batch.transfer_manifest_root,
        batch.artifact_admission_binding_root,
        batch.transfer_transaction_root,
        *recomputed_batch,
        *recomputed_adopted};
    if (foreign_ack) ++fields.challenge_identity;
    auto ack = DeepSeekRankArtifactTransferAck::Create(fields);
    if (!ack.ok()) return ack.status();
    pending_ack = encode_deepseek_rank_artifact_transfer_ack(*ack);
    return Status::Ok();
  }

  Result<std::uint64_t> monotonic_now_ns() override { return now_ns; }

  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>>>
  poll_ack(std::uint32_t) override {
    ++ack_polls;
    if (!pending_ack) {
      return std::optional<std::array<
          std::byte, kDeepSeekRankArtifactTransferAckBytes>>{};
    }
    auto result = pending_ack;
    pending_ack.reset();
    return result;
  }

  Status abort_generation(
      std::uint64_t engine_epoch, std::uint64_t worker_generation,
      const Status& cause) override {
    ++abort_count;
    aborted_epoch = engine_epoch;
    aborted_generation = worker_generation;
    abort_cause = cause.code();
    return Status::Ok();
  }

  std::uint64_t now_ns = 100;
  bool manifest_unavailable_once = false;
  bool batch_unavailable_once = false;
  bool fatal_batch_send = false;
  bool foreign_ack = false;
  std::uint32_t manifest_attempts = 0;
  std::uint32_t manifest_accepts = 0;
  std::uint32_t batch_attempts = 0;
  std::uint32_t batch_accepts = 0;
  std::uint32_t ack_polls = 0;
  std::uint32_t abort_count = 0;
  std::uint64_t aborted_epoch = 0;
  std::uint64_t aborted_generation = 0;
  StatusCode abort_cause = StatusCode::kOk;
  std::optional<Sha256Digest> first_attempt_batch_root;
  std::vector<DeepSeekRankArtifactDescriptorExpectation>
      adopted_expectations;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>> pending_ack;
};

TEST(DeepSeekRankArtifactTransferTransactionTest,
     SharedCanonicalizationCoversSixteenPlusOneBatchBoundary) {
  std::vector<DeepSeekRankArtifactDescriptorExpectation> expectations;
  for (std::uint32_t ordinal = 0; ordinal < 17; ++ordinal) {
    expectations.push_back(
        {ordinal,
         "shard-" + std::to_string(ordinal) + ".safetensors",
         {100U + ordinal, 200U + ordinal, 300U + ordinal,
          400 + ordinal, ordinal},
         ArtifactImmutabilityMode::kUncalibrated,
         {}});
  }
  auto first =
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          0, 0, std::span<const DeepSeekRankArtifactDescriptorExpectation>(
                    expectations)
                    .first(16));
  auto second =
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          0, 1, std::span<const DeepSeekRankArtifactDescriptorExpectation>(
                    expectations)
                    .subspan(16));
  auto adopted =
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          0, expectations);
  ASSERT_TRUE(first.ok()) << first.status().message();
  ASSERT_TRUE(second.ok()) << second.status().message();
  ASSERT_TRUE(adopted.ok()) << adopted.status().message();
  EXPECT_NE(*first, *second);
  EXPECT_NE(*first, *adopted);

  auto reordered = expectations;
  std::swap(reordered[15], reordered[16]);
  EXPECT_FALSE(
      compile_deepseek_rank_artifact_transfer_adopted_descriptor_set_root(
          0, reordered)
          .ok());
  EXPECT_FALSE(
      compile_deepseek_rank_artifact_transfer_descriptor_batch_root(
          0, 1,
          std::span<const DeepSeekRankArtifactDescriptorExpectation>(
              expectations)
              .first(1))
          .ok());
}

Result<DeepSeekRankArtifactTransferTransaction> make_transaction(
    const std::filesystem::path& root,
    FakeTransferOperations& operations) {
  const auto artifact_root = test_fixture::reduced_target_artifact_root();
  auto evidence = transfer_evidence(artifact_root);
  auto projection = transfer_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  if (!startup) return Status::Internal("startup fixture failed");
  auto model_startup = startup->compile_model_startup();
  if (!model_startup.ok()) return model_startup.status();
  auto binding = DeepSeekRuntimeArtifactAdmissionBinding::Issue(
      startup->admission(), evidence);
  if (!binding.ok()) return binding.status();
  test_fixture::write_reduced_target_generation_fixture(root);
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
      root, artifact_root, *pipeline, 1024,
      ArtifactImmutabilityMode::kUncalibrated);
  if (!catalog.ok()) return catalog.status();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(*catalog);
  if (!handoff.ok()) return handoff.status();
  auto plan = DeepSeekRankArtifactTransferPlan::Compile(
      std::move(*model_startup), std::move(*binding), std::move(*handoff));
  if (!plan.ok()) return plan.status();
  return DeepSeekRankArtifactTransferTransaction::Create(
      std::move(*plan), operations);
}

TEST(DeepSeekRankArtifactTransferTransactionTest,
     RetainsPlanUntilExactAcknowledgementCompletes) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-transaction-test";
  std::filesystem::remove_all(root);
  FakeTransferOperations operations;
  {
    auto transaction = make_transaction(root, operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_EQ(kDeepSeekRankArtifactTransferTransactionAbi,
              "pih_deepseek_rank_artifact_transfer_transaction_v1");
    EXPECT_NE(transaction->transaction_root(), Sha256Digest{});
    EXPECT_TRUE(transaction->retains_transfer_antecedents());
    auto status = transaction->advance();
    EXPECT_EQ(status.code(), StatusCode::kUnavailable);
    EXPECT_TRUE(transaction->manifest_accepted(0));
    EXPECT_EQ(transaction->acknowledged_descriptor_count(0), 0U);
    EXPECT_FALSE(transaction->complete());
    EXPECT_TRUE(transaction->retains_transfer_antecedents());

    status = transaction->advance();
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_TRUE(transaction->complete());
    EXPECT_FALSE(transaction->poisoned());
    EXPECT_EQ(transaction->acknowledged_descriptor_count(0), 1U);
    EXPECT_TRUE(transaction->retains_transfer_antecedents());
    EXPECT_TRUE(transaction->advance().ok());
    EXPECT_EQ(operations.manifest_accepts, 1U);
    EXPECT_EQ(operations.batch_accepts, 1U);
    EXPECT_EQ(operations.ack_polls, 1U);
    EXPECT_EQ(operations.abort_count, 0U);
    EXPECT_THROW((void)transaction->manifest_accepted(1), std::out_of_range);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferTransactionTest,
     RetriesUnacceptedManifestAndExactSameBatchWithoutDuplication) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-backpressure-test";
  std::filesystem::remove_all(root);
  FakeTransferOperations operations;
  operations.manifest_unavailable_once = true;
  operations.batch_unavailable_once = true;
  {
    auto transaction = make_transaction(root, operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_EQ(transaction->advance().code(), StatusCode::kUnavailable);
    EXPECT_FALSE(transaction->manifest_accepted(0));
    EXPECT_EQ(transaction->advance().code(), StatusCode::kUnavailable);
    EXPECT_TRUE(transaction->manifest_accepted(0));
    EXPECT_EQ(transaction->advance().code(), StatusCode::kUnavailable);
    EXPECT_TRUE(transaction->advance().ok());
    EXPECT_TRUE(transaction->complete());
    EXPECT_EQ(operations.manifest_attempts, 2U);
    EXPECT_EQ(operations.manifest_accepts, 1U);
    EXPECT_EQ(operations.batch_attempts, 2U);
    EXPECT_EQ(operations.batch_accepts, 1U);
    EXPECT_EQ(operations.abort_count, 0U);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferTransactionTest,
     ForeignAcknowledgementPoisonsAndAbortsGenerationExactlyOnce) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-foreign-ack-test";
  std::filesystem::remove_all(root);
  FakeTransferOperations operations;
  operations.foreign_ack = true;
  {
    auto transaction = make_transaction(root, operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_EQ(transaction->advance().code(), StatusCode::kUnavailable);
    EXPECT_FALSE(transaction->advance().ok());
    EXPECT_TRUE(transaction->poisoned());
    EXPECT_FALSE(transaction->complete());
    EXPECT_TRUE(transaction->retains_transfer_antecedents());
    EXPECT_EQ(operations.abort_count, 1U);
    EXPECT_EQ(operations.aborted_epoch, 7U);
    EXPECT_EQ(operations.aborted_generation, 8U);
    EXPECT_FALSE(transaction->advance().ok());
    EXPECT_EQ(operations.abort_count, 1U);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferTransactionTest,
     DeadlineEqualityPoisonsBeforeAnyTransportSideEffect) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-deadline-test";
  std::filesystem::remove_all(root);
  FakeTransferOperations operations;
  operations.now_ns = 900;
  {
    auto transaction = make_transaction(root, operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    const auto status = transaction->advance();
    EXPECT_EQ(status.code(), StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(transaction->poisoned());
    EXPECT_EQ(operations.manifest_attempts, 0U);
    EXPECT_EQ(operations.batch_attempts, 0U);
    EXPECT_EQ(operations.abort_count, 1U);
    EXPECT_EQ(operations.abort_cause, StatusCode::kDeadlineExceeded);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferTransactionTest,
     FatalBatchSendPoisonsWholeGeneration) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-fatal-send-test";
  std::filesystem::remove_all(root);
  FakeTransferOperations operations;
  operations.fatal_batch_send = true;
  {
    auto transaction = make_transaction(root, operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_FALSE(transaction->advance().ok());
    EXPECT_TRUE(transaction->poisoned());
    EXPECT_EQ(operations.manifest_accepts, 1U);
    EXPECT_EQ(operations.batch_accepts, 0U);
    EXPECT_EQ(operations.abort_count, 1U);
  }
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace pih
