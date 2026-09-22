#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"

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

DeepSeekRuntimeArtifactEvidenceManifest metadata_transfer_evidence(
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

RuntimeEvidenceProjection metadata_transfer_projection(
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

class CompletingDescriptorOperations final
    : public DeepSeekRankArtifactTransferOperations {
 public:
  Status send_manifest(
      std::uint32_t,
      std::span<const std::byte> frame) override {
    return frame.size() == kDeepSeekRankArtifactTransferManifestBytes
               ? Status::Ok()
               : Status::InvalidArgument("descriptor manifest size");
  }

  Status send_descriptor_batch(
      const DeepSeekRankArtifactDescriptorBatchView& batch) override {
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
        batch.descriptor_batch_root,
        batch.adopted_descriptor_set_root};
    auto ack = DeepSeekRankArtifactTransferAck::Create(fields);
    if (!ack.ok()) return ack.status();
    pending_ack = encode_deepseek_rank_artifact_transfer_ack(*ack);
    return Status::Ok();
  }

  Result<std::uint64_t> monotonic_now_ns() override { return 100; }

  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>>>
  poll_ack(std::uint32_t) override {
    auto result = pending_ack;
    pending_ack.reset();
    return result;
  }

  Status abort_generation(
      std::uint64_t, std::uint64_t, const Status&) override {
    ++abort_count;
    return Status::Ok();
  }

  std::uint32_t abort_count = 0;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>> pending_ack;
};

class FakeMetadataOperations final
    : public DeepSeekRankArtifactMetadataTransferOperations {
 public:
  Status send_chunk(
      std::uint32_t rank,
      std::span<const std::byte> frame) override {
    ++send_attempts;
    if (unavailable_once && send_attempts == 1) {
      first_attempt.assign(frame.begin(), frame.end());
      return Status::Unavailable("metadata backpressure");
    }
    if (!first_attempt.empty() &&
        !std::equal(first_attempt.begin(), first_attempt.end(),
                    frame.begin(), frame.end())) {
      return Status::FailedPrecondition("metadata retry bytes changed");
    }
    auto chunk = decode_deepseek_rank_artifact_metadata_chunk(frame);
    if (!chunk.ok()) return chunk.status();
    if (chunk->fields().rank != rank ||
        chunk->fields().chunk_index != accepted_chunks) {
      return Status::FailedPrecondition("metadata chunk order changed");
    }
    assembled.insert(assembled.end(), chunk->payload().begin(),
                     chunk->payload().end());
    ++accepted_chunks;
    DeepSeekRankArtifactMetadataChunkAckFields fields{
        chunk->fields().engine_epoch,
        chunk->fields().worker_generation,
        chunk->fields().world_size,
        chunk->fields().rank,
        chunk->fields().chunk_index,
        chunk->fields().chunk_count,
        chunk->fields().payload_offset + chunk->fields().payload_bytes,
        chunk->fields().total_blob_bytes,
        chunk->fields().process_manifest_identity,
        chunk->fields().process_identity,
        chunk->fields().pidfd_identity,
        chunk->fields().control_identity,
        chunk->fields().challenge_identity,
        chunk->fields().transfer_manifest_root,
        chunk->fields().descriptor_transfer_transaction_root,
        chunk->fields().metadata_root,
        chunk->fields().metadata_transaction_root,
        chunk->fields().blob_sha256,
        chunk->fields().chunk_sha256};
    if (foreign_ack) ++fields.challenge_identity;
    auto ack = DeepSeekRankArtifactMetadataChunkAck::Create(fields);
    if (!ack.ok()) return ack.status();
    pending_ack = encode_deepseek_rank_artifact_metadata_chunk_ack(*ack);
    return Status::Ok();
  }

  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>>
  poll_ack(std::uint32_t) override {
    ++ack_polls;
    auto result = pending_ack;
    pending_ack.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override { return now_ns; }

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
  bool unavailable_once = false;
  bool foreign_ack = false;
  std::uint32_t send_attempts = 0;
  std::uint32_t accepted_chunks = 0;
  std::uint32_t ack_polls = 0;
  std::uint32_t abort_count = 0;
  std::uint64_t aborted_epoch = 0;
  std::uint64_t aborted_generation = 0;
  StatusCode abort_cause = StatusCode::kOk;
  std::vector<std::byte> first_attempt;
  std::vector<std::byte> assembled;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>
      pending_ack;
};

Result<DeepSeekRankArtifactTransferTransaction> make_descriptor_transaction(
    const std::filesystem::path& root,
    CompletingDescriptorOperations& operations) {
  const auto artifact_root = test_fixture::reduced_target_artifact_root();
  auto evidence = metadata_transfer_evidence(artifact_root);
  auto projection = metadata_transfer_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  if (!startup) return Status::Internal("metadata startup fixture failed");
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

Result<DeepSeekRankArtifactMetadataTransferTransaction>
make_metadata_transaction(
    const std::filesystem::path& root,
    CompletingDescriptorOperations& descriptor_operations,
    FakeMetadataOperations& metadata_operations) {
  auto descriptor = make_descriptor_transaction(root, descriptor_operations);
  if (!descriptor.ok()) return descriptor.status();
  auto status = descriptor->advance();
  if (status.code() != StatusCode::kUnavailable) return status;
  status = descriptor->advance();
  if (!status.ok()) return status;
  return DeepSeekRankArtifactMetadataTransferTransaction::Create(
      std::move(*descriptor), metadata_operations);
}

TEST(DeepSeekRankArtifactMetadataTransferTransactionTest,
     CompletesExactAckAndRetainsDescriptorTransaction) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-transfer-test";
  std::filesystem::remove_all(root);
  CompletingDescriptorOperations descriptor_operations;
  FakeMetadataOperations metadata_operations;
  {
    auto transaction = make_metadata_transaction(
        root, descriptor_operations, metadata_operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_EQ(kDeepSeekRankArtifactMetadataTransferTransactionAbi,
              "pih_deepseek_rank_artifact_metadata_transfer_transaction_v1");
    EXPECT_TRUE(transaction->retains_descriptor_transaction());
    EXPECT_NE(transaction->transaction_root(), Sha256Digest{});
    const auto& rank_prefault = transaction->expected_prefault_layout(0);
    const auto& node_prefault =
        transaction->expected_node_prefault_layout();
    EXPECT_EQ(rank_prefault.rank, 0U);
    EXPECT_EQ(rank_prefault.page_bytes, 4096U);
    EXPECT_EQ(rank_prefault.mapped_interval_bytes,
              transaction->expected_mapped_interval_bytes(0));
    EXPECT_NE(rank_prefault.layout_root, Sha256Digest{});
    EXPECT_EQ(node_prefault.world_size, 1U);
    EXPECT_EQ(node_prefault.summed_rank_selected_page_bytes,
              rank_prefault.selected_page_union_bytes);
    EXPECT_EQ(node_prefault.node_selected_page_union_bytes,
              rank_prefault.selected_page_union_bytes);
    EXPECT_EQ(node_prefault.node_duplicate_selected_page_bytes, 0U);
    EXPECT_NE(node_prefault.layout_root, Sha256Digest{});
    EXPECT_EQ(transaction->chunk_count(0), 1U);
    EXPECT_EQ(transaction->acknowledged_bytes(0), 0U);
    auto status = transaction->advance();
    EXPECT_TRUE(status.ok()) << status.message();
    EXPECT_TRUE(transaction->complete());
    EXPECT_FALSE(transaction->poisoned());
    EXPECT_EQ(transaction->acknowledged_bytes(0),
              transaction->blob_bytes(0));
    EXPECT_TRUE(transaction->retains_descriptor_transaction());
    EXPECT_EQ(metadata_operations.accepted_chunks, 1U);
    auto decoded = decode_deepseek_rank_artifact_metadata_blob(
        metadata_operations.assembled);
    ASSERT_TRUE(decoded.ok()) << decoded.status().message();
    EXPECT_EQ(decoded->metadata_root(), transaction->metadata_root(0));
    EXPECT_EQ(sha256(metadata_operations.assembled).value(),
              transaction->blob_sha256(0));
    EXPECT_TRUE(transaction->advance().ok());
    EXPECT_EQ(metadata_operations.send_attempts, 1U);
    EXPECT_THROW((void)transaction->blob_bytes(1), std::out_of_range);
    EXPECT_THROW((void)transaction->expected_prefault_layout(1),
                 std::out_of_range);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataTransferTransactionTest,
     RetriesOnlyIdenticalUnacceptedChunk) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-backpressure-test";
  std::filesystem::remove_all(root);
  CompletingDescriptorOperations descriptor_operations;
  FakeMetadataOperations metadata_operations;
  metadata_operations.unavailable_once = true;
  {
    auto transaction = make_metadata_transaction(
        root, descriptor_operations, metadata_operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_EQ(transaction->advance().code(), StatusCode::kUnavailable);
    EXPECT_EQ(transaction->acknowledged_bytes(0), 0U);
    EXPECT_TRUE(transaction->advance().ok());
    EXPECT_TRUE(transaction->complete());
    EXPECT_EQ(metadata_operations.send_attempts, 2U);
    EXPECT_EQ(metadata_operations.accepted_chunks, 1U);
    EXPECT_EQ(metadata_operations.abort_count, 0U);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataTransferTransactionTest,
     ForeignAckPoisonsAndAbortsGenerationExactlyOnce) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-foreign-ack-test";
  std::filesystem::remove_all(root);
  CompletingDescriptorOperations descriptor_operations;
  FakeMetadataOperations metadata_operations;
  metadata_operations.foreign_ack = true;
  {
    auto transaction = make_metadata_transaction(
        root, descriptor_operations, metadata_operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    EXPECT_FALSE(transaction->advance().ok());
    EXPECT_TRUE(transaction->poisoned());
    EXPECT_FALSE(transaction->complete());
    EXPECT_TRUE(transaction->retains_descriptor_transaction());
    EXPECT_EQ(metadata_operations.abort_count, 1U);
    EXPECT_EQ(metadata_operations.aborted_epoch, 7U);
    EXPECT_EQ(metadata_operations.aborted_generation, 8U);
    EXPECT_FALSE(transaction->advance().ok());
    EXPECT_EQ(metadata_operations.abort_count, 1U);
  }
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataTransferTransactionTest,
     RejectsIncompleteDescriptorTransactionAndDeadlineEquality) {
  const auto incomplete_root = std::filesystem::temp_directory_path() /
                               "pih-artifact-metadata-incomplete-test";
  std::filesystem::remove_all(incomplete_root);
  CompletingDescriptorOperations incomplete_descriptor_operations;
  FakeMetadataOperations incomplete_metadata_operations;
  {
    auto descriptor = make_descriptor_transaction(
        incomplete_root, incomplete_descriptor_operations);
    ASSERT_TRUE(descriptor.ok()) << descriptor.status().message();
    EXPECT_FALSE(DeepSeekRankArtifactMetadataTransferTransaction::Create(
                     std::move(*descriptor), incomplete_metadata_operations)
                     .ok());
  }
  std::filesystem::remove_all(incomplete_root);

  const auto deadline_root = std::filesystem::temp_directory_path() /
                             "pih-artifact-metadata-deadline-test";
  std::filesystem::remove_all(deadline_root);
  CompletingDescriptorOperations descriptor_operations;
  FakeMetadataOperations metadata_operations;
  metadata_operations.now_ns = 900;
  {
    auto transaction = make_metadata_transaction(
        deadline_root, descriptor_operations, metadata_operations);
    ASSERT_TRUE(transaction.ok()) << transaction.status().message();
    const auto status = transaction->advance();
    EXPECT_EQ(status.code(), StatusCode::kDeadlineExceeded);
    EXPECT_TRUE(transaction->poisoned());
    EXPECT_EQ(metadata_operations.send_attempts, 0U);
    EXPECT_EQ(metadata_operations.ack_polls, 0U);
    EXPECT_EQ(metadata_operations.abort_count, 1U);
    EXPECT_EQ(metadata_operations.abort_cause,
              StatusCode::kDeadlineExceeded);
  }
  std::filesystem::remove_all(deadline_root);
}

}  // namespace
}  // namespace pih
