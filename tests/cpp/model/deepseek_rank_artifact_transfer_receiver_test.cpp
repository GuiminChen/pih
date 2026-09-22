#include "pih/model/deepseek_rank_artifact_transfer_receiver.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_rank_startup_test_fixture.h"
#include "deepseek_target_artifact_test_fixture.h"
#include "pih/model/deepseek_controller_artifact_catalog.h"
#include "pih/model/deepseek_rank_artifact_adoption_receipt.h"
#include "pih/model/deepseek_rank_artifact_metadata_receipt.h"
#include "pih/model/deepseek_rank_artifact_metadata_transfer_transaction.h"
#include "pih/model/deepseek_rank_artifact_mapping_owner.h"
#include "pih/model/deepseek_rank_artifact_prefault.h"
#include "pih/model/deepseek_rank_engine_resources.h"
#include "pih/model/deepseek_rank_materialization_exchange.h"
#include "pih/model/deepseek_rank_materialization_grant.h"
#include "pih/model/deepseek_rank_materialization_warmup.h"
#include "pih/model/deepseek_rank_post_mapping_resource_exchange.h"
#include "pih/model/runtime_evidence_projection.h"

namespace pih {
namespace {

DeepSeekRuntimeArtifactEvidenceManifest receiver_evidence(
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

RuntimeEvidenceProjection receiver_projection(
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

DeepSeekRankMaterializationAllocationAuthority
materialization_allocation_authority(std::int32_t device_ordinal,
                                     RuntimeProfileResidency residency) {
  DeepSeekRankMaterializationAllocationAuthority authority;
  authority.device_ordinal = device_ordinal;
  authority.residency = residency;
  authority.physical_gpu_identity = test_fixture::rank_capacity_digest(201);
  authority.cuda_allocation_identity = 202;
  authority.cuda_owner_identity = test_fixture::rank_capacity_digest(203);
  authority.cuda_resource_identity = test_fixture::rank_capacity_digest(204);
  if (residency == RuntimeProfileResidency::kHostSpill) {
    authority.pinned_registration_identity = 205;
    authority.pinned_owner_identity = test_fixture::rank_capacity_digest(206);
    authority.pinned_resource_identity =
        test_fixture::rank_capacity_digest(207);
    authority.pinned_numa_node = 0;
  }
  return authority;
}

class FakeDuplexArtifactChannel final
    : public DeepSeekRankArtifactTransferOperations,
      public DeepSeekRankArtifactTransferReceiverOperations {
 public:
  explicit FakeDuplexArtifactChannel(std::filesystem::path artifact_root)
      : artifact_root_(std::move(artifact_root)) {}

  Status send_manifest(
      std::uint32_t,
      std::span<const std::byte> manifest_frame) override {
    if (queued_manifest) {
      return Status::FailedPrecondition("manifest duplicated");
    }
    queued_manifest =
        std::vector<std::byte>(manifest_frame.begin(), manifest_frame.end());
    return Status::Ok();
  }

  Status send_descriptor_batch(
      const DeepSeekRankArtifactDescriptorBatchView& view) override {
    if (queued_batch) {
      return Status::FailedPrecondition("descriptor batch duplicated");
    }
    auto batch = DeepSeekRankArtifactDescriptorBatch::FromView(view);
    if (!batch.ok()) return batch.status();
    queued_batch = std::move(*batch);
    return Status::Ok();
  }

  Result<std::uint64_t> monotonic_now_ns() override { return now_ns; }

  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>>>
  poll_ack(std::uint32_t) override {
    auto result = queued_ack;
    queued_ack.reset();
    return result;
  }

  Status abort_generation(
      std::uint64_t, std::uint64_t, const Status&) override {
    ++abort_count;
    return Status::Ok();
  }

  Result<std::optional<std::vector<std::byte>>> receive_manifest(
      std::int32_t) override {
    auto result = std::move(queued_manifest);
    queued_manifest.reset();
    return result;
  }

  Result<std::optional<DeepSeekRankArtifactReceivedBatch>>
  receive_descriptor_batch(std::int32_t) override {
    if (!queued_batch) {
      return std::optional<DeepSeekRankArtifactReceivedBatch>{};
    }
    std::vector<DeepSeekWorkerShardDescriptor> descriptors;
    for (const auto& expectation : queued_batch->expectations()) {
      const auto member = substitute_descriptor
                              ? std::string_view{"foreign.safetensors"}
                              : std::string_view{expectation.shard_name};
      auto lease = ControllerFileLease::OpenBeneath(
          artifact_root_, member, 4096,
          ArtifactImmutabilityMode::kUncalibrated);
      if (!lease.ok()) return lease.status();
      auto descriptor = lease->duplicate_for_worker();
      if (!descriptor.ok()) return descriptor.status();
      descriptors.emplace_back(expectation.shard_name,
                               std::move(*descriptor));
    }
    auto metadata = std::move(*queued_batch);
    queued_batch.reset();
    return std::optional<DeepSeekRankArtifactReceivedBatch>{
        std::in_place,
        DeepSeekRankArtifactReceivedBatch{
            std::move(metadata), std::move(descriptors)}};
  }

  Status send_ack(
      std::int32_t, std::span<const std::byte> frame) override {
    ++ack_send_attempts;
    if (ack_unavailable_once && ack_send_attempts == 1) {
      return Status::Unavailable("ack backpressure");
    }
    if (frame.size() != kDeepSeekRankArtifactTransferAckBytes || queued_ack) {
      return Status::FailedPrecondition("ack frame state invalid");
    }
    std::array<std::byte, kDeepSeekRankArtifactTransferAckBytes> copy{};
    std::copy(frame.begin(), frame.end(), copy.begin());
    queued_ack = copy;
    return Status::Ok();
  }

  std::uint64_t now_ns = 100;
  bool ack_unavailable_once = false;
  bool substitute_descriptor = false;
  std::uint32_t ack_send_attempts = 0;
  std::uint32_t abort_count = 0;
  std::optional<std::vector<std::byte>> queued_manifest;
  std::optional<DeepSeekRankArtifactDescriptorBatch> queued_batch;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactTransferAckBytes>> queued_ack;

 private:
  std::filesystem::path artifact_root_;
};

Result<DeepSeekRankArtifactTransferTransaction> make_controller_transaction(
    const std::filesystem::path& root,
    FakeDuplexArtifactChannel& channel) {
  const auto artifact_root = test_fixture::reduced_target_artifact_root();
  auto evidence = receiver_evidence(artifact_root);
  auto projection = receiver_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  if (!startup) return Status::Internal("startup fixture failed");
  auto model_startup = startup->compile_model_startup();
  if (!model_startup.ok()) return model_startup.status();
  auto binding = DeepSeekRuntimeArtifactAdmissionBinding::Issue(
      startup->admission(), evidence);
  if (!binding.ok()) return binding.status();
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  auto catalog = DeepSeekControllerArtifactCatalog::OpenTargetGeneration(
      root, artifact_root, *pipeline, 4096,
      ArtifactImmutabilityMode::kUncalibrated);
  if (!catalog.ok()) return catalog.status();
  auto handoff = DeepSeekRankArtifactHandoffPlan::Compile(*catalog);
  if (!handoff.ok()) return handoff.status();
  auto plan = DeepSeekRankArtifactTransferPlan::Compile(
      std::move(*model_startup), std::move(*binding), std::move(*handoff));
  if (!plan.ok()) return plan.status();
  return DeepSeekRankArtifactTransferTransaction::Create(
      std::move(*plan), channel);
}

DeepSeekRankExecReady local_exec_ready(
    const DeepSeekRankProcessManifest& manifest) {
  return {{manifest.engine_epoch,
           manifest.worker_generation,
           manifest.rank,
           manifest.physical_device_identity,
           manifest.process_manifest_identity,
           100,
           200,
           300,
           manifest.physical_device_uuid_commitment,
           manifest.startup_device_ordinal,
           manifest.startup_deadline_ns},
          400};
}

Result<std::pair<DeepSeekRankArtifactTransferTransaction,
                 DeepSeekRankArtifactAdoptionReceipt>>
complete_descriptor_handoff(
    const std::filesystem::path& root,
    FakeDuplexArtifactChannel& channel,
    DeepSeekRankScmRightsInFlightLedger& ledger) {
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto receiver = DeepSeekRankArtifactTransferReceiver::Create(
      manifest, local_exec_ready(manifest), 9, 1000, ledger, channel);
  if (!receiver.ok()) return receiver.status();
  auto controller = make_controller_transaction(root, channel);
  if (!controller.ok()) return controller.status();
  auto status = controller->advance();
  if (status.code() != StatusCode::kUnavailable) return status;
  status = receiver->advance();
  if (!status.ok()) return status;
  status = controller->advance();
  if (!status.ok()) return status;
  auto adoption = DeepSeekRankArtifactAdoptionReceipt::Create(
      std::move(*receiver));
  if (!adoption.ok()) return adoption.status();
  return std::pair<DeepSeekRankArtifactTransferTransaction,
                   DeepSeekRankArtifactAdoptionReceipt>{
      std::move(*controller), std::move(*adoption)};
}

struct FakeMetadataChannelState final {
  std::uint64_t now_ns = 100;
  bool chunk_unavailable_once = false;
  bool ack_unavailable_once = false;
  std::uint32_t chunk_send_attempts = 0;
  std::uint32_t chunk_accepts = 0;
  std::uint32_t ack_send_attempts = 0;
  std::uint32_t abort_count = 0;
  std::vector<std::byte> first_chunk_attempt;
  std::optional<std::vector<std::byte>> queued_chunk;
  std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>> queued_ack;
};

class FakeMetadataControllerOperations final
    : public DeepSeekRankArtifactMetadataTransferOperations {
 public:
  explicit FakeMetadataControllerOperations(FakeMetadataChannelState& state)
      : state_(&state) {}

  Status send_chunk(
      std::uint32_t,
      std::span<const std::byte> frame) override {
    ++state_->chunk_send_attempts;
    if (state_->chunk_unavailable_once &&
        state_->chunk_send_attempts == 1) {
      state_->first_chunk_attempt.assign(frame.begin(), frame.end());
      return Status::Unavailable("metadata chunk backpressure");
    }
    if (!state_->first_chunk_attempt.empty() &&
        !std::equal(state_->first_chunk_attempt.begin(),
                    state_->first_chunk_attempt.end(), frame.begin(),
                    frame.end())) {
      return Status::FailedPrecondition("metadata retry changed");
    }
    if (state_->queued_chunk) {
      return Status::FailedPrecondition("metadata chunk duplicated");
    }
    state_->queued_chunk =
        std::vector<std::byte>(frame.begin(), frame.end());
    ++state_->chunk_accepts;
    return Status::Ok();
  }

  Result<std::optional<std::array<
      std::byte, kDeepSeekRankArtifactMetadataChunkAckFrameBytes>>>
  poll_ack(std::uint32_t) override {
    auto result = state_->queued_ack;
    state_->queued_ack.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    return state_->now_ns;
  }

  Status abort_generation(
      std::uint64_t, std::uint64_t, const Status&) override {
    ++state_->abort_count;
    return Status::Ok();
  }

 private:
  FakeMetadataChannelState* state_ = nullptr;
};

class FakeMetadataReceiverOperations final
    : public DeepSeekRankArtifactMetadataReceiverOperations {
 public:
  explicit FakeMetadataReceiverOperations(FakeMetadataChannelState& state)
      : state_(&state) {}

  Result<std::optional<std::vector<std::byte>>> receive_chunk(
      std::int32_t) override {
    auto result = std::move(state_->queued_chunk);
    state_->queued_chunk.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    return state_->now_ns;
  }

  Status send_ack(
      std::int32_t, std::span<const std::byte> frame) override {
    ++state_->ack_send_attempts;
    if (state_->ack_unavailable_once &&
        state_->ack_send_attempts == 1) {
      return Status::Unavailable("metadata ACK backpressure");
    }
    if (frame.size() !=
            kDeepSeekRankArtifactMetadataChunkAckFrameBytes ||
        state_->queued_ack) {
      return Status::FailedPrecondition("metadata ACK state invalid");
    }
    std::array<std::byte,
               kDeepSeekRankArtifactMetadataChunkAckFrameBytes> copy{};
    std::copy(frame.begin(), frame.end(), copy.begin());
    state_->queued_ack = copy;
    return Status::Ok();
  }

 private:
  FakeMetadataChannelState* state_ = nullptr;
};

struct FakePostMappingExchangeState final {
  std::uint64_t now_ns = 901;
  bool authority_unavailable_once = true;
  bool observation_unavailable_once = true;
  std::uint32_t authority_attempts = 0;
  std::uint32_t observation_attempts = 0;
  std::vector<std::byte> first_authority;
  std::vector<std::byte> first_observation;
  std::optional<std::vector<std::byte>> queued_authority;
  std::optional<std::vector<std::byte>> queued_observation;
};

class FakePostMappingCoordinatorChannel final
    : public DeepSeekRankPostMappingResourceChannel {
 public:
  explicit FakePostMappingCoordinatorChannel(
      FakePostMappingExchangeState& state)
      : state_(&state) {}

  Status send_authority(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override {
    if (handle.process_identity != 100 || state_->queued_authority) {
      return Status::FailedPrecondition(
          "post-mapping authority handle or queue is invalid");
    }
    ++state_->authority_attempts;
    if (state_->authority_unavailable_once &&
        state_->authority_attempts == 1) {
      state_->first_authority.assign(frame.begin(), frame.end());
      return Status::Unavailable("post-mapping authority backpressure");
    }
    if (!state_->first_authority.empty() &&
        !std::equal(state_->first_authority.begin(),
                    state_->first_authority.end(), frame.begin(),
                    frame.end())) {
      return Status::FailedPrecondition(
          "post-mapping authority retry changed");
    }
    state_->queued_authority =
        std::vector<std::byte>(frame.begin(), frame.end());
    return Status::Ok();
  }

  Result<std::optional<std::vector<std::byte>>> poll_observation(
      const DeepSeekRankProcessHandle& handle) override {
    if (handle.process_identity != 100) {
      return Status::FailedPrecondition(
          "post-mapping observation handle is invalid");
    }
    auto result = std::move(state_->queued_observation);
    state_->queued_observation.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    return state_->now_ns;
  }

 private:
  FakePostMappingExchangeState* state_ = nullptr;
};

class FakePostMappingReporterOperations final
    : public DeepSeekRankPostMappingResourceReporterOperations {
 public:
  explicit FakePostMappingReporterOperations(
      FakePostMappingExchangeState& state)
      : state_(&state) {}

  Result<std::optional<std::vector<std::byte>>> receive_authority(
      std::int32_t control_fd) override {
    if (control_fd != 9) {
      return Status::FailedPrecondition(
          "post-mapping reporter control descriptor differs");
    }
    auto result = std::move(state_->queued_authority);
    state_->queued_authority.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    return state_->now_ns;
  }

  Status send_observation(
      std::int32_t control_fd,
      std::span<const std::byte> frame) override {
    if (control_fd != 9 || state_->queued_observation) {
      return Status::FailedPrecondition(
          "post-mapping observation queue is invalid");
    }
    ++state_->observation_attempts;
    if (state_->observation_unavailable_once &&
        state_->observation_attempts == 1) {
      state_->first_observation.assign(frame.begin(), frame.end());
      return Status::Unavailable("post-mapping observation backpressure");
    }
    if (!state_->first_observation.empty() &&
        !std::equal(state_->first_observation.begin(),
                    state_->first_observation.end(), frame.begin(),
                    frame.end())) {
      return Status::FailedPrecondition(
          "post-mapping observation retry changed");
    }
    state_->queued_observation =
        std::vector<std::byte>(frame.begin(), frame.end());
    return Status::Ok();
  }

 private:
  FakePostMappingExchangeState* state_ = nullptr;
};

class FakePostMappingResourceProbe final
    : public DeepSeekRankPostExecResourceProbe {
 public:
  Result<DeepSeekRankPostExecResourceSnapshot> sample() override {
    ++samples;
    return DeepSeekRankPostExecResourceSnapshot{
        100, 8, 20, 0, 200, 24, 100, 1000, 210, true};
  }

  std::uint32_t samples = 0;
};

struct FakeMaterializationExchangeState final {
  std::uint64_t now_ns = 1001;
  bool grant_unavailable_once = true;
  bool ack_unavailable_once = true;
  std::uint32_t grant_attempts = 0;
  std::uint32_t ack_attempts = 0;
  std::vector<std::byte> first_grant;
  std::vector<std::byte> first_ack;
  std::optional<std::vector<std::byte>> queued_grant;
  std::optional<std::vector<std::byte>> queued_ack;
};

class FakeMaterializationExchange final
    : public DeepSeekRankMaterializationGrantChannel,
      public DeepSeekRankMaterializationGrantReceiverOperations {
 public:
  explicit FakeMaterializationExchange(
      FakeMaterializationExchangeState& state)
      : state_(&state) {}

  Status send_grant(
      const DeepSeekRankProcessHandle& handle,
      std::span<const std::byte> frame) override {
    if (handle.process_identity != 100 || state_->queued_grant) {
      return Status::FailedPrecondition(
          "materialization grant queue is invalid");
    }
    ++state_->grant_attempts;
    if (state_->grant_unavailable_once &&
        state_->grant_attempts == 1) {
      state_->first_grant.assign(frame.begin(), frame.end());
      return Status::Unavailable("materialization grant backpressure");
    }
    if (!state_->first_grant.empty() &&
        !std::equal(state_->first_grant.begin(),
                    state_->first_grant.end(), frame.begin(),
                    frame.end())) {
      return Status::FailedPrecondition(
          "materialization grant retry changed");
    }
    state_->queued_grant =
        std::vector<std::byte>(frame.begin(), frame.end());
    return Status::Ok();
  }

  Result<std::optional<std::vector<std::byte>>> poll_ack(
      const DeepSeekRankProcessHandle& handle) override {
    if (handle.process_identity != 100) {
      return Status::FailedPrecondition(
          "materialization ACK handle differs");
    }
    auto result = std::move(state_->queued_ack);
    state_->queued_ack.reset();
    return result;
  }

  Result<std::optional<std::vector<std::byte>>> receive_grant(
      std::int32_t control_fd) override {
    if (control_fd != 9) {
      return Status::FailedPrecondition(
          "materialization receiver control descriptor differs");
    }
    auto result = std::move(state_->queued_grant);
    state_->queued_grant.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override {
    return state_->now_ns;
  }

  Status send_ack(
      std::int32_t control_fd,
      std::span<const std::byte> frame) override {
    if (control_fd != 9 || state_->queued_ack) {
      return Status::FailedPrecondition(
          "materialization ACK queue is invalid");
    }
    ++state_->ack_attempts;
    if (state_->ack_unavailable_once && state_->ack_attempts == 1) {
      state_->first_ack.assign(frame.begin(), frame.end());
      return Status::Unavailable("materialization ACK backpressure");
    }
    if (!state_->first_ack.empty() &&
        !std::equal(state_->first_ack.begin(), state_->first_ack.end(),
                    frame.begin(), frame.end())) {
      return Status::FailedPrecondition(
          "materialization ACK retry changed");
    }
    state_->queued_ack =
        std::vector<std::byte>(frame.begin(), frame.end());
    return Status::Ok();
  }

 private:
  FakeMaterializationExchangeState* state_ = nullptr;
};

class FakeArtifactPrefaultOperations final
    : public DeepSeekRankArtifactPrefaultOperations {
 public:
  Result<DeepSeekRankArtifactPrefaultResourceSnapshot>
  sample_resources() override {
    ++snapshot_count;
    return snapshot_count == 1
               ? DeepSeekRankArtifactPrefaultResourceSnapshot{
                     1050, 7, 4096, 100000, 80000, 10000, 10000}
               : DeepSeekRankArtifactPrefaultResourceSnapshot{
                     1060, 9, 8192, 104096, 84096, 10000, 10000};
  }

  Result<DeepSeekRankArtifactPrefaultRangeObservation>
  prefault_range(std::span<const std::byte> mapping,
                 std::uint64_t page_bytes) override {
    if (mapping.empty() || page_bytes != 4096) {
      return Status::InvalidArgument("fake prefault range is invalid");
    }
    ++range_count;
    const auto pages =
        (static_cast<std::uint64_t>(mapping.size()) + page_bytes - 1U) /
        page_bytes;
    return DeepSeekRankArtifactPrefaultRangeObservation{
        0, pages * page_bytes, pages};
  }

  std::uint32_t snapshot_count = 0;
  std::uint32_t range_count = 0;
};

class FakeMaterializationWarmupOperations final
    : public DeepSeekRankMaterializationWarmupOperations {
 public:
  Result<std::optional<std::vector<std::byte>>> poll_completion(
      const DeepSeekRankProcessHandle&) override {
    ++poll_count;
    auto result = queued_completion;
    queued_completion.reset();
    return result;
  }

  Result<std::uint64_t> monotonic_now_ns() override { return now_ns; }

  Status abort_generation(
      std::uint64_t, std::uint64_t, const Status&) override {
    ++abort_count;
    return Status::Ok();
  }

  std::uint64_t now_ns = 1070;
  std::uint32_t poll_count = 0;
  std::uint32_t abort_count = 0;
  std::optional<std::vector<std::byte>> queued_completion;
};

TEST(DeepSeekRankArtifactTransferReceiverTest,
     HoldsDescriptorsAndLedgerUntilAckIsAccepted) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-receiver-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  FakeDuplexArtifactChannel channel(root);
  channel.ack_unavailable_once = true;
  DeepSeekRankScmRightsInFlightLedger ledger;
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto receiver = DeepSeekRankArtifactTransferReceiver::Create(
      manifest, local_exec_ready(manifest), 9, 1000, ledger, channel);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  ASSERT_TRUE(receiver->wait_event().has_value());
  EXPECT_EQ(*receiver->wait_event(),
            DeepSeekRankArtifactTransferReceiverWaitEvent::kPacketReadable);
  EXPECT_EQ(receiver->wait_deadline_ns(), 1000U);
  auto controller = make_controller_transaction(root, channel);
  ASSERT_TRUE(controller.ok()) << controller.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactTransferReceiverAbi,
            "pih_deepseek_rank_artifact_transfer_receiver_v1");

  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);
  auto status = receiver->advance();
  EXPECT_EQ(status.code(), StatusCode::kUnavailable);
  EXPECT_TRUE(receiver->manifest_received());
  EXPECT_FALSE(receiver->complete());
  EXPECT_FALSE(receiver->poisoned());
  EXPECT_EQ(receiver->adopted_descriptor_count(), 1U);
  EXPECT_NE(receiver->transaction_root(), nullptr);
  ASSERT_TRUE(receiver->wait_event().has_value());
  EXPECT_EQ(*receiver->wait_event(),
            DeepSeekRankArtifactTransferReceiverWaitEvent::kAckWritable);
  EXPECT_LT(receiver->wait_deadline_ns(), 1000U);
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 1U);
  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);

  status = receiver->advance();
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(receiver->complete());
  EXPECT_FALSE(receiver->wait_event().has_value());
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 0U);
  EXPECT_TRUE(controller->advance().ok());
  EXPECT_TRUE(controller->complete());
  EXPECT_EQ(channel.ack_send_attempts, 2U);
  EXPECT_EQ(channel.abort_count, 0U);
  auto adoption = DeepSeekRankArtifactAdoptionReceipt::Create(
      std::move(*receiver));
  ASSERT_TRUE(adoption.ok()) << adoption.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactAdoptionReceiptAbi,
            "pih_deepseek_rank_artifact_adoption_receipt_v1");
  EXPECT_EQ(adoption->descriptor_count(), 1U);
  EXPECT_TRUE(adoption->retains_descriptor_owners());
  EXPECT_EQ(adoption->transaction_root(), controller->transaction_root());
  EXPECT_NE(adoption->adopted_descriptor_set_root(), Sha256Digest{});
  EXPECT_NE(adoption->receipt_root(), Sha256Digest{});
  EXPECT_EQ(adoption->transfer_manifest().fields().rank, 0U);
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferReceiverTest,
     AdoptionReceiptRejectsIncompleteReceiver) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-adoption-incomplete-test";
  FakeDuplexArtifactChannel channel(root);
  DeepSeekRankScmRightsInFlightLedger ledger;
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto receiver = DeepSeekRankArtifactTransferReceiver::Create(
      manifest, local_exec_ready(manifest), 9, 1000, ledger, channel);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  auto adoption = DeepSeekRankArtifactAdoptionReceipt::Create(
      std::move(*receiver));
  EXPECT_FALSE(adoption.ok());
  EXPECT_EQ(adoption.status().code(), StatusCode::kFailedPrecondition);
}

TEST(DeepSeekRankArtifactTransferReceiverTest,
     RejectsSubstitutedDescriptorBeforeLedgerOrAck) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-receiver-substitute-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  test_fixture::write_target_safetensors(
      root / "foreign.safetensors",
      R"({"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}})",
      "z");
  FakeDuplexArtifactChannel channel(root);
  channel.substitute_descriptor = true;
  DeepSeekRankScmRightsInFlightLedger ledger;
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto receiver = DeepSeekRankArtifactTransferReceiver::Create(
      manifest, local_exec_ready(manifest), 9, 1000, ledger, channel);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  auto controller = make_controller_transaction(root, channel);
  ASSERT_TRUE(controller.ok()) << controller.status().message();
  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(receiver->advance().ok());
  EXPECT_TRUE(receiver->poisoned());
  EXPECT_EQ(receiver->adopted_descriptor_count(), 0U);
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 0U);
  EXPECT_EQ(channel.ack_send_attempts, 0U);
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferReceiverTest,
     DeadlineEqualityRejectsBeforeManifestReceive) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-receiver-deadline-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  FakeDuplexArtifactChannel channel(root);
  channel.now_ns = 1000;
  DeepSeekRankScmRightsInFlightLedger ledger;
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto receiver = DeepSeekRankArtifactTransferReceiver::Create(
      manifest, local_exec_ready(manifest), 9, 1000, ledger, channel);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  EXPECT_EQ(receiver->advance().code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(receiver->poisoned());
  EXPECT_FALSE(receiver->manifest_received());
  EXPECT_EQ(ledger.sample_inflight_fd_count().value(), 0U);
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactTransferReceiverTest,
     RejectsLocalReadyWithDifferentPhysicalUuidCommitment) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-transfer-receiver-uuid-test";
  FakeDuplexArtifactChannel channel(root);
  DeepSeekRankScmRightsInFlightLedger ledger;
  const auto manifest = test_fixture::startup_manifests(1).front();
  auto ready = local_exec_ready(manifest);
  ready.receipt.physical_device_uuid_commitment =
      test_fixture::rank_capacity_digest(99);
  EXPECT_FALSE(DeepSeekRankArtifactTransferReceiver::Create(
                   manifest, ready, 9, 1000, ledger, channel)
                   .ok());
}

TEST(DeepSeekRankArtifactMetadataReceiverTest,
     WithholdsFinalAckUntilDecodeAndRetainsOwnersAcrossBackpressure) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-receiver-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  FakeDuplexArtifactChannel descriptor_channel(root);
  DeepSeekRankScmRightsInFlightLedger ledger;
  auto handoff = complete_descriptor_handoff(
      root, descriptor_channel, ledger);
  ASSERT_TRUE(handoff.ok()) << handoff.status().message();

  FakeMetadataChannelState state;
  state.ack_unavailable_once = true;
  FakeMetadataControllerOperations controller_operations(state);
  FakeMetadataReceiverOperations receiver_operations(state);
  auto controller =
      DeepSeekRankArtifactMetadataTransferTransaction::Create(
          std::move(handoff->first), controller_operations);
  ASSERT_TRUE(controller.ok()) << controller.status().message();
  const auto descriptor_transaction_root =
      controller->descriptor_transaction_root();
  auto receiver = DeepSeekRankArtifactMetadataReceiver::Create(
      std::move(handoff->second), false,
      kDeepSeekRankArtifactMetadataBlobMaximumBytes,
      receiver_operations);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMetadataReceiverAbi,
            "pih_deepseek_rank_artifact_metadata_receiver_v1");
  EXPECT_TRUE(receiver->retains_descriptor_owners());
  ASSERT_TRUE(receiver->wait_event().has_value());
  EXPECT_EQ(*receiver->wait_event(),
            DeepSeekRankArtifactMetadataReceiverWaitEvent::kPacketReadable);

  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);
  auto status = receiver->advance();
  EXPECT_EQ(status.code(), StatusCode::kUnavailable);
  EXPECT_FALSE(receiver->complete());
  EXPECT_FALSE(receiver->poisoned());
  EXPECT_TRUE(receiver->retains_descriptor_owners());
  EXPECT_EQ(receiver->reassembled_bytes(), controller->blob_bytes(0));
  EXPECT_EQ(receiver->acknowledged_chunk_count(), 0U);
  ASSERT_TRUE(receiver->metadata_transaction_root() != nullptr);
  EXPECT_EQ(*receiver->metadata_transaction_root(),
            controller->transaction_root());
  ASSERT_TRUE(receiver->wait_event().has_value());
  EXPECT_EQ(*receiver->wait_event(),
            DeepSeekRankArtifactMetadataReceiverWaitEvent::kAckWritable);
  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);

  status = receiver->advance();
  EXPECT_TRUE(status.ok()) << status.message();
  EXPECT_TRUE(receiver->complete());
  EXPECT_FALSE(receiver->wait_event().has_value());
  EXPECT_EQ(receiver->acknowledged_chunk_count(), 1U);
  EXPECT_TRUE(controller->advance().ok());
  EXPECT_TRUE(controller->complete());
  EXPECT_EQ(state.chunk_accepts, 1U);
  EXPECT_EQ(state.ack_send_attempts, 2U);
  EXPECT_EQ(state.abort_count, 0U);

  auto receipt = DeepSeekRankArtifactMetadataReceipt::Create(
      std::move(*receiver));
  ASSERT_TRUE(receipt.ok()) << receipt.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMetadataReceiptAbi,
            "pih_deepseek_rank_artifact_metadata_receipt_v1");
  EXPECT_TRUE(receipt->retains_descriptor_and_metadata_owners());
  EXPECT_EQ(receipt->descriptor_transaction_root(),
            descriptor_transaction_root);
  EXPECT_EQ(receipt->metadata_transaction_root(),
            controller->transaction_root());
  EXPECT_EQ(receipt->metadata_root(), controller->metadata_root(0));
  EXPECT_EQ(receipt->blob_sha256(), controller->blob_sha256(0));
  EXPECT_EQ(receipt->blob_bytes(), controller->blob_bytes(0));
  EXPECT_NE(receipt->receipt_root(), Sha256Digest{});
  const auto metadata_receipt_root = receipt->receipt_root();
  auto mapping = DeepSeekRankArtifactMappingOwner::Create(
      std::move(*receipt));
  ASSERT_TRUE(mapping.ok()) << mapping.status().message();
  EXPECT_EQ(kDeepSeekRankArtifactMappingOwnerAbi,
            "pih_deepseek_rank_artifact_mapping_owner_v1");
  EXPECT_EQ((*mapping)->rank(), 0U);
  EXPECT_EQ((*mapping)->tensor_count(), 1U);
  EXPECT_GT((*mapping)->mapped_interval_bytes(), 0U);
  EXPECT_EQ((*mapping)->immutability_mode(),
            ArtifactImmutabilityMode::kUncalibrated);
  EXPECT_FALSE((*mapping)->source_catalog_production_eligible());
  EXPECT_FALSE((*mapping)->dspark_enabled());
  EXPECT_EQ((*mapping)->metadata_receipt_root(), metadata_receipt_root);
  EXPECT_NE((*mapping)->mapping_owner_root(), Sha256Digest{});
  EXPECT_EQ((*mapping)->mapping_owner_root(),
            controller->expected_mapping_owner_root(0));
  auto tensor = (*mapping)->resolve("embed.weight");
  ASSERT_TRUE(tensor.ok()) << tensor.status().message();
  ASSERT_EQ(tensor->bytes.size(), 4U);
  EXPECT_EQ(std::to_integer<char>(tensor->bytes[0]), 'a');
  EXPECT_EQ(std::to_integer<char>(tensor->bytes[3]), 'd');
  EXPECT_FALSE((*mapping)->resolve("foreign.weight").ok());

  auto evidence = receiver_evidence(
      test_fixture::reduced_target_artifact_root());
  auto projection = receiver_projection(evidence);
  auto startup = test_fixture::DeepSeekRankStartupFixture::Create(
      1, &projection, false);
  ASSERT_NE(startup, nullptr);
  DeepSeekRankPostMappingResourceObservation observation{
      {7,
       8,
       0,
       100,
       400,
       startup->supervisor().capacity_plan_instance_root(),
       test_fixture::rank_capacity_digest(30),
       8,
       20,
       0,
       200,
       24,
       100,
       1000,
       210,
       true},
      startup->resource_seal().seal_root(),
      controller->descriptor_transaction_root(),
      controller->transaction_root(),
      controller->metadata_root(0),
      (*mapping)->mapping_owner_root(),
      (*mapping)->mapped_interval_bytes(),
      (*mapping)->immutability_mode(),
      (*mapping)->source_catalog_production_eligible(),
      (*mapping)->dspark_enabled()};
  auto post_mapping_receipt =
      DeepSeekRankPostMappingResourceReceipt::Compile(
          startup->supervisor(), startup->manifests(),
          startup->spawn_plan(), startup->resource_plans(),
          startup->resource_seal(), *controller, observation);
  ASSERT_TRUE(post_mapping_receipt.ok())
      << post_mapping_receipt.status().message();
  EXPECT_EQ(kDeepSeekRankPostMappingResourceReceiptAbi,
            "pih_deepseek_rank_post_mapping_resource_receipt_v1");
  EXPECT_EQ(post_mapping_receipt->mapping_owner_root(),
            (*mapping)->mapping_owner_root());
  EXPECT_EQ(post_mapping_receipt->first_resource_seal_root(),
            startup->resource_seal().seal_root());
  EXPECT_NE(startup->resource_seal().plan_set_root(), Sha256Digest{});
  const std::array<DeepSeekRankPostMappingResourceReceipt, 1> receipts{
      *post_mapping_receipt};
  auto post_mapping_seal = DeepSeekRankPostMappingResourceSeal::Compile(
      startup->supervisor(), startup->manifests(), startup->spawn_plan(),
      startup->resource_plans(), startup->resource_seal(), *controller,
      receipts);
  ASSERT_TRUE(post_mapping_seal.ok())
      << post_mapping_seal.status().message();
  EXPECT_EQ(kDeepSeekRankPostMappingResourceSealAbi,
            "pih_deepseek_rank_post_mapping_resource_seal_v1");
  EXPECT_EQ(post_mapping_seal->world_size(), 1U);
  EXPECT_EQ(post_mapping_seal->metadata_transaction_root(),
            controller->transaction_root());
  EXPECT_FALSE(post_mapping_seal->production_eligible());
  EXPECT_FALSE(post_mapping_seal->dspark_enabled());
  EXPECT_NE(post_mapping_seal->seal_root(), Sha256Digest{});
  auto replay_seal = DeepSeekRankPostMappingResourceSeal::Compile(
      startup->supervisor(), startup->manifests(), startup->spawn_plan(),
      startup->resource_plans(), startup->resource_seal(), *controller,
      receipts);
  ASSERT_TRUE(replay_seal.ok()) << replay_seal.status().message();
  EXPECT_EQ(replay_seal->seal_root(), post_mapping_seal->seal_root());

  FakePostMappingExchangeState exchange_state;
  FakePostMappingCoordinatorChannel coordinator_channel(exchange_state);
  FakePostMappingReporterOperations reporter_operations(exchange_state);
  FakePostMappingResourceProbe post_mapping_probe;
  auto post_mapping_collector =
      StableDeepSeekRankPostExecResourceCollector::Create(
          post_mapping_probe, 2);
  ASSERT_TRUE(post_mapping_collector.ok())
      << post_mapping_collector.status().message();
  auto post_mapping_reporter =
      DeepSeekRankPostMappingResourceReporter::Create(
          startup->manifests()[0],
          *startup->supervisor().exec_ready(0), 9, **mapping,
          *post_mapping_collector, reporter_operations);
  ASSERT_TRUE(post_mapping_reporter.ok())
      << post_mapping_reporter.status().message();
  auto post_mapping_coordinator =
      DeepSeekRankPostMappingResourceCoordinator::Create(
          startup->mutable_supervisor(), startup->manifests(),
          startup->spawn_plan(), startup->resource_plans(),
          startup->resource_seal(), *controller, 1000,
          coordinator_channel);
  ASSERT_TRUE(post_mapping_coordinator.ok())
      << post_mapping_coordinator.status().message();
  EXPECT_EQ(kDeepSeekRankPostMappingResourceExchangeAbi,
            "pih_deepseek_rank_post_mapping_resource_exchange_v1");
  EXPECT_EQ(post_mapping_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(exchange_state.authority_attempts, 1U);
  EXPECT_EQ(post_mapping_reporter->advance().code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(post_mapping_reporter->wait_event().has_value());
  EXPECT_EQ(*post_mapping_reporter->wait_event(),
            DeepSeekRankPostMappingResourceReporterWaitEvent::
                kAuthorityReadable);
  EXPECT_EQ(post_mapping_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(exchange_state.authority_attempts, 2U);
  EXPECT_EQ(post_mapping_reporter->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(post_mapping_probe.samples, 2U);
  ASSERT_TRUE(post_mapping_reporter->wait_event().has_value());
  EXPECT_EQ(*post_mapping_reporter->wait_event(),
            DeepSeekRankPostMappingResourceReporterWaitEvent::
                kObservationWritable);
  EXPECT_EQ(post_mapping_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_TRUE(post_mapping_reporter->advance().ok());
  EXPECT_TRUE(post_mapping_reporter->reported());
  EXPECT_FALSE(post_mapping_reporter->wait_event().has_value());
  ASSERT_NE(post_mapping_reporter->report(), nullptr);
  EXPECT_EQ(exchange_state.observation_attempts, 2U);
  EXPECT_TRUE(post_mapping_coordinator->advance().ok());
  EXPECT_TRUE(post_mapping_coordinator->sealed());
  EXPECT_EQ(post_mapping_coordinator->receipt_count(), 1U);
  ASSERT_NE(post_mapping_coordinator->seal(), nullptr);
  ASSERT_NE(post_mapping_coordinator->report(0), nullptr);
  ASSERT_NE(post_mapping_coordinator->receipt(0), nullptr);
  EXPECT_EQ(post_mapping_coordinator->report(0)->report_root(),
            post_mapping_reporter->report()->report_root());
  EXPECT_EQ(post_mapping_coordinator->receipt(0)->receipt_root(),
            post_mapping_receipt->receipt_root());
  EXPECT_EQ(post_mapping_coordinator->seal()->seal_root(),
            post_mapping_seal->seal_root());
  EXPECT_FALSE(post_mapping_coordinator->seal()->production_eligible());

  const auto* materialization_profile =
      startup->supervisor().capacity_admission();
  ASSERT_NE(materialization_profile, nullptr);
  const std::array materialization_allocation_authorities{
      materialization_allocation_authority(
          materialization_profile->device_ordinals()[0],
          materialization_profile->residency())};
  auto materialization_grant =
      compile_deepseek_rank_materialization_grant(
          *materialization_profile, startup->supervisor(),
          startup->manifests(), *post_mapping_coordinator, 0, 1100,
          materialization_allocation_authorities[0]);
  ASSERT_TRUE(materialization_grant.ok())
      << materialization_grant.status().message();
  const auto materialization_authority_root =
      compile_deepseek_rank_materialization_allocation_authority_root(
          materialization_allocation_authorities[0]);
  ASSERT_TRUE(materialization_authority_root.ok())
      << materialization_authority_root.status().message();
  EXPECT_EQ(materialization_grant->allocation_authority_root,
            *materialization_authority_root);
  EXPECT_EQ(kDeepSeekRankMaterializationGrantAbi,
            "pih_deepseek_rank_materialization_grant_v2");
  EXPECT_EQ(materialization_grant->gpu_family,
            RuntimeProfileGpuFamily::kH100Pcie80GiB);
  EXPECT_EQ(materialization_grant->residency,
            RuntimeProfileResidency::kHostSpill);
  EXPECT_FALSE(materialization_grant->production_eligible);
  EXPECT_EQ(materialization_grant->post_mapping_seal_root,
            post_mapping_seal->seal_root());
  auto materialization_frame =
      encode_deepseek_rank_materialization_grant(
          *materialization_grant);
  ASSERT_TRUE(materialization_frame.ok())
      << materialization_frame.status().message();
  EXPECT_EQ(kDeepSeekRankMaterializationGrantFrameAbi,
            "pih_deepseek_rank_materialization_grant_frame_v2");
  static_assert(kDeepSeekRankMaterializationGrantFrameBytes == 580);
  auto decoded_materialization_grant =
      decode_deepseek_rank_materialization_grant(
          *materialization_frame);
  ASSERT_TRUE(decoded_materialization_grant.ok())
      << decoded_materialization_grant.status().message();
  auto materialization_admission =
      DeepSeekRankMaterializationAdmission::Accept(
          startup->manifests()[0],
          *startup->supervisor().exec_ready(0), **mapping,
          *post_mapping_reporter->report(),
          std::move(*decoded_materialization_grant));
  ASSERT_TRUE(materialization_admission.ok())
      << materialization_admission.status().message();
  EXPECT_TRUE(
      validate_deepseek_rank_materialization_completion_worker_identity(
          *materialization_admission, *startup->supervisor().exec_ready(0))
          .ok());
  auto foreign_completion_ready = *startup->supervisor().exec_ready(0);
  ++foreign_completion_ready.challenge_identity;
  EXPECT_FALSE(
      validate_deepseek_rank_materialization_completion_worker_identity(
          *materialization_admission, foreign_completion_ready)
          .ok());
  EXPECT_EQ(materialization_admission->mapping_owner_root(),
            (*mapping)->mapping_owner_root());
  EXPECT_EQ(materialization_admission->post_mapping_seal_root(),
            post_mapping_seal->seal_root());
  EXPECT_NE(materialization_admission->grant_root(), Sha256Digest{});
  for (std::size_t index = 0; index < materialization_frame->size();
       ++index) {
    auto changed_frame = *materialization_frame;
    changed_frame[index] ^= std::byte{1};
    EXPECT_FALSE(decode_deepseek_rank_materialization_grant(
                     changed_frame)
                     .ok())
        << "accepted materialization grant mutation at byte " << index;
  }
  auto changed_grant = *materialization_grant;
  changed_grant.report_root = test_fixture::rank_capacity_digest(99);
  EXPECT_FALSE(DeepSeekRankMaterializationAdmission::Accept(
                   startup->manifests()[0],
                   *startup->supervisor().exec_ready(0), **mapping,
                   *post_mapping_reporter->report(),
                   std::move(changed_grant))
                   .ok());
  changed_grant = *materialization_grant;
  changed_grant.production_eligible = true;
  EXPECT_FALSE(DeepSeekRankMaterializationAdmission::Accept(
                   startup->manifests()[0],
                   *startup->supervisor().exec_ready(0), **mapping,
                   *post_mapping_reporter->report(),
                   std::move(changed_grant))
                   .ok());
  EXPECT_FALSE(compile_deepseek_rank_materialization_grant(
                   *materialization_profile, startup->supervisor(),
                   startup->manifests(), *post_mapping_coordinator, 0,
                   post_mapping_coordinator->deadline_ns(),
                   materialization_allocation_authorities[0])
                   .ok());

  FakeMaterializationExchangeState materialization_state;
  FakeMaterializationExchange materialization_exchange(
      materialization_state);
  auto materialization_receiver =
      DeepSeekRankMaterializationGrantReceiver::Create(
          startup->manifests()[0],
          *startup->supervisor().exec_ready(0), 9, 1100, **mapping,
          *post_mapping_reporter->report(), materialization_exchange);
  ASSERT_TRUE(materialization_receiver.ok())
      << materialization_receiver.status().message();
  auto materialization_coordinator =
      DeepSeekRankMaterializationGrantCoordinator::Create(
          *materialization_profile, startup->mutable_supervisor(),
          startup->manifests(), *post_mapping_coordinator, 1100,
          materialization_allocation_authorities, materialization_exchange);
  ASSERT_TRUE(materialization_coordinator.ok())
      << materialization_coordinator.status().message();
  EXPECT_EQ(kDeepSeekRankMaterializationExchangeAbi,
            "pih_deepseek_rank_materialization_exchange_v1");
  EXPECT_FALSE(materialization_receiver->take_admission().ok());
  EXPECT_EQ(materialization_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(materialization_state.grant_attempts, 1U);
  EXPECT_EQ(materialization_receiver->advance().code(),
            StatusCode::kUnavailable);
  ASSERT_TRUE(materialization_receiver->wait_event().has_value());
  EXPECT_EQ(*materialization_receiver->wait_event(),
            DeepSeekRankMaterializationGrantReceiverWaitEvent::
                kGrantReadable);
  EXPECT_EQ(materialization_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_EQ(materialization_state.grant_attempts, 2U);
  EXPECT_EQ(materialization_receiver->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_FALSE(materialization_receiver->admission_available());
  ASSERT_TRUE(materialization_receiver->wait_event().has_value());
  EXPECT_EQ(*materialization_receiver->wait_event(),
            DeepSeekRankMaterializationGrantReceiverWaitEvent::
                kAckWritable);
  EXPECT_TRUE(materialization_receiver->advance().ok());
  EXPECT_TRUE(materialization_receiver->complete());
  EXPECT_TRUE(materialization_receiver->admission_available());
  EXPECT_FALSE(materialization_receiver->wait_event().has_value());
  EXPECT_EQ(materialization_state.ack_attempts, 2U);
  ASSERT_TRUE(materialization_state.queued_ack.has_value());
  const auto acknowledged_frame = *materialization_state.queued_ack;
  EXPECT_TRUE(materialization_coordinator->advance().ok());
  EXPECT_TRUE(materialization_coordinator->complete());
  EXPECT_EQ(materialization_coordinator->acknowledgment_count(), 1U);
  ASSERT_NE(materialization_coordinator->grant(0), nullptr);
  ASSERT_NE(materialization_coordinator->acknowledgment(0), nullptr);
  const auto& warm_grant = *materialization_coordinator->grant(0);
  const auto& warm_layout = controller->expected_prefault_layout(0);
  auto warm_grant_root =
      compile_deepseek_rank_materialization_grant_root(warm_grant);
  ASSERT_TRUE(warm_grant_root.ok())
      << warm_grant_root.status().message();
  DeepSeekRankMaterializationCompletionFields warm_fields{
      1,
      warm_grant.engine_epoch,
      warm_grant.worker_generation,
      warm_grant.world_size,
      warm_grant.rank,
      warm_grant.process_manifest_identity,
      warm_grant.process_identity,
      warm_grant.pidfd_identity,
      warm_grant.control_identity,
      warm_grant.challenge_identity,
      warm_grant.device_ordinal,
      warm_grant.gpu_family,
      warm_grant.residency,
      warm_grant.production_eligible,
      warm_grant.dspark_enabled,
      1050,
      1060,
      warm_grant.deadline_ns,
      warm_layout.mapped_interval_bytes,
      warm_layout.selected_page_union_bytes,
      warm_layout.selected_page_union_bytes,
      9,
      9,
      8192,
      4,
      101,
      2 * DeepSeekExpertBundleLayout::kBundleBytes,
      202,
      2,
      2,
      0,
      warm_grant.profile_envelope_root,
      warm_grant.device_observation_root,
      warm_grant.capacity_plan_instance_root,
      warm_grant.post_mapping_seal_root,
      warm_grant.metadata_transaction_root,
      warm_grant.mapping_owner_root,
      *warm_grant_root,
      warm_layout.layout_root,
      test_fixture::rank_capacity_digest(80),
      test_fixture::rank_capacity_digest(81),
      test_fixture::rank_capacity_digest(82),
      test_fixture::rank_capacity_digest(83),
      test_fixture::rank_capacity_digest(84)};
  auto warm_frame =
      encode_deepseek_rank_materialization_completion(warm_fields);
  ASSERT_TRUE(warm_frame.ok()) << warm_frame.status().message();
  FakeMaterializationWarmupOperations warm_operations;
  warm_operations.queued_completion = std::vector<std::byte>(
      warm_frame->begin(), warm_frame->end());
  auto warmup = DeepSeekRankMaterializationWarmupCoordinator::Create(
      *materialization_coordinator, *controller, warm_operations);
  ASSERT_TRUE(warmup.ok()) << warmup.status().message();
  EXPECT_TRUE(warmup->advance().ok());
  EXPECT_TRUE(warmup->complete());
  EXPECT_FALSE(warmup->poisoned());
  EXPECT_EQ(warmup->completion_count(), 1U);
  ASSERT_NE(warmup->seal(), nullptr);
  EXPECT_EQ(warmup->seal()->world_size(), 1U);
  EXPECT_EQ(warmup->seal()->node_selected_page_union_bytes(),
            warm_layout.selected_page_union_bytes);
  EXPECT_NE(warmup->seal()->seal_root(), Sha256Digest{});
  EXPECT_EQ(warm_operations.poll_count, 1U);
  EXPECT_EQ(warm_operations.abort_count, 0U);
  EXPECT_EQ(kDeepSeekRankMaterializationGrantAckAbi,
            "pih_deepseek_rank_materialization_grant_ack_v1");
  EXPECT_EQ(kDeepSeekRankMaterializationGrantAckFrameAbi,
            "pih_deepseek_rank_materialization_grant_ack_frame_v1");
  static_assert(kDeepSeekRankMaterializationGrantAckFrameBytes == 232);
  auto dispatched_admission =
      materialization_receiver->take_admission();
  ASSERT_TRUE(dispatched_admission.ok())
      << dispatched_admission.status().message();
  EXPECT_EQ(dispatched_admission->grant_root(),
            materialization_coordinator->acknowledgment(0)->grant_root);
  auto authorized_capacity = DeepSeekPipelineCapacity::Create(
      1, 17, 9, 3, false);
  ASSERT_TRUE(authorized_capacity.ok())
      << authorized_capacity.status().message();
  EXPECT_TRUE(validate_deepseek_rank_authorized_materialization_source(
                  *dispatched_admission, **mapping,
                  *authorized_capacity)
                  .ok());
  auto foreign_capacity = DeepSeekPipelineCapacity::Create(
      1, 17, 9, 3, true);
  ASSERT_TRUE(foreign_capacity.ok())
      << foreign_capacity.status().message();
  EXPECT_FALSE(validate_deepseek_rank_authorized_materialization_source(
                   *dispatched_admission, **mapping, *foreign_capacity)
                   .ok());
  EXPECT_FALSE(materialization_receiver->take_admission().ok());
  for (std::size_t index = 0; index < acknowledged_frame.size(); ++index) {
    auto changed_frame = acknowledged_frame;
    changed_frame[index] ^= std::byte{1};
    EXPECT_FALSE(decode_deepseek_rank_materialization_grant_ack(
                     changed_frame)
                     .ok())
        << "accepted materialization ACK mutation at byte " << index;
  }
  EXPECT_FALSE(DeepSeekRankMaterializationGrantReceiver::Create(
                   startup->manifests()[0],
                   *startup->supervisor().exec_ready(0), 9,
                   post_mapping_reporter->report()->authority_deadline_ns(),
                   **mapping, *post_mapping_reporter->report(),
                   materialization_exchange)
                   .ok());

  auto deadline_startup =
      test_fixture::DeepSeekRankStartupFixture::Create(
          1, &projection, false);
  ASSERT_NE(deadline_startup, nullptr);
  FakePostMappingExchangeState deadline_state;
  deadline_state.now_ns = 1000;
  deadline_state.authority_unavailable_once = false;
  FakePostMappingCoordinatorChannel deadline_channel(deadline_state);
  auto deadline_coordinator =
      DeepSeekRankPostMappingResourceCoordinator::Create(
          deadline_startup->mutable_supervisor(),
          deadline_startup->manifests(), deadline_startup->spawn_plan(),
          deadline_startup->resource_plans(),
          deadline_startup->resource_seal(), *controller, 1000,
          deadline_channel);
  ASSERT_TRUE(deadline_coordinator.ok())
      << deadline_coordinator.status().message();
  EXPECT_EQ(deadline_coordinator->advance().code(),
            StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(deadline_coordinator->poisoned());
  EXPECT_TRUE(deadline_startup->supervisor().failed());
  EXPECT_EQ(deadline_state.authority_attempts, 0U);
  EXPECT_EQ(deadline_coordinator->advance().code(),
            StatusCode::kFailedPrecondition);

  auto changed = observation;
  changed.mapping_owner_root = test_fixture::rank_capacity_digest(99);
  EXPECT_FALSE(DeepSeekRankPostMappingResourceReceipt::Compile(
                   startup->supervisor(), startup->manifests(),
                   startup->spawn_plan(), startup->resource_plans(),
                   startup->resource_seal(), *controller, changed)
                   .ok());
  auto drifted_plans = startup->resource_plans();
  drifted_plans[0].worker_task_peak = 9;
  EXPECT_FALSE(DeepSeekRankPostMappingResourceReceipt::Compile(
                   startup->supervisor(), startup->manifests(),
                   startup->spawn_plan(), drifted_plans,
                   startup->resource_seal(), *controller, observation)
                   .ok());
  changed = observation;
  changed.resources.scm_rights_inflight_fd_count = 1;
  EXPECT_FALSE(DeepSeekRankPostMappingResourceReceipt::Compile(
                   startup->supervisor(), startup->manifests(),
                   startup->spawn_plan(), startup->resource_plans(),
                   startup->resource_seal(), *controller, changed)
                   .ok());
  changed = observation;
  changed.resources.vma_count = 201;
  const auto over_limit = DeepSeekRankPostMappingResourceReceipt::Compile(
      startup->supervisor(), startup->manifests(), startup->spawn_plan(),
      startup->resource_plans(), startup->resource_seal(), *controller,
      changed);
  EXPECT_EQ(over_limit.status().code(), StatusCode::kResourceExhausted);

  FakeMaterializationExchangeState late_materialization_state;
  late_materialization_state.now_ns = 1100;
  late_materialization_state.queued_grant =
      std::vector<std::byte>(materialization_frame->begin(),
                             materialization_frame->end());
  FakeMaterializationExchange late_materialization_exchange(
      late_materialization_state);
  auto late_receiver = DeepSeekRankMaterializationGrantReceiver::Create(
      startup->manifests()[0], *startup->supervisor().exec_ready(0), 9,
      1100, **mapping, *post_mapping_reporter->report(),
      late_materialization_exchange);
  ASSERT_TRUE(late_receiver.ok()) << late_receiver.status().message();
  EXPECT_EQ(late_receiver->advance().code(),
            StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(late_receiver->poisoned());

  FakeMaterializationExchangeState foreign_ack_state;
  foreign_ack_state.now_ns = 1101;
  foreign_ack_state.grant_unavailable_once = false;
  foreign_ack_state.ack_unavailable_once = false;
  FakeMaterializationExchange foreign_ack_exchange(foreign_ack_state);
  auto foreign_ack_receiver =
      DeepSeekRankMaterializationGrantReceiver::Create(
          startup->manifests()[0],
          *startup->supervisor().exec_ready(0), 9, 1200, **mapping,
          *post_mapping_reporter->report(), foreign_ack_exchange);
  ASSERT_TRUE(foreign_ack_receiver.ok())
      << foreign_ack_receiver.status().message();
  auto foreign_ack_coordinator =
      DeepSeekRankMaterializationGrantCoordinator::Create(
          *materialization_profile, startup->mutable_supervisor(),
          startup->manifests(), *post_mapping_coordinator, 1200,
          materialization_allocation_authorities, foreign_ack_exchange);
  ASSERT_TRUE(foreign_ack_coordinator.ok())
      << foreign_ack_coordinator.status().message();
  EXPECT_EQ(foreign_ack_coordinator->advance().code(),
            StatusCode::kUnavailable);
  EXPECT_TRUE(foreign_ack_receiver->advance().ok());
  ASSERT_TRUE(foreign_ack_state.queued_ack.has_value());
  auto foreign_ack = decode_deepseek_rank_materialization_grant_ack(
      *foreign_ack_state.queued_ack);
  ASSERT_TRUE(foreign_ack.ok()) << foreign_ack.status().message();
  foreign_ack->report_root = test_fixture::rank_capacity_digest(99);
  auto foreign_ack_frame =
      encode_deepseek_rank_materialization_grant_ack(*foreign_ack);
  ASSERT_TRUE(foreign_ack_frame.ok())
      << foreign_ack_frame.status().message();
  foreign_ack_state.queued_ack = std::vector<std::byte>(
      foreign_ack_frame->begin(), foreign_ack_frame->end());
  EXPECT_EQ(foreign_ack_coordinator->advance().code(),
            StatusCode::kFailedPrecondition);
  EXPECT_TRUE(foreign_ack_coordinator->poisoned());
  EXPECT_TRUE(startup->supervisor().failed());

  FakeArtifactPrefaultOperations prefault_operations;
  auto prefaulted = DeepSeekRankArtifactPrefaultTransaction::Run(
      DeepSeekRankAuthorizedMaterializationInputs(
          std::move(*dispatched_admission), std::move(*mapping)),
      prefault_operations);
  ASSERT_TRUE(prefaulted.ok()) << prefaulted.status().message();
  EXPECT_TRUE(validate_deepseek_rank_prefaulted_materialization_source(
                  *prefaulted)
                  .ok());
  EXPECT_EQ(prefault_operations.snapshot_count, 2U);
  EXPECT_EQ(prefault_operations.range_count,
            prefaulted->prefault_receipt.interval_count());
  EXPECT_EQ(prefaulted->prefault_receipt.resident_page_bytes_after(),
            prefaulted->prefault_receipt.selected_page_union_bytes());
  EXPECT_EQ(prefaulted->prefault_receipt.touched_page_count(),
            prefaulted->prefault_receipt.selected_page_union_bytes() /
                prefaulted->prefault_receipt.page_bytes());
  EXPECT_EQ(prefaulted->prefault_receipt.grant_root(),
            prefaulted->admission.grant_root());
  EXPECT_EQ(prefaulted->prefault_receipt.mapping_owner_root(),
            prefaulted->mapping_owner->mapping_owner_root());
  EXPECT_NE(prefaulted->prefault_receipt.receipt_root(), Sha256Digest{});
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataReceiverTest,
     RejectsIndependentDsparkMismatchBeforeFinalAck) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-dspark-splice-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  FakeDuplexArtifactChannel descriptor_channel(root);
  DeepSeekRankScmRightsInFlightLedger ledger;
  auto handoff = complete_descriptor_handoff(
      root, descriptor_channel, ledger);
  ASSERT_TRUE(handoff.ok()) << handoff.status().message();

  FakeMetadataChannelState state;
  FakeMetadataControllerOperations controller_operations(state);
  FakeMetadataReceiverOperations receiver_operations(state);
  auto controller =
      DeepSeekRankArtifactMetadataTransferTransaction::Create(
          std::move(handoff->first), controller_operations);
  ASSERT_TRUE(controller.ok()) << controller.status().message();
  auto receiver = DeepSeekRankArtifactMetadataReceiver::Create(
      std::move(handoff->second), true,
      kDeepSeekRankArtifactMetadataBlobMaximumBytes,
      receiver_operations);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(receiver->advance().ok());
  EXPECT_TRUE(receiver->poisoned());
  EXPECT_FALSE(receiver->complete());
  EXPECT_EQ(state.ack_send_attempts, 0U);
  EXPECT_FALSE(state.queued_ack.has_value());
  auto receipt = DeepSeekRankArtifactMetadataReceipt::Create(
      std::move(*receiver));
  EXPECT_FALSE(receipt.ok());
  std::filesystem::remove_all(root);
}

TEST(DeepSeekRankArtifactMetadataReceiverTest,
     RejectsReassemblyBeyondWorkerAuthorityBeforeAck) {
  const auto root = std::filesystem::temp_directory_path() /
                    "pih-artifact-metadata-capacity-test";
  std::filesystem::remove_all(root);
  test_fixture::write_reduced_target_generation_fixture(root);
  FakeDuplexArtifactChannel descriptor_channel(root);
  DeepSeekRankScmRightsInFlightLedger ledger;
  auto handoff = complete_descriptor_handoff(
      root, descriptor_channel, ledger);
  ASSERT_TRUE(handoff.ok()) << handoff.status().message();

  FakeMetadataChannelState state;
  FakeMetadataControllerOperations controller_operations(state);
  FakeMetadataReceiverOperations receiver_operations(state);
  auto controller =
      DeepSeekRankArtifactMetadataTransferTransaction::Create(
          std::move(handoff->first), controller_operations);
  ASSERT_TRUE(controller.ok()) << controller.status().message();
  ASSERT_GT(controller->blob_bytes(0), 1U);
  auto receiver = DeepSeekRankArtifactMetadataReceiver::Create(
      std::move(handoff->second), false,
      controller->blob_bytes(0) - 1U, receiver_operations);
  ASSERT_TRUE(receiver.ok()) << receiver.status().message();
  EXPECT_EQ(controller->advance().code(), StatusCode::kUnavailable);
  const auto status = receiver->advance();
  EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
  EXPECT_TRUE(receiver->poisoned());
  EXPECT_EQ(receiver->reassembled_bytes(), 0U);
  EXPECT_EQ(state.ack_send_attempts, 0U);
  std::filesystem::remove_all(root);
}

}  // namespace
}  // namespace pih
