#include "pih/model/deepseek_rank_exec_identity_verifier.h"
#include "pih/model/deepseek_rank_control_codec.h"

#include <gtest/gtest.h>

namespace pih {
namespace {

Sha256Digest uuid_commitment() {
  Sha256Digest value{}; value.bytes.fill(std::byte{7}); return value;
}

DeepSeekRankExecChallenge challenge() {
  return {1, {7, 8, 2, 1, 10, 11, uuid_commitment(), 5, 200},
          {100, 200, 300}, 90, 400};
}

DeepSeekRankExecObservation observation() {
  return {{7, 8, 2, 1, 10, 11, {}, 5, 200}, 100, 90, 90, 10,
          uuid_commitment(), 5, 9, true, true};
}

DeepSeekRankPostExecResourceObservation resource_observation() {
  Sha256Digest capacity{};
  capacity.bytes.fill(std::byte{8});
  Sha256Digest envelope{};
  envelope.bytes.fill(std::byte{9});
  return {7, 8, 1, 100, 400, capacity, envelope, 10, 20, 0, 200,
          24, 100, 1000, 210, true};
}

DeepSeekRankPostExecResourceAuthority resource_authority() {
  Sha256Digest manifest{};
  manifest.bytes.fill(std::byte{5});
  Sha256Digest spawn{};
  spawn.bytes.fill(std::byte{6});
  Sha256Digest capacity{};
  capacity.bytes.fill(std::byte{8});
  Sha256Digest envelope{};
  envelope.bytes.fill(std::byte{9});
  return {1, 7, 8, 2, 1, 11, 100, 200, 300, 400,
          manifest, spawn, capacity, envelope, 300,
          {8, 20, 200, 2, 4, 10, envelope}};
}

TEST(DeepSeekRankExecIdentityVerifierTest, BindsExecObservationToChallenge) {
  auto verified = verify_deepseek_rank_exec_identity(challenge(), observation());
  ASSERT_TRUE(verified.ok()) << verified.status().message();
  EXPECT_EQ(verified->challenge_identity, 400U);
  EXPECT_EQ(verified->receipt.rank, 1U);
  EXPECT_EQ(verified->receipt.control_identity, 300U);
}

TEST(DeepSeekRankExecIdentityVerifierTest, RejectsEveryAuthorityDrift) {
  {
    auto value = observation(); ++value.argument_manifest.worker_generation;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); ++value.actual_parent_process_identity;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); ++value.controller_pidfd_target_identity;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); ++value.actual_physical_device_identity;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); value.actual_physical_device_uuid_commitment.bytes[0] = std::byte{8};
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); ++value.actual_startup_device_ordinal;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); value.parent_death_signal = 0;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); value.control_channel_is_seqpacket = false;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
  {
    auto value = observation(); value.challenge_received_on_control_channel = false;
    EXPECT_FALSE(verify_deepseek_rank_exec_identity(challenge(), value).ok());
  }
}

TEST(DeepSeekRankControlCodecTest, RoundTripsCanonicalChallengeAndReady) {
  const auto expected_challenge = challenge();
  const auto challenge_bytes = encode_deepseek_rank_challenge(expected_challenge);
  auto decoded_challenge = decode_deepseek_rank_challenge(challenge_bytes);
  ASSERT_TRUE(decoded_challenge.ok()) << decoded_challenge.status().message();
  EXPECT_EQ(decoded_challenge->manifest.process_manifest_identity, 11U);
  EXPECT_EQ(decoded_challenge->handle.control_identity, 300U);
  EXPECT_EQ(decoded_challenge->challenge_identity, 400U);
  EXPECT_EQ(decoded_challenge->manifest.startup_deadline_ns, 200U);
  EXPECT_EQ(decoded_challenge->manifest.physical_device_uuid_commitment,
            uuid_commitment());

  auto expected_ready = verify_deepseek_rank_exec_identity(
      expected_challenge, observation()).value();
  const auto ready_bytes = encode_deepseek_rank_ready(expected_ready);
  auto decoded_ready = decode_deepseek_rank_ready(ready_bytes);
  ASSERT_TRUE(decoded_ready.ok()) << decoded_ready.status().message();
  EXPECT_EQ(decoded_ready->receipt.process_identity, 100U);
  EXPECT_EQ(decoded_ready->challenge_identity, 400U);
  EXPECT_EQ(decoded_ready->receipt.startup_deadline_ns, 200U);
  EXPECT_EQ(decoded_ready->receipt.physical_device_uuid_commitment,
            uuid_commitment());
}

TEST(DeepSeekRankControlCodecTest, RejectsTruncationTypeAndVersionDrift) {
  auto bytes = encode_deepseek_rank_challenge(challenge());
  EXPECT_FALSE(decode_deepseek_rank_challenge(
      std::span<const std::byte>(bytes).first(bytes.size() - 1)).ok());
  EXPECT_FALSE(decode_deepseek_rank_challenge(
      std::span<const std::byte>(bytes).first(88)).ok());
  bytes[4] = std::byte{2};
  EXPECT_FALSE(decode_deepseek_rank_challenge(bytes).ok());
  bytes = encode_deepseek_rank_challenge(challenge());
  bytes[6] = std::byte{2};
  EXPECT_FALSE(decode_deepseek_rank_challenge(bytes).ok());
}

TEST(DeepSeekRankControlCodecTest,
     RoundTripsPostExecResourceObservationExactly) {
  EXPECT_EQ(kDeepSeekRankPostExecResourceObservationFrameAbi,
            "pih_deepseek_rank_post_exec_resource_observation_frame_v1");
  const auto expected = resource_observation();
  const auto bytes = encode_deepseek_rank_post_exec_resource_observation(
      expected);
  static_assert(bytes.size() ==
                kDeepSeekRankPostExecResourceObservationBytes);
  auto decoded = decode_deepseek_rank_post_exec_resource_observation(bytes);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->engine_epoch, expected.engine_epoch);
  EXPECT_EQ(decoded->worker_generation, expected.worker_generation);
  EXPECT_EQ(decoded->rank, expected.rank);
  EXPECT_EQ(decoded->process_identity, expected.process_identity);
  EXPECT_EQ(decoded->challenge_identity, expected.challenge_identity);
  EXPECT_EQ(decoded->acknowledged_capacity_plan_instance_root,
            expected.acknowledged_capacity_plan_instance_root);
  EXPECT_EQ(decoded->acknowledged_os_resource_envelope_root,
            expected.acknowledged_os_resource_envelope_root);
  EXPECT_EQ(decoded->task_count, expected.task_count);
  EXPECT_EQ(decoded->open_fd_count, expected.open_fd_count);
  EXPECT_EQ(decoded->scm_rights_inflight_fd_count,
            expected.scm_rights_inflight_fd_count);
  EXPECT_EQ(decoded->vma_count, expected.vma_count);
  EXPECT_EQ(decoded->rlimit_nofile_soft, expected.rlimit_nofile_soft);
  EXPECT_EQ(decoded->rlimit_nofile_hard, expected.rlimit_nofile_hard);
  EXPECT_EQ(decoded->fs_nr_open, expected.fs_nr_open);
  EXPECT_EQ(decoded->vm_max_map_count, expected.vm_max_map_count);
  EXPECT_EQ(decoded->non_dumpable, expected.non_dumpable);
}

TEST(DeepSeekRankControlCodecTest,
     RejectsPostExecFrameTruncationTypeVersionAndBooleanDrift) {
  auto bytes = encode_deepseek_rank_post_exec_resource_observation(
      resource_observation());
  EXPECT_FALSE(decode_deepseek_rank_post_exec_resource_observation(
                   std::span<const std::byte>(bytes).first(bytes.size() - 1))
                   .ok());
  bytes[4] = std::byte{2};
  EXPECT_FALSE(
      decode_deepseek_rank_post_exec_resource_observation(bytes).ok());
  bytes = encode_deepseek_rank_post_exec_resource_observation(
      resource_observation());
  bytes[6] = std::byte{2};
  EXPECT_FALSE(
      decode_deepseek_rank_post_exec_resource_observation(bytes).ok());
  bytes = encode_deepseek_rank_post_exec_resource_observation(
      resource_observation());
  bytes[172] = std::byte{2};
  EXPECT_FALSE(
      decode_deepseek_rank_post_exec_resource_observation(bytes).ok());
}

TEST(DeepSeekRankControlCodecTest,
     RoundTripsControllerResourceAuthorityExactly) {
  EXPECT_EQ(kDeepSeekRankPostExecResourceAuthorityFrameAbi,
            "pih_deepseek_rank_post_exec_resource_authority_frame_v1");
  const auto expected = resource_authority();
  const auto bytes = encode_deepseek_rank_post_exec_resource_authority(
      expected);
  static_assert(bytes.size() ==
                kDeepSeekRankPostExecResourceAuthorityBytes);
  auto decoded = decode_deepseek_rank_post_exec_resource_authority(bytes);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->protocol_version, 1U);
  EXPECT_EQ(decoded->world_size, expected.world_size);
  EXPECT_EQ(decoded->rank, expected.rank);
  EXPECT_EQ(decoded->process_manifest_identity,
            expected.process_manifest_identity);
  EXPECT_EQ(decoded->process_identity, expected.process_identity);
  EXPECT_EQ(decoded->pidfd_identity, expected.pidfd_identity);
  EXPECT_EQ(decoded->control_identity, expected.control_identity);
  EXPECT_EQ(decoded->challenge_identity, expected.challenge_identity);
  EXPECT_EQ(decoded->manifest_root, expected.manifest_root);
  EXPECT_EQ(decoded->spawn_resource_plan_root,
            expected.spawn_resource_plan_root);
  EXPECT_EQ(decoded->capacity_plan_instance_root,
            expected.capacity_plan_instance_root);
  EXPECT_EQ(decoded->os_resource_envelope_root,
            expected.os_resource_envelope_root);
  EXPECT_EQ(decoded->deadline_ns, expected.deadline_ns);
  EXPECT_EQ(decoded->resource_plan.worker_task_peak,
            expected.resource_plan.worker_task_peak);
  EXPECT_EQ(decoded->resource_plan.vma_emergency_reserve,
            expected.resource_plan.vma_emergency_reserve);
  EXPECT_EQ(decoded->resource_plan.os_resource_envelope_root,
            expected.os_resource_envelope_root);
}

TEST(DeepSeekRankControlCodecTest,
     RejectsResourceAuthorityTruncationTypeAndVersionDrift) {
  auto bytes = encode_deepseek_rank_post_exec_resource_authority(
      resource_authority());
  EXPECT_FALSE(decode_deepseek_rank_post_exec_resource_authority(
                   std::span<const std::byte>(bytes).first(bytes.size() - 1))
                   .ok());
  bytes[4] = std::byte{3};
  EXPECT_FALSE(
      decode_deepseek_rank_post_exec_resource_authority(bytes).ok());
  bytes = encode_deepseek_rank_post_exec_resource_authority(
      resource_authority());
  bytes[6] = std::byte{2};
  EXPECT_FALSE(
      decode_deepseek_rank_post_exec_resource_authority(bytes).ok());
}

}  // namespace
}  // namespace pih
