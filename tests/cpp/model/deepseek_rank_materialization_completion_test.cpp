#include "pih/model/deepseek_rank_materialization_completion.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_rank_startup_test_fixture.h"

namespace pih {
namespace {

class FakeCompletionSenderOperations final
    : public DeepSeekRankMaterializationCompletionSenderOperations {
 public:
  Status send_completion(
      std::int32_t control_fd,
      std::span<const std::byte> frame) override {
    ++send_attempts;
    if (control_fd != 9) {
      return Status::FailedPrecondition("completion control fd");
    }
    if (unavailable_once && send_attempts == 1) {
      first_frame.assign(frame.begin(), frame.end());
      return Status::Unavailable("completion backpressure");
    }
    if (!first_frame.empty() &&
        !std::equal(first_frame.begin(), first_frame.end(), frame.begin(),
                    frame.end())) {
      return Status::FailedPrecondition("completion retry changed");
    }
    accepted_frame.assign(frame.begin(), frame.end());
    return Status::Ok();
  }

  Result<std::uint64_t> monotonic_now_ns() override { return now_ns; }

  std::uint64_t now_ns = 250;
  bool unavailable_once = false;
  std::uint32_t send_attempts = 0;
  std::vector<std::byte> first_frame;
  std::vector<std::byte> accepted_frame;
};

DeepSeekRankMaterializationCompletionFields valid_completion(
    RuntimeProfileResidency residency = RuntimeProfileResidency::kHostSpill) {
  DeepSeekRankMaterializationCompletionFields fields;
  fields.protocol_version = 1;
  fields.engine_epoch = 7;
  fields.worker_generation = 8;
  fields.world_size = 4;
  fields.rank = 2;
  fields.process_manifest_identity = 11;
  fields.process_identity = 12;
  fields.pidfd_identity = 13;
  fields.control_identity = 14;
  fields.challenge_identity = 15;
  fields.device_ordinal = 2;
  fields.gpu_family = RuntimeProfileGpuFamily::kH100Pcie80GiB;
  fields.residency = residency;
  fields.production_eligible = true;
  fields.dspark_enabled = false;
  fields.prefault_completed_monotonic_ns = 100;
  fields.completion_monotonic_ns = 200;
  fields.deadline_ns = 300;
  fields.mapped_interval_bytes = 4097;
  fields.selected_page_union_bytes = 8192;
  fields.resident_selected_page_bytes = 8192;
  fields.prefault_major_fault_count = 9;
  fields.completion_major_fault_count = 9;
  fields.fixed_weight_backing_bytes = 8192;
  fields.fixed_weight_payload_bytes = 6000;
  fields.fixed_weight_allocation_generation = 21;
  if (residency == RuntimeProfileResidency::kHostSpill) {
    fields.pinned_staging_bytes = 2 * 13'369'344ULL;
    fields.pinned_staging_allocation_generation = 22;
    fields.expert_slot_count = 2;
    fields.staging_extent_count = 2;
    fields.pinned_allocation_root =
        test_fixture::rank_capacity_digest(13);
  }
  fields.profile_envelope_root = test_fixture::rank_capacity_digest(1);
  fields.device_observation_root = test_fixture::rank_capacity_digest(2);
  fields.capacity_plan_instance_root =
      test_fixture::rank_capacity_digest(3);
  fields.post_mapping_seal_root = test_fixture::rank_capacity_digest(4);
  fields.metadata_transaction_root = test_fixture::rank_capacity_digest(5);
  fields.mapping_owner_root = test_fixture::rank_capacity_digest(6);
  fields.grant_root = test_fixture::rank_capacity_digest(7);
  fields.prefault_layout_root = test_fixture::rank_capacity_digest(8);
  fields.prefault_receipt_root = test_fixture::rank_capacity_digest(9);
  fields.weight_layout_root = test_fixture::rank_capacity_digest(10);
  fields.weight_seal_root = test_fixture::rank_capacity_digest(11);
  fields.cuda_allocation_root = test_fixture::rank_capacity_digest(12);
  return fields;
}

TEST(DeepSeekRankMaterializationCompletionTest,
     RoundTripsExactHostSpillCompletion) {
  const auto fields = valid_completion();
  auto expected_root =
      compile_deepseek_rank_materialization_completion_root(fields);
  ASSERT_TRUE(expected_root.ok()) << expected_root.status().message();
  auto frame = encode_deepseek_rank_materialization_completion(fields);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  static_assert(kDeepSeekRankMaterializationCompletionFrameBytes == 656);
  EXPECT_EQ(kDeepSeekRankMaterializationCompletionAbi,
            "pih_deepseek_rank_materialization_completion_v1");
  EXPECT_EQ(kDeepSeekRankMaterializationCompletionFrameAbi,
            "pih_deepseek_rank_materialization_completion_frame_v1");

  auto decoded = decode_deepseek_rank_materialization_completion(*frame);
  ASSERT_TRUE(decoded.ok()) << decoded.status().message();
  EXPECT_EQ(decoded->rank, 2U);
  EXPECT_EQ(decoded->residency, RuntimeProfileResidency::kHostSpill);
  EXPECT_EQ(decoded->selected_page_union_bytes, 8192U);
  EXPECT_EQ(decoded->pinned_staging_bytes, 2 * 13'369'344ULL);
  EXPECT_EQ(
      compile_deepseek_rank_materialization_completion_root(*decoded).value(),
      *expected_root);
}

TEST(DeepSeekRankMaterializationCompletionTest,
     AcceptsCanonicalFullResidentCompletionWithoutPinnedState) {
  const auto fields =
      valid_completion(RuntimeProfileResidency::kFullResident);
  auto root = compile_deepseek_rank_materialization_completion_root(fields);
  ASSERT_TRUE(root.ok()) << root.status().message();
  auto frame = encode_deepseek_rank_materialization_completion(fields);
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  EXPECT_TRUE(decode_deepseek_rank_materialization_completion(*frame).ok());
}

TEST(DeepSeekRankMaterializationCompletionTest,
     RejectsTimelineResidencyFaultAndPagerDrift) {
  auto fields = valid_completion();
  fields.completion_monotonic_ns = fields.deadline_ns;
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_completion_root(fields).ok());

  fields = valid_completion();
  ++fields.completion_major_fault_count;
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_completion_root(fields).ok());

  fields = valid_completion();
  fields.resident_selected_page_bytes -= 4096;
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_completion_root(fields).ok());

  fields = valid_completion();
  fields.pager_transfer_reservations = 1;
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_completion_root(fields).ok());

  fields = valid_completion(RuntimeProfileResidency::kFullResident);
  fields.pinned_allocation_root = test_fixture::rank_capacity_digest(14);
  EXPECT_FALSE(
      compile_deepseek_rank_materialization_completion_root(fields).ok());
}

TEST(DeepSeekRankMaterializationCompletionTest,
     RejectsTamperedRootHeaderAndFrameSize) {
  auto frame =
      encode_deepseek_rank_materialization_completion(valid_completion());
  ASSERT_TRUE(frame.ok()) << frame.status().message();
  for (std::size_t index = 0; index < frame->size(); ++index) {
    auto changed = *frame;
    changed[index] ^= std::byte{0x01};
    EXPECT_FALSE(
        decode_deepseek_rank_materialization_completion(changed).ok())
        << "accepted completion mutation at byte " << index;
  }

  frame = encode_deepseek_rank_materialization_completion(valid_completion());
  ASSERT_TRUE(frame.ok());
  (*frame)[4] ^= std::byte{0x01};
  EXPECT_EQ(decode_deepseek_rank_materialization_completion(*frame)
                .status()
                .code(),
            StatusCode::kInvalidArgument);
  EXPECT_FALSE(decode_deepseek_rank_materialization_completion(
                   std::span<const std::byte>(frame->data(), frame->size() - 1U))
                   .ok());
}

TEST(DeepSeekRankMaterializationCompletionTest,
     SenderRetriesOnlyIdenticalUnacceptedFrameBeforeDeadline) {
  FakeCompletionSenderOperations operations;
  operations.unavailable_once = true;
  auto sender = DeepSeekRankMaterializationCompletionSender::Create(
      valid_completion(), 9, operations);
  ASSERT_TRUE(sender.ok()) << sender.status().message();
  EXPECT_EQ(sender->advance().code(), StatusCode::kUnavailable);
  EXPECT_FALSE(sender->complete());
  EXPECT_FALSE(sender->poisoned());
  EXPECT_TRUE(sender->advance().ok());
  EXPECT_TRUE(sender->complete());
  EXPECT_EQ(operations.send_attempts, 2U);
  EXPECT_EQ(operations.first_frame, operations.accepted_frame);
  EXPECT_TRUE(decode_deepseek_rank_materialization_completion(
                  operations.accepted_frame)
                  .ok());
  EXPECT_TRUE(sender->advance().ok());
  EXPECT_EQ(operations.send_attempts, 2U);
}

TEST(DeepSeekRankMaterializationCompletionTest,
     SenderRejectsDeadlineEqualityBeforeAnyWrite) {
  FakeCompletionSenderOperations operations;
  operations.now_ns = 300;
  auto sender = DeepSeekRankMaterializationCompletionSender::Create(
      valid_completion(), 9, operations);
  ASSERT_TRUE(sender.ok()) << sender.status().message();
  EXPECT_EQ(sender->advance().code(), StatusCode::kDeadlineExceeded);
  EXPECT_TRUE(sender->poisoned());
  EXPECT_EQ(operations.send_attempts, 0U);
  EXPECT_EQ(sender->advance().code(), StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
