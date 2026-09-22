#include "pih/model/deepseek_rank_materialization_warmup.h"
#include "pih/model/deepseek_rank_serving_protocol.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "deepseek_rank_startup_test_fixture.h"

namespace pih {
namespace {

DeepSeekRankMaterializationGrantFields make_grant(
    std::uint32_t rank, std::uint32_t world_size,
    const Sha256Digest& metadata_root) {
  DeepSeekRankMaterializationGrantFields grant;
  grant.protocol_version = 2;
  grant.engine_epoch = 7;
  grant.worker_generation = 8;
  grant.world_size = world_size;
  grant.rank = rank;
  grant.process_manifest_identity = 100 + rank;
  grant.process_identity = 200 + rank;
  grant.pidfd_identity = 300 + rank;
  grant.control_identity = 400 + rank;
  grant.challenge_identity = 500 + rank;
  grant.device_ordinal = static_cast<std::int32_t>(rank);
  grant.gpu_family = RuntimeProfileGpuFamily::kH100Pcie80GiB;
  grant.residency = RuntimeProfileResidency::kHostSpill;
  grant.production_eligible = true;
  grant.deadline_ns = 1000;
  grant.profile_envelope_root = test_fixture::rank_capacity_digest(1);
  grant.device_observation_root = test_fixture::rank_capacity_digest(2);
  grant.kernel_closure_root = test_fixture::rank_capacity_digest(3);
  grant.graph_snapshot_root = test_fixture::rank_capacity_digest(4);
  grant.capacity_plan_instance_root = test_fixture::rank_capacity_digest(5);
  grant.first_resource_seal_root = test_fixture::rank_capacity_digest(6);
  grant.metadata_transaction_root = metadata_root;
  grant.mapping_owner_root = test_fixture::rank_capacity_digest(20 + rank);
  grant.report_root = test_fixture::rank_capacity_digest(30 + rank);
  grant.post_mapping_receipt_root =
      test_fixture::rank_capacity_digest(40 + rank);
  grant.post_mapping_seal_root = test_fixture::rank_capacity_digest(7);
  grant.mapping_owner_set_root = test_fixture::rank_capacity_digest(8);
  grant.receipt_set_root = test_fixture::rank_capacity_digest(9);
  grant.allocation_authority_root =
      test_fixture::rank_capacity_digest(100 + rank);
  return grant;
}

DeepSeekRankMappingPlan make_mapping(std::uint32_t rank) {
  DeepSeekRankMappingPlan mapping;
  mapping.rank = rank;
  mapping.owned_tensor_count = 1;
  mapping.logical_tensor_bytes = 1;
  mapping.mapped_interval_bytes = 4097;
  mapping.shards.push_back({"shared.safetensors", 8192});
  mapping.intervals.push_back({"shared.safetensors", 0, 4097});
  return mapping;
}

DeepSeekRankMaterializationCompletionFields make_completion(
    const DeepSeekRankMaterializationGrantFields& grant,
    const DeepSeekRankArtifactPrefaultLayout& layout) {
  DeepSeekRankMaterializationCompletionFields completion;
  completion.protocol_version = 1;
  completion.engine_epoch = grant.engine_epoch;
  completion.worker_generation = grant.worker_generation;
  completion.world_size = grant.world_size;
  completion.rank = grant.rank;
  completion.process_manifest_identity = grant.process_manifest_identity;
  completion.process_identity = grant.process_identity;
  completion.pidfd_identity = grant.pidfd_identity;
  completion.control_identity = grant.control_identity;
  completion.challenge_identity = grant.challenge_identity;
  completion.device_ordinal = grant.device_ordinal;
  completion.gpu_family = grant.gpu_family;
  completion.residency = grant.residency;
  completion.production_eligible = grant.production_eligible;
  completion.dspark_enabled = grant.dspark_enabled;
  completion.prefault_completed_monotonic_ns = 100 + grant.rank;
  completion.completion_monotonic_ns = 200 + grant.rank;
  completion.deadline_ns = grant.deadline_ns;
  completion.mapped_interval_bytes = layout.mapped_interval_bytes;
  completion.selected_page_union_bytes = layout.selected_page_union_bytes;
  completion.resident_selected_page_bytes = layout.selected_page_union_bytes;
  completion.prefault_major_fault_count = 3;
  completion.completion_major_fault_count = 3;
  completion.fixed_weight_backing_bytes = 8192;
  completion.fixed_weight_payload_bytes = 4096;
  completion.fixed_weight_allocation_generation = 600 + grant.rank;
  completion.pinned_staging_bytes = 2 * 13'369'344ULL;
  completion.pinned_staging_allocation_generation = 700 + grant.rank;
  completion.expert_slot_count = 2;
  completion.staging_extent_count = 2;
  completion.profile_envelope_root = grant.profile_envelope_root;
  completion.device_observation_root = grant.device_observation_root;
  completion.capacity_plan_instance_root = grant.capacity_plan_instance_root;
  completion.post_mapping_seal_root = grant.post_mapping_seal_root;
  completion.metadata_transaction_root = grant.metadata_transaction_root;
  completion.mapping_owner_root = grant.mapping_owner_root;
  completion.grant_root =
      compile_deepseek_rank_materialization_grant_root(grant).value();
  completion.prefault_layout_root = layout.layout_root;
  completion.prefault_receipt_root =
      test_fixture::rank_capacity_digest(50 + grant.rank);
  completion.weight_layout_root =
      test_fixture::rank_capacity_digest(60 + grant.rank);
  completion.weight_seal_root =
      test_fixture::rank_capacity_digest(70 + grant.rank);
  completion.cuda_allocation_root =
      test_fixture::rank_capacity_digest(80 + grant.rank);
  completion.pinned_allocation_root =
      test_fixture::rank_capacity_digest(90 + grant.rank);
  return completion;
}

struct WarmInputs final {
  Sha256Digest metadata_root{};
  std::vector<DeepSeekRankMappingPlan> mappings;
  std::vector<DeepSeekRankArtifactPrefaultLayout> rank_layouts;
  DeepSeekNodeArtifactPrefaultLayout node_layout;
  std::vector<DeepSeekRankMaterializationGrantFields> grants;
  std::vector<DeepSeekRankMaterializationCompletionFields> completions;
};

WarmInputs make_inputs(std::uint32_t world_size) {
  WarmInputs inputs;
  inputs.metadata_root = test_fixture::rank_capacity_digest(10);
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    inputs.mappings.push_back(make_mapping(rank));
  }
  inputs.node_layout =
      compile_deepseek_node_artifact_prefault_layout(inputs.mappings).value();
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    inputs.rank_layouts.push_back(
        compile_deepseek_rank_artifact_prefault_layout(
            inputs.mappings[rank])
            .value());
    inputs.grants.push_back(make_grant(rank, world_size,
                                       inputs.metadata_root));
    inputs.completions.push_back(make_completion(
        inputs.grants.back(), inputs.rank_layouts.back()));
  }
  return inputs;
}

TEST(DeepSeekRankMaterializationWarmupTest,
     DerivesServingBindingOnlyFromProductionWarmSealAndReadyReceipt) {
  auto inputs = make_inputs(2);
  auto seal = DeepSeekRankMaterializationWarmSeal::Compile(
      inputs.grants, inputs.rank_layouts, inputs.node_layout,
      inputs.metadata_root, inputs.completions);
  ASSERT_TRUE(seal.ok()) << seal.status().message();
  DeepSeekRankReadyReceipt ready{7, 8, 1, 301, 101, 201, 301, 401,
                                  test_fixture::rank_capacity_digest(1), 1,
                                  1000};

  auto binding = compile_deepseek_rank_serving_session_binding(ready, *seal);
  ASSERT_TRUE(binding.ok()) << binding.status().message();
  EXPECT_EQ(binding->engine_epoch, seal->engine_epoch());
  EXPECT_EQ(binding->worker_generation, seal->worker_generation());
  EXPECT_EQ(binding->world_size, seal->world_size());
  EXPECT_EQ(binding->rank, ready.rank);
  EXPECT_EQ(binding->process_identity, ready.process_identity);
  EXPECT_EQ(binding->pidfd_identity, ready.pidfd_identity);
  EXPECT_EQ(binding->control_identity, ready.control_identity);
  EXPECT_EQ(binding->materialization_warm_seal_root, seal->seal_root());

  ready.worker_generation++;
  EXPECT_EQ(compile_deepseek_rank_serving_session_binding(ready, *seal)
                .status()
                .code(),
            StatusCode::kFailedPrecondition);
}

TEST(DeepSeekRankMaterializationWarmupTest,
     DerivesAllServingSessionsOnlyFromTheCompleteReadySet) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto inputs = make_inputs(world_size);
    auto seal = DeepSeekRankMaterializationWarmSeal::Compile(
        inputs.grants, inputs.rank_layouts, inputs.node_layout,
        inputs.metadata_root, inputs.completions);
    ASSERT_TRUE(seal.ok()) << seal.status().message();
    std::vector<DeepSeekRankReadyReceipt> receipts;
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      receipts.push_back({7, 8, rank, 300 + rank, 100 + rank, 200 + rank,
                          300 + rank, 400 + rank,
                          test_fixture::rank_capacity_digest(1 + rank),
                          static_cast<std::int32_t>(rank), 1000});
    }
    auto sessions = compile_deepseek_rank_serving_session_set(receipts, *seal);
    ASSERT_TRUE(sessions.ok()) << sessions.status().message();
    ASSERT_EQ(sessions->size(), world_size);
    for (std::uint32_t rank = 0; rank < world_size; ++rank) {
      EXPECT_EQ((*sessions)[rank].rank, rank);
      EXPECT_EQ((*sessions)[rank].materialization_warm_seal_root,
                seal->seal_root());
    }

    if (world_size > 1) {
      std::swap(receipts[0], receipts[1]);
      EXPECT_EQ(compile_deepseek_rank_serving_session_set(receipts, *seal)
                    .status()
                    .code(),
                StatusCode::kFailedPrecondition);
    }
  }
}

TEST(DeepSeekRankMaterializationWarmupTest,
     SealsEverySupportedWorldSizeAndDeduplicatesSharedPages) {
  for (std::uint32_t world_size = 1; world_size <= 4; ++world_size) {
    auto inputs = make_inputs(world_size);
    auto seal = DeepSeekRankMaterializationWarmSeal::Compile(
        inputs.grants, inputs.rank_layouts, inputs.node_layout,
        inputs.metadata_root, inputs.completions);
    ASSERT_TRUE(seal.ok()) << "world=" << world_size << ": "
                           << seal.status().message();
    EXPECT_EQ(kDeepSeekRankMaterializationWarmupAbi,
              "pih_deepseek_rank_materialization_warmup_v1");
    EXPECT_EQ(seal->world_size(), world_size);
    EXPECT_EQ(seal->summed_rank_selected_page_bytes(),
              world_size * 8192ULL);
    EXPECT_EQ(seal->node_selected_page_union_bytes(), 8192U);
    EXPECT_EQ(seal->node_duplicate_selected_page_bytes(),
              (world_size - 1ULL) * 8192ULL);
    EXPECT_EQ(seal->fixed_weight_backing_bytes(),
              world_size * 8192ULL);
    EXPECT_EQ(seal->pinned_staging_bytes(),
              world_size * 2ULL * 13'369'344ULL);
    EXPECT_TRUE(seal->production_eligible());
    auto grant_set =
        compile_deepseek_rank_materialization_warm_grant_set_root(
            inputs.grants);
    ASSERT_TRUE(grant_set.ok()) << grant_set.status().message();
    EXPECT_EQ(seal->grant_set_root(), *grant_set);
    EXPECT_NE(seal->completion_set_root(), Sha256Digest{});
    EXPECT_NE(seal->seal_root(), Sha256Digest{});
  }
}

TEST(DeepSeekRankMaterializationWarmupTest,
     RejectsForeignCompletionAndNodeAccountingDrift) {
  auto inputs = make_inputs(4);
  ++inputs.completions[2].process_identity;
  EXPECT_EQ(DeepSeekRankMaterializationWarmSeal::Compile(
                inputs.grants, inputs.rank_layouts, inputs.node_layout,
                inputs.metadata_root, inputs.completions)
                .status()
                .code(),
            StatusCode::kFailedPrecondition);

  inputs = make_inputs(4);
  ++inputs.node_layout.node_duplicate_selected_page_bytes;
  EXPECT_EQ(DeepSeekRankMaterializationWarmSeal::Compile(
                inputs.grants, inputs.rank_layouts, inputs.node_layout,
                inputs.metadata_root, inputs.completions)
                .status()
                .code(),
            StatusCode::kFailedPrecondition);
}

TEST(DeepSeekRankMaterializationWarmupTest,
     RejectsMissingAndReorderedRanks) {
  auto inputs = make_inputs(2);
  inputs.completions.pop_back();
  EXPECT_EQ(DeepSeekRankMaterializationWarmSeal::Compile(
                inputs.grants, inputs.rank_layouts, inputs.node_layout,
                inputs.metadata_root, inputs.completions)
                .status()
                .code(),
            StatusCode::kInvalidArgument);

  inputs = make_inputs(2);
  std::swap(inputs.completions[0], inputs.completions[1]);
  EXPECT_EQ(DeepSeekRankMaterializationWarmSeal::Compile(
                inputs.grants, inputs.rank_layouts, inputs.node_layout,
                inputs.metadata_root, inputs.completions)
                .status()
                .code(),
            StatusCode::kFailedPrecondition);
}

}  // namespace
}  // namespace pih
