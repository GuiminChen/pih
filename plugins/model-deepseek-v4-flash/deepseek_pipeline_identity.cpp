#include "pih/model/deepseek_pipeline_identity.h"

#include <cstddef>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool same_stage(const DeepSeekStagePlan& left,
                const DeepSeekStagePlan& right) noexcept {
  return left.rank == right.rank && left.layers == right.layers &&
         left.owns_embedding == right.owns_embedding &&
         left.owns_lm_head == right.owns_lm_head &&
         left.owns_dspark == right.owns_dspark;
}

Status validate_plan(const DeepSeekPipelinePlan& plan) {
  const auto world_size = plan.world_size();
  if (world_size < 1 || world_size > 4 ||
      plan.ranges().size() != world_size) {
    return Status::InvalidArgument(
        "DeepSeek pipeline plan is not canonical");
  }
  const bool dspark_enabled = plan.rank(world_size - 1).owns_dspark;
  auto expected = DeepSeekPipelinePlan::Create(world_size, dspark_enabled);
  if (!expected.ok()) return expected.status();
  for (std::uint32_t rank = 0; rank < world_size; ++rank) {
    if (!(plan.ranges()[rank] == expected->ranges()[rank]) ||
        !same_stage(plan.rank(rank), expected->rank(rank))) {
      return Status::InvalidArgument(
          "DeepSeek pipeline plan is not canonical");
    }
  }
  return Status::Ok();
}

bool same_capacity(const DeepSeekPipelineCapacity& left,
                   const DeepSeekPipelineCapacity& right) noexcept {
  if (left.world_size != right.world_size ||
      left.max_pipeline_tokens != right.max_pipeline_tokens ||
      left.expert_tokens != right.expert_tokens ||
      left.max_sequences != right.max_sequences ||
      left.max_prefill_chunk_tokens != right.max_prefill_chunk_tokens ||
      left.max_decode_sequences != right.max_decode_sequences ||
      left.max_verify_sequences != right.max_verify_sequences ||
      left.dspark_enabled != right.dspark_enabled ||
      left.boundary_slot_bytes != right.boundary_slot_bytes ||
      left.control_payload_bytes != right.control_payload_bytes ||
      left.expert_workspace_bytes != right.expert_workspace_bytes ||
      left.ranks.size() != right.ranks.size()) {
    return false;
  }
  for (std::size_t rank = 0; rank < left.ranks.size(); ++rank) {
    if (left.ranks[rank].recv_bytes != right.ranks[rank].recv_bytes ||
        left.ranks[rank].output_hold_bytes !=
            right.ranks[rank].output_hold_bytes) {
      return false;
    }
  }
  return true;
}

Status validate_capacity(const DeepSeekPipelineCapacity& capacity) {
  auto expected = DeepSeekPipelineCapacity::Create(
      capacity.world_size, capacity.max_prefill_chunk_tokens,
      capacity.max_decode_sequences, capacity.max_verify_sequences,
      capacity.dspark_enabled);
  if (!expected.ok() || !same_capacity(capacity, *expected)) {
    return Status::InvalidArgument(
        "DeepSeek pipeline capacity is not canonical");
  }
  return Status::Ok();
}

}  // namespace

Result<Sha256Digest> compile_deepseek_pipeline_plan_root(
    const DeepSeekPipelinePlan& plan) {
  auto status = validate_plan(plan);
  if (!status.ok()) return status;

  const auto world_size = plan.world_size();
  const bool dspark_enabled = plan.rank(world_size - 1).owns_dspark;
  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-pipeline-plan:v1", world_size + 2U);
  if (!collection.ok()) return collection.status();
  status = collection->add_u32(1, world_size);
  if (status.ok()) {
    status = collection->add_u32(2, dspark_enabled ? 1U : 0U);
  }
  for (std::uint32_t rank = 0; status.ok() && rank < world_size; ++rank) {
    const auto& stage = plan.rank(rank);
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-pipeline-stage-plan:v1", 6);
    if (!record.ok()) return record.status();
    status = record->add_u32(1, stage.rank);
    if (status.ok()) status = record->add_u32(2, stage.layers.first_layer);
    if (status.ok()) status = record->add_u32(3, stage.layers.last_layer);
    if (status.ok()) {
      status = record->add_u32(4, stage.owns_embedding ? 1U : 0U);
    }
    if (status.ok()) {
      status = record->add_u32(5, stage.owns_lm_head ? 1U : 0U);
    }
    if (status.ok()) {
      status = record->add_u32(6, stage.owns_dspark ? 1U : 0U);
    }
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = collection->add_hash(
        static_cast<std::uint16_t>(rank + 3U), *root);
  }
  if (!status.ok()) return status;
  return collection->finalize();
}

Result<Sha256Digest> compile_deepseek_pipeline_capacity_root(
    const DeepSeekPipelineCapacity& capacity) {
  auto status = validate_capacity(capacity);
  if (!status.ok()) return status;

  auto collection = CanonicalHashBuilder::Create(
      "pih:deepseek-pipeline-capacity:v1",
      capacity.world_size + 12U);
  if (!collection.ok()) return collection.status();
  status = collection->add_u32(1, DeepSeekPipelineCapacity::kBoundaryCredits);
  if (status.ok()) status = collection->add_u32(2, capacity.world_size);
  if (status.ok()) {
    status = collection->add_u32(3, capacity.max_pipeline_tokens);
  }
  if (status.ok()) status = collection->add_u32(4, capacity.expert_tokens);
  if (status.ok()) status = collection->add_u32(5, capacity.max_sequences);
  if (status.ok()) {
    status = collection->add_u32(6, capacity.max_prefill_chunk_tokens);
  }
  if (status.ok()) {
    status = collection->add_u32(7, capacity.max_decode_sequences);
  }
  if (status.ok()) {
    status = collection->add_u32(8, capacity.max_verify_sequences);
  }
  if (status.ok()) {
    status = collection->add_u32(9, capacity.dspark_enabled ? 1U : 0U);
  }
  if (status.ok()) {
    status = collection->add_u64(10, capacity.boundary_slot_bytes);
  }
  if (status.ok()) {
    status = collection->add_u64(11, capacity.control_payload_bytes);
  }
  if (status.ok()) {
    status = collection->add_u64(12, capacity.expert_workspace_bytes);
  }
  for (std::uint32_t rank = 0;
       status.ok() && rank < capacity.world_size; ++rank) {
    const auto& rank_capacity = capacity.ranks[rank];
    auto record = CanonicalHashBuilder::Create(
        "pih:deepseek-pipeline-rank-capacity:v1", 3);
    if (!record.ok()) return record.status();
    status = record->add_u32(1, rank);
    if (status.ok()) status = record->add_u64(2, rank_capacity.recv_bytes);
    if (status.ok()) {
      status = record->add_u64(3, rank_capacity.output_hold_bytes);
    }
    if (!status.ok()) return status;
    auto root = record->finalize();
    if (!root.ok()) return root.status();
    status = collection->add_hash(
        static_cast<std::uint16_t>(rank + 13U), *root);
  }
  if (!status.ok()) return status;
  return collection->finalize();
}

}  // namespace pih
