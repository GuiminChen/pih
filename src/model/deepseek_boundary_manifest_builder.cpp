#include "pih/model/deepseek_boundary_manifest_builder.h"

namespace pih {
namespace {

Result<DeepSeekNcclP2pManifest> build_manifest(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::uint32_t world_size, std::uint32_t directed_boundary_id,
    std::uint64_t communicator_generation, std::uint32_t wire_token_count,
    DeepSeekNcclRole role, Buffer& buffer, std::uint64_t buffer_owner_id,
    std::uint64_t buffer_offset_bytes, std::uint64_t buffer_capacity_bytes,
    std::uintptr_t context_identity) {
  if (descriptor.engine_epoch == 0 || communicator_generation == 0 ||
      wire_token_count == 0 || buffer_owner_id == 0 ||
      context_identity == 0 || buffer.device().type() != DeviceType::kCuda ||
      buffer.device().index() !=
          static_cast<std::int32_t>(role == DeepSeekNcclRole::kSend
                                        ? directed_boundary_id
                                        : directed_boundary_id + 1)) {
    return Status::InvalidArgument(
        "DeepSeek boundary manifest identity is invalid");
  }
  auto ordinal = DeepSeekNcclOrdinalPlan::Create(
      world_size, descriptor.plan_sequence, directed_boundary_id, role);
  if (!ordinal.ok()) return ordinal.status();
  DeepSeekNcclP2pManifest manifest{
      .operation_plan_id = ordinal->global_issue_ordinal,
      .engine_epoch = descriptor.engine_epoch,
      .communicator_generation = communicator_generation,
      .pipeline_plan_sequence = descriptor.plan_sequence,
      .operation_ordinal = ordinal->rank_local_ordinal,
      .global_issue_ordinal = ordinal->global_issue_ordinal,
      .directed_boundary_id = directed_boundary_id,
      .role = role,
      .local_global_rank = ordinal->local_global_rank,
      .peer_global_rank = ordinal->peer_global_rank,
      .communicator_local_rank = role == DeepSeekNcclRole::kSend ? 0U : 1U,
      .communicator_peer_rank = role == DeepSeekNcclRole::kSend ? 1U : 0U,
      .buffer_owner_id = buffer_owner_id,
      .buffer_offset_bytes = buffer_offset_bytes,
      .buffer_capacity_bytes = buffer_capacity_bytes,
      .buffer_generation = buffer.generation(),
      .context_identity = context_identity,
      .token_count = wire_token_count};
  auto validated = DeepSeekNcclP2pPlan::Create(manifest);
  if (!validated.ok()) return validated.status();
  return manifest;
}

}  // namespace

Result<DeepSeekNcclP2pManifest>
DeepSeekBoundaryManifestBuilder::CreateSend(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::uint32_t world_size, std::uint32_t directed_boundary_id,
    std::uint64_t communicator_generation, std::uint32_t wire_token_count,
    const DeepSeekBoundarySendSource& source) {
  if (source.token_count() != wire_token_count) {
    return Status::InvalidArgument(
        "DeepSeek boundary send source token count does not match wire plan");
  }
  return build_manifest(
      descriptor, world_size, directed_boundary_id, communicator_generation,
      wire_token_count, DeepSeekNcclRole::kSend, source.buffer(),
      source.buffer_owner_id(), source.offset_bytes(),
      source.buffer().size_bytes(),
      source.context_identity());
}

Result<DeepSeekNcclP2pManifest>
DeepSeekBoundaryManifestBuilder::CreateRecv(
    const DeepSeekPipelinePlanDescriptor& descriptor,
    std::uint32_t world_size, std::uint32_t directed_boundary_id,
    std::uint64_t communicator_generation, std::uint32_t wire_token_count,
    Buffer& credit_slot, std::uint64_t buffer_owner_id,
    std::uintptr_t context_identity) {
  return build_manifest(
      descriptor, world_size, directed_boundary_id, communicator_generation,
      wire_token_count, DeepSeekNcclRole::kRecv, credit_slot, buffer_owner_id,
      0, credit_slot.size_bytes(), context_identity);
}

}  // namespace pih
