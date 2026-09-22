#include "pih/model/deepseek_weight_materialization_plan.h"

#include <algorithm>
#include <map>
#include <set>
#include <tuple>

#include "pih/core/checked_math.h"

namespace pih {
namespace {

using SourceKey = std::tuple<std::string_view, std::uint64_t, std::uint64_t>;

bool zero_digest(const Sha256Digest& digest) {
  return std::ranges::all_of(
      digest.bytes, [](std::byte value) { return value == std::byte{0}; });
}

bool direct_semantics_match(DType dtype,
                            DeepSeekStorageSemantics semantics) {
  switch (dtype) {
    case DType::kFloat32:
      return semantics ==
             DeepSeekStorageSemantics::kDirectF32LittleEndianBits;
    case DType::kFloat16:
      return semantics ==
             DeepSeekStorageSemantics::kDirectF16LittleEndianBits;
    case DType::kBFloat16:
      return semantics ==
             DeepSeekStorageSemantics::kDirectBf16LittleEndianBits;
    case DType::kInt8:
      return semantics ==
             DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits;
    case DType::kUInt8:
      return semantics == DeepSeekStorageSemantics::kDirectU8Bits;
    case DType::kFloat8E4M3:
      return semantics == DeepSeekStorageSemantics::kDirectFp8E4m3Bits;
    case DType::kFloat8E8M0:
      return semantics == DeepSeekStorageSemantics::kDirectUe8m0ScaleBits;
    case DType::kInt32:
      return semantics ==
             DeepSeekStorageSemantics::kDirectI32LittleEndianBits;
    case DType::kInt64:
      return semantics ==
             DeepSeekStorageSemantics::kDirectI64LittleEndianBits;
    case DType::kBool:
      return semantics == DeepSeekStorageSemantics::kDirectBoolBits;
    case DType::kFloat64:
    case DType::kUInt32:
      return false;
  }
  return false;
}

}  // namespace

Result<DeepSeekWeightMaterializationPlan>
DeepSeekWeightMaterializationPlan::Create(
    const DeepSeekRankWeightDispositionManifest& disposition,
    const DeepSeekExpertBundleManifest& experts,
    std::span<const DeepSeekRankTensorRecord> rank_tensors) {
  if (rank_tensors.empty() || disposition.records().size() != rank_tensors.size()) {
    return Status::InvalidArgument("DeepSeek materialization source set differs from disposition");
  }
  std::map<std::string_view, const DeepSeekRankTensorRecord*, std::less<>> by_name;
  std::map<SourceKey, const DeepSeekRankTensorRecord*> by_source;
  bool saw_target_authority = false;
  bool saw_legacy_source = false;
  Sha256Digest artifact_root;
  Sha256Digest layout_root;
  Sha256Digest disposition_root;
  std::set<std::string, std::less<>> runtime_roots;
  for (const auto& tensor : rank_tensors) {
    const auto bytes = tensor.file_end > tensor.file_begin
                           ? tensor.file_end - tensor.file_begin
                           : 0;
    if (tensor.tensor_name.empty() || tensor.shard_name.empty() || bytes == 0 ||
        !by_name.emplace(tensor.tensor_name, &tensor).second ||
        !by_source.emplace(SourceKey{tensor.shard_name, tensor.file_begin, bytes},
                           &tensor).second) {
      return Status::InvalidArgument("DeepSeek materialization source identity is invalid");
    }
    const bool any_authority =
        !zero_digest(tensor.artifact_root) ||
        !zero_digest(tensor.layout_root) ||
        !zero_digest(tensor.disposition_root) ||
        !zero_digest(tensor.target_logical_root) ||
        !zero_digest(tensor.disposition_record_root) ||
        !zero_digest(tensor.layout_record_root) ||
        !zero_digest(tensor.runtime_record_root) ||
        tensor.storage_semantics != DeepSeekStorageSemantics{};
    if (!any_authority) {
      saw_legacy_source = true;
      continue;
    }
    if (zero_digest(tensor.artifact_root) ||
        zero_digest(tensor.layout_root) ||
        zero_digest(tensor.disposition_root) ||
        zero_digest(tensor.target_logical_root) ||
        zero_digest(tensor.disposition_record_root) ||
        zero_digest(tensor.layout_record_root) ||
        zero_digest(tensor.runtime_record_root) ||
        tensor.tensor_bytes != bytes ||
        !direct_semantics_match(tensor.dtype, tensor.storage_semantics)) {
      return Status::InvalidArgument(
          "DeepSeek target materialization authority is incomplete");
    }
    if (!saw_target_authority) {
      artifact_root = tensor.artifact_root;
      layout_root = tensor.layout_root;
      disposition_root = tensor.disposition_root;
    } else if (tensor.artifact_root != artifact_root ||
               tensor.layout_root != layout_root ||
               tensor.disposition_root != disposition_root) {
      return Status::InvalidArgument(
          "DeepSeek target materialization roots disagree within rank");
    }
    saw_target_authority = true;
    if (!runtime_roots.emplace(tensor.runtime_record_root.hex()).second) {
      return Status::InvalidArgument(
          "DeepSeek target materialization runtime record is duplicated");
    }
  }
  if (saw_target_authority && saw_legacy_source) {
    return Status::InvalidArgument(
        "DeepSeek target and legacy materialization authorities are mixed");
  }

  bool has_materialized_expert = false;
  bool has_paged_expert = false;
  for (const auto& record : disposition.records()) {
    const auto source = by_name.find(record.tensor_name);
    if (source == by_name.end() ||
        source->second->file_end - source->second->file_begin != record.source_bytes) {
      return Status::InvalidArgument("DeepSeek disposition does not match source tensor");
    }
    if (record.expert_identity.has_value()) {
      has_materialized_expert |=
          record.kind == DeepSeekWeightDispositionKind::kMaterializeFixed;
      has_paged_expert |=
          record.kind == DeepSeekWeightDispositionKind::kPagedSource;
    } else if (record.kind != DeepSeekWeightDispositionKind::kMaterializeFixed) {
      return Status::InvalidArgument("DeepSeek non-expert tensor cannot be paged source");
    }
  }
  if (has_materialized_expert && has_paged_expert) {
    return Status::InvalidArgument("DeepSeek expert residency is mixed within one rank");
  }

  DeepSeekWeightMaterializationPlan result;
  result.owner_rank_ = disposition.owner_rank();
  result.paged_source_bytes_ = disposition.paged_source_bytes();
  result.target_authority_bound_ = saw_target_authority;
  result.artifact_root_ = artifact_root;
  result.layout_root_ = layout_root;
  result.disposition_root_ = disposition_root;
  std::uint64_t cursor = 0;
  for (const auto& record : disposition.records()) {
    if (record.expert_identity.has_value()) continue;
    const auto* source = by_name.at(record.tensor_name);
    auto aligned = checked_align_up_u64(cursor, kFinalAlignment);
    if (!aligned.ok()) return aligned.status();
    cursor = *aligned;
    result.copies_.push_back(
        {source->tensor_name, source->shard_name, source->dtype, source->shape,
         source->file_begin, cursor, record.source_bytes, {},
         source->logical_layer, source->storage_semantics,
         source->artifact_root, source->layout_root,
         source->disposition_root, source->target_logical_root,
         source->disposition_record_root, source->layout_record_root,
         source->runtime_record_root});
    auto next = checked_add_u64(cursor, record.source_bytes);
    if (!next.ok()) return next.status();
    cursor = *next;
  }

  if (has_materialized_expert) {
    for (const auto& bundle : experts.bundles()) {
      auto aligned = checked_align_up_u64(cursor, kFinalAlignment);
      if (!aligned.ok()) return aligned.status();
      const auto bundle_base = *aligned;
      std::uint64_t bundle_offset = 0;
      for (const auto& segment : bundle.segments) {
        const auto source = by_source.find(
            SourceKey{segment.shard_name, segment.file_offset, segment.bytes});
        if (source == by_source.end()) {
          return Status::InvalidArgument("DeepSeek expert segment has no source tensor");
        }
        const auto destination = checked_add_u64(bundle_base, bundle_offset);
        if (!destination.ok()) return destination.status();
        result.copies_.push_back(
            {source->second->tensor_name, source->second->shard_name,
             source->second->dtype, source->second->shape,
             source->second->file_begin, *destination, segment.bytes,
             bundle.identity, source->second->logical_layer,
             source->second->storage_semantics,
             source->second->artifact_root, source->second->layout_root,
             source->second->disposition_root,
             source->second->target_logical_root,
             source->second->disposition_record_root,
             source->second->layout_record_root,
             source->second->runtime_record_root});
        auto next_offset = checked_add_u64(bundle_offset, segment.bytes);
        if (!next_offset.ok()) return next_offset.status();
        bundle_offset = *next_offset;
      }
      if (bundle_offset != DeepSeekExpertBundleLayout::kBundleBytes) {
        return Status::Internal("DeepSeek materialized expert bundle is non-canonical");
      }
      auto next = checked_add_u64(bundle_base, bundle_offset);
      if (!next.ok()) return next.status();
      cursor = *next;
    }
  }

  std::uint64_t payload = 0;
  for (const auto& copy : result.copies_) {
    auto next = checked_add_u64(payload, copy.bytes);
    if (!next.ok()) return next.status();
    payload = *next;
  }
  if (payload != disposition.materialize_fixed_bytes()) {
    return Status::Internal("DeepSeek materialization payload ledger is inconsistent");
  }
  auto backing = checked_align_up_u64(cursor, kFinalAlignment);
  if (!backing.ok()) return backing.status();
  result.payload_bytes_ = payload;
  result.backing_bytes_ = *backing;
  return result;
}

const DeepSeekWeightCopyRecord* DeepSeekWeightMaterializationPlan::find(
    std::string_view tensor_name) const noexcept {
  const auto found = std::find_if(copies_.begin(), copies_.end(),
                                  [&](const auto& copy) {
                                    return copy.tensor_name == tensor_name;
                                  });
  return found == copies_.end() ? nullptr : &*found;
}

}  // namespace pih
