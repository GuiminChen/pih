#include "generation_metadata.h"

namespace pih::offline_deepseek {
namespace {
Result<DeepSeekStorageSemantics> StorageSemantics(DType dtype) {
  switch (dtype) {
    case DType::kFloat32: return DeepSeekStorageSemantics::kDirectF32LittleEndianBits;
    case DType::kFloat16: return DeepSeekStorageSemantics::kDirectF16LittleEndianBits;
    case DType::kBFloat16: return DeepSeekStorageSemantics::kDirectBf16LittleEndianBits;
    // The complete geometry validator restricts I8 to packed routed weights;
    // this must not be used as a generic signed-int8 conversion rule.
    case DType::kInt8: return DeepSeekStorageSemantics::kDirectMxfp4E2m1PackedBits;
    case DType::kUInt8: return DeepSeekStorageSemantics::kDirectU8Bits;
    case DType::kFloat8E4M3: return DeepSeekStorageSemantics::kDirectFp8E4m3Bits;
    case DType::kFloat8E8M0: return DeepSeekStorageSemantics::kDirectUe8m0ScaleBits;
    case DType::kInt32: return DeepSeekStorageSemantics::kDirectI32LittleEndianBits;
    case DType::kInt64: return DeepSeekStorageSemantics::kDirectI64LittleEndianBits;
    case DType::kBool: return DeepSeekStorageSemantics::kDirectBoolBits;
    default: return Status::InvalidArgument("unsupported generation checkpoint storage dtype");
  }
}
}  // namespace

Result<GenerationMetadata> BuildGenerationMetadata(
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> layout_authorities,
    std::span<const TensorDispositionAuthority> disposition_authorities,
    const GenerationLayoutAuthority& authority) {
  if (tensors.size() != 67'612 || layout_authorities.size() != tensors.size() ||
      disposition_authorities.size() != tensors.size())
    return Status::InvalidArgument("generation metadata requires complete joined PP1 inventory");
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    if (disposition_authorities[i].name != tensors[i].name ||
        layout_authorities[i].name != tensors[i].name ||
        disposition_authorities[i].disposition_record_root == Sha256Digest{})
      return Status::InvalidArgument("generation disposition name join or root invalid");
  }
  auto bound = BindGenerationLayout(tensors, layout_authorities, authority);
  if (!bound.ok()) return bound.status();
  auto pipeline = DeepSeekPipelinePlan::Create(1, false);
  if (!pipeline.ok()) return pipeline.status();
  std::vector<DeepSeekRuntimeRecord> records(tensors.size());
  std::vector<SafetensorsShardBinding> bindings;
  bindings.reserve(tensors.size());
  for (const auto& shard : bound->layout.shards) {
    for (std::size_t local = 0; local < shard.tensor_indices.size(); ++local) {
      const auto ordinal = shard.tensor_indices[local];
      const auto& tensor = tensors[ordinal];
      auto semantics = StorageSemantics(tensor.dtype);
      if (!semantics.ok()) return semantics.status();
      auto& record = records[ordinal];
      record.tensor_name = tensor.name;
      record.shard_name = shard.layout.member_name;
      record.name_space = shard.name_space;
      record.dtype = tensor.dtype;
      record.shape = tensor.shape;
      record.file_begin = shard.layout.file_ranges[local].first;
      record.file_end = shard.layout.file_ranges[local].second;
      record.tensor_bytes = tensor.source.bytes;
      record.storage_semantics = *semantics;
      record.target_logical_root = layout_authorities[ordinal].target_logical_root;
      record.disposition_record_root = disposition_authorities[ordinal].disposition_record_root;
      record.layout_record_root = bound->layout_record_roots[ordinal];
      bindings.push_back({tensor.name, shard.layout.member_name});
    }
  }
  // Classification is confined to this offline compiler. Production continues
  // to bind ownership from the admitted runtime-record manifest.
  auto ownership = DeepSeekTensorOwnershipPlan::Create(*pipeline, bindings);
  if (!ownership.ok()) return ownership.status();
  if (ownership->records().size() != records.size() ||
      ownership->owned_count(0) != records.size() || ownership->excluded_dspark_count() != 0)
    return Status::Internal("generation ownership does not close PP1 inventory");
  for (std::size_t i = 0; i < records.size(); ++i) {
    const auto& owner = ownership->records()[i];
    auto& record = records[i];
    if (owner.tensor_name != record.tensor_name || owner.shard_name != record.shard_name)
      return Status::Internal("generation ownership join differs from layout");
    record.role = owner.role;
    record.logical_layer = owner.logical_layer;
    record.owner_rank = owner.owner_rank;
  }
  auto encoded = DeepSeekRuntimeRecordsManifest::Encode(
      records, bound->layout_root, authority.disposition_root, *pipeline);
  if (!encoded.ok()) return encoded.status();
  if (encoded->authority.record_count != tensors.size() ||
      encoded->authority.record_bytes != bound->layout.tensor_bytes ||
      encoded->authority.world_size != 1 || encoded->authority.dspark_enabled)
    return Status::Internal("encoded generation record ledger differs from byte layout");
  return GenerationMetadata{std::move(*bound), std::move(*encoded)};
}
}  // namespace pih::offline_deepseek
