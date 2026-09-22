#pragma once
#include "shard_layout.h"
namespace pih::offline_deepseek {
struct SourceTensorAuthority final {
  std::string name;
  Sha256Digest source_record_root;
  DType dtype{};
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
};
struct ObservedSourceTensor final {
  IdentityTensor tensor;
  Sha256Digest source_record_root;
  Sha256Digest payload_record_root;
};
// Borrowed descriptor must remain open through conversion. Reads actual header
// and payloads and binds their hashes to caller-admitted source/object roots.
// This is not official provenance or full source-inventory admission.
Result<std::vector<ObservedSourceTensor>> ObserveSourcePayloads(int source_fd,
    std::string_view shard_name, const Sha256Digest& artifact_object_root,
    std::span<const SourceTensorAuthority> authorities);
struct SourcePayloadShardInput final {
  int descriptor = -1;
  std::string name;
  Sha256Digest artifact_object_root;
  Sha256Digest inventory_root;
  std::uint64_t tensor_bytes = 0;
  std::vector<SourceTensorAuthority> tensors;
};
struct SourcePayloadClosure final {
  // Global strict name order, including all three MTP namespaces.
  std::vector<ObservedSourceTensor> tensors;
  std::vector<Sha256Digest> shard_roots;
  Sha256Digest closure_root;
};
// Scans all 48 shards, validates the frozen complete source name/count/byte
// inventory, and derives the existing typed closure. Supplied provenance,
// semantic and inventory roots still require independent admission.
Result<SourcePayloadClosure> ObserveSourcePayloadClosure(
    std::span<const SourcePayloadShardInput> shards,
    const Sha256Digest& artifact_model_digest, const Sha256Digest& semantic_root,
    const Sha256Digest& source_inventory_root);
}  // namespace pih::offline_deepseek
