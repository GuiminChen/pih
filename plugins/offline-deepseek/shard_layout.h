#pragma once
#include <string>
#include <vector>
#include "identity_copy.h"
#include "pih/core/dtype.h"

namespace pih::offline_deepseek {
struct IdentityTensor final {
  std::string name;
  DType dtype;
  std::vector<std::uint64_t> shape;
  CopyRange source;
};
struct ShardLayout final {
  std::string member_name;
  std::vector<std::byte> header_prefix;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> file_ranges;
  std::uint64_t file_bytes;
  Sha256Digest header_sha256;
};
// Inputs are already admitted and strictly name-sorted, with no tensors from
// another namespace. This is byte layout, not full source/geometry admission.
Result<ShardLayout> BuildShardLayout(std::string_view name_space,
                                    std::span<const IdentityTensor> tensors);
Result<CopyReceipt> CopyPlannedIdentityShard(int staging_directory_fd,
    std::string_view name_space, std::span<const IdentityTensor> tensors);

struct TensorLayoutAuthority final {
  std::string name;
  Sha256Digest source_payload_record_root;
  Sha256Digest target_logical_root;
};
struct BoundShardLayout final {
  ShardLayout layout;
  std::vector<Sha256Digest> layout_record_roots;
  Sha256Digest record_layout_set_root;
  Sha256Digest weight_map_set_root;
  Sha256Digest shard_layout_root;
};
// Derives typed layout hashes from freshly rebuilt byte geometry and admitted
// per-tensor roots. Does not authenticate supplied roots or read source payloads.
Result<BoundShardLayout> BindShardLayout(std::string_view name_space,
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> authorities);
}  // namespace pih::offline_deepseek
