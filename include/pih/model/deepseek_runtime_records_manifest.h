#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/dtype.h"
#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_tensor_ownership_plan.h"

namespace pih {

enum class DeepSeekStorageSemantics : std::uint8_t {
  kDirectF32LittleEndianBits = 1,
  kDirectF16LittleEndianBits = 2,
  kDirectBf16LittleEndianBits = 3,
  kDirectMxfp4E2m1PackedBits = 4,
  kDirectU8Bits = 5,
  kDirectFp8E4m3Bits = 6,
  kDirectUe8m0ScaleBits = 7,
  kDirectI32LittleEndianBits = 8,
  kDirectI64LittleEndianBits = 9,
  kDirectBoolBits = 10,
};

struct DeepSeekRuntimeRecord final {
  std::string tensor_name;
  std::string shard_name;
  std::string name_space;
  DeepSeekTensorRole role{};
  std::uint32_t logical_layer = UINT32_MAX;
  std::uint32_t owner_rank = UINT32_MAX;
  DType dtype{};
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin = 0;
  std::uint64_t file_end = 0;
  std::uint64_t tensor_bytes = 0;
  DeepSeekStorageSemantics storage_semantics{};
  Sha256Digest target_logical_root;
  Sha256Digest disposition_record_root;
  Sha256Digest layout_record_root;
  Sha256Digest runtime_record_root;
};

struct DeepSeekRuntimeRecordsAuthority final {
  Sha256Digest layout_root;
  Sha256Digest disposition_root;
  Sha256Digest record_set_root;
  Sha256Digest runtime_records_root;
  Sha256Digest body_sha256;
  Sha256Digest object_sha256;
  std::uint64_t object_bytes = 0;
  std::uint64_t record_bytes = 0;
  std::uint32_t record_count = 0;
  std::uint32_t world_size = 0;
  bool dspark_enabled = false;
};

class DeepSeekRuntimeRecordsManifest final {
 public:
  static constexpr std::size_t kMaximumBytes = 128ULL * 1024 * 1024;
  static constexpr std::size_t kMaximumRecordCount = 72'317;
  static constexpr std::size_t kMaximumTensorNameBytes = 512;

  static Result<DeepSeekRuntimeRecordsManifest> Parse(
      std::string_view json,
      const DeepSeekRuntimeRecordsAuthority& authority,
      const DeepSeekPipelinePlan& pipeline);

  struct Encoded final {
    std::string json;
    DeepSeekRuntimeRecordsAuthority authority;
  };
  // Offline serialization only. Caller supplies admitted layout/disposition and
  // per-tensor roots; this does not establish source provenance or publish data.
  // Runtime record roots are recomputed, never trusted from input records.
  static Result<Encoded> Encode(std::span<const DeepSeekRuntimeRecord> records,
      Sha256Digest layout_root, Sha256Digest disposition_root,
      const DeepSeekPipelinePlan& pipeline);

  [[nodiscard]] const std::vector<DeepSeekRuntimeRecord>& records()
      const noexcept {
    return records_;
  }
  [[nodiscard]] const DeepSeekRuntimeRecord* find(
      std::string_view tensor_name) const noexcept;
  [[nodiscard]] const DeepSeekRuntimeRecordsAuthority& authority()
      const noexcept {
    return authority_;
  }

 private:
  DeepSeekRuntimeRecordsAuthority authority_;
  std::vector<DeepSeekRuntimeRecord> records_;
};

}  // namespace pih
