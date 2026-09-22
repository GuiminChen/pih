#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/deepseek_runtime_records_manifest.h"

namespace pih {

struct DeepSeekRuntimeArtifactShard final {
  std::string name_space;
  std::string shard_name;
  std::uint64_t file_bytes = 0;
  std::uint32_t tensor_count = 0;
  std::uint64_t payload_bytes = 0;
  Sha256Digest object_sha256;
  Sha256Digest shard_layout_root;
  Sha256Digest converted_shard_root;
};

struct DeepSeekRuntimeArtifactIndex final {
  std::string name;
  std::uint64_t file_bytes = 0;
  Sha256Digest object_sha256;
  Sha256Digest index_root;
  Sha256Digest index_object_root;
};

class DeepSeekRuntimeArtifactManifest final {
 public:
  static constexpr std::size_t kMaximumBytes = 16ULL * 1024 * 1024;
  static constexpr std::size_t kMaximumTargetMemberCount = 50;

  static Result<DeepSeekRuntimeArtifactManifest> Parse(
      std::string_view json, const Sha256Digest& expected_artifact_root);
  Status validate_flash_0731_geometry() const;

  struct EncodeInput final {
    Sha256Digest logical_layout_root;
    Sha256Digest source_inventory_root;
    Sha256Digest source_payload_closure_root;
    Sha256Digest converter_identity_root;
    std::vector<DeepSeekRuntimeArtifactShard> shards;
    DeepSeekRuntimeArtifactIndex index;
    DeepSeekRuntimeRecordsAuthority records;
    std::uint32_t maximum_copy_chunk_bytes = 0;
    std::uint32_t source_descriptor_count = 0;
    std::uint32_t maximum_output_descriptors = 0;
  };
  struct Encoded final {
    std::string json;
    Sha256Digest artifact_root;
    Sha256Digest object_sha256;
  };
  // Offline encoder for a complete frozen family geometry. Recomputes derived
  // object/conversion/artifact roots, then reparses its output. Input hashes and
  // resource observations require independent admission; no files are inspected
  // or published here. Resource contract: 1 MiB copy chunks, 50 source FDs, one
  // output FD. This API does not certify those observations actually occurred.
  static Result<Encoded> Encode(const EncodeInput& input);

  [[nodiscard]] const Sha256Digest& artifact_root() const noexcept {
    return artifact_root_;
  }
  [[nodiscard]] const Sha256Digest& conversion_root() const noexcept {
    return conversion_root_;
  }
  [[nodiscard]] const Sha256Digest& layout_root() const noexcept {
    return layout_root_;
  }
  [[nodiscard]] const Sha256Digest& disposition_root() const noexcept {
    return disposition_root_;
  }
  [[nodiscard]] bool dspark_enabled() const noexcept {
    return dspark_enabled_;
  }
  [[nodiscard]] std::uint32_t world_size() const noexcept {
    return world_size_;
  }
  [[nodiscard]] std::uint32_t tensor_count() const noexcept {
    return tensor_count_;
  }
  [[nodiscard]] std::uint64_t tensor_bytes() const noexcept {
    return tensor_bytes_;
  }
  [[nodiscard]] const std::vector<DeepSeekRuntimeArtifactShard>& shards()
      const noexcept {
    return shards_;
  }
  [[nodiscard]] const DeepSeekRuntimeArtifactIndex& index() const noexcept {
    return index_;
  }
  [[nodiscard]] const DeepSeekRuntimeRecordsAuthority& runtime_records()
      const noexcept {
    return runtime_records_;
  }

 private:
  Sha256Digest artifact_root_;
  Sha256Digest conversion_root_;
  Sha256Digest layout_root_;
  Sha256Digest disposition_root_;
  bool dspark_enabled_ = false;
  std::uint32_t world_size_ = 0;
  std::uint32_t tensor_count_ = 0;
  std::uint64_t tensor_bytes_ = 0;
  std::vector<DeepSeekRuntimeArtifactShard> shards_;
  DeepSeekRuntimeArtifactIndex index_;
  DeepSeekRuntimeRecordsAuthority runtime_records_;
};

}  // namespace pih
