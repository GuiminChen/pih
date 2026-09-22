#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/qwen3_int4_artifact_layout.h"

namespace pih {

struct QwenInt4PayloadDigest final {
  std::string identity;
  Sha256Digest sha256;
};

struct QwenInt4ArtifactRoots final {
  Sha256Digest source_artifact;
  Sha256Digest source_binding;
  Sha256Digest disposition;
};

struct QwenInt4ArtifactMetadataRecord final {
  QwenInt4ArtifactRecordPlan layout;
  Sha256Digest payload_sha256;
};

class QwenInt4ArtifactMetadata final {
 public:
  static constexpr std::string_view kFormatId = "xing-w4a16-sym-g128-v1";
  static constexpr std::string_view kLayoutId =
      "canonical-nk-low-nibble-k-v1";

  static Result<QwenInt4ArtifactMetadata> Create(
      const QwenInt4ArtifactLayout& layout, QwenInt4ArtifactRoots roots,
      std::vector<QwenInt4PayloadDigest> payload_digests);
  static Result<QwenInt4ArtifactMetadata> ParseAndVerify(
      std::span<const std::byte> fixed_region,
      const QwenInt4ArtifactLayout& expected_layout);

  [[nodiscard]] Result<std::vector<std::byte>> serialize_fixed_region() const;
  [[nodiscard]] const QwenInt4ArtifactRoots& roots() const noexcept { return roots_; }
  [[nodiscard]] const std::vector<QwenInt4ArtifactMetadataRecord>& records()
      const noexcept { return records_; }
  [[nodiscard]] std::string_view format_id() const noexcept { return kFormatId; }
  [[nodiscard]] std::string_view layout_id() const noexcept { return kLayoutId; }
  [[nodiscard]] std::size_t encoded_bytes() const noexcept { return encoded_bytes_; }

 private:
  QwenInt4ArtifactMetadata(QwenInt4ArtifactRoots roots,
                           std::vector<QwenInt4ArtifactMetadataRecord> records,
                           std::size_t encoded_bytes)
      : roots_(roots), records_(std::move(records)), encoded_bytes_(encoded_bytes) {}

  QwenInt4ArtifactRoots roots_;
  std::vector<QwenInt4ArtifactMetadataRecord> records_;
  std::size_t encoded_bytes_ = 0;
};

}  // namespace pih
