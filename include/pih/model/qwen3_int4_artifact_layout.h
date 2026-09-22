#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class QwenInt4ArtifactRecordKind : std::uint8_t {
  kPackedValues = 1,
  kFp16Scales = 2,
  kBf16Embedding = 3,
  kBf16Norm = 4,
  kAlias = 5,
};

struct QwenInt4ArtifactRecordPlan final {
  std::string identity;
  QwenInt4ArtifactRecordKind kind;
  std::string source_name;
  std::string alias_owner;
  std::uint64_t logical_bytes;
  std::uint64_t file_offset;
  std::uint64_t extent_bytes;
};

class QwenInt4ArtifactLayout final {
 public:
  static constexpr std::uint64_t kMetadataBytes = 1'048'576;
  static constexpr std::uint64_t kExtentAlignment = 4'096;
  static constexpr std::size_t kOfficialRecordCount = 507;
  static constexpr std::size_t kOfficialPayloadRecordCount = 506;
  static constexpr std::uint64_t kOfficialLogicalPayloadBytes = 538'378'240;
  static constexpr std::uint64_t kOfficialPaddedDataBytes = 538'710'016;
  static constexpr std::uint64_t kOfficialFileBytes = 539'758'592;

  static Result<QwenInt4ArtifactLayout> CreateOfficialPureW4();

  [[nodiscard]] const std::vector<QwenInt4ArtifactRecordPlan>& records()
      const noexcept {
    return records_;
  }
  [[nodiscard]] std::size_t payload_record_count() const noexcept {
    return payload_record_count_;
  }
  [[nodiscard]] std::uint64_t logical_payload_bytes() const noexcept {
    return logical_payload_bytes_;
  }
  [[nodiscard]] std::uint64_t padded_data_bytes() const noexcept {
    return padded_data_bytes_;
  }
  [[nodiscard]] std::uint64_t file_bytes() const noexcept {
    return file_bytes_;
  }
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;

 private:
  QwenInt4ArtifactLayout(std::vector<QwenInt4ArtifactRecordPlan> records,
                         std::size_t payload_record_count,
                         std::uint64_t logical_payload_bytes,
                         std::uint64_t padded_data_bytes,
                         std::uint64_t file_bytes)
      : records_(std::move(records)),
        payload_record_count_(payload_record_count),
        logical_payload_bytes_(logical_payload_bytes),
        padded_data_bytes_(padded_data_bytes),
        file_bytes_(file_bytes) {}

  std::vector<QwenInt4ArtifactRecordPlan> records_;
  std::size_t payload_record_count_;
  std::uint64_t logical_payload_bytes_;
  std::uint64_t padded_data_bytes_;
  std::uint64_t file_bytes_;
};

}  // namespace pih
