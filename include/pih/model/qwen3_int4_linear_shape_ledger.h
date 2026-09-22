#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class QwenInt4LinearShapeFamily : std::uint8_t {
  kQProj = 1,
  kKvProj = 2,
  kOProj = 3,
  kGateUpProj = 4,
  kDownProj = 5,
};

struct QwenInt4LinearRecordPlan final {
  QwenInt4LinearShapeFamily family;
  std::string source_name;
  std::string values_name;
  std::string scales_name;
  std::uint64_t rows;
  std::uint64_t columns;
  std::uint64_t groups_per_row;
  std::uint64_t packed_bytes;
  std::uint64_t scale_bytes;
  std::uint64_t bf16_bytes;
  std::uint64_t override_added_bytes;
};

class QwenInt4LinearShapeLedger final {
 public:
  static constexpr std::size_t kOfficialRecordCount = 196;
  static constexpr std::uint64_t kOfficialPackedBytes = 220'200'960;
  static constexpr std::uint64_t kOfficialScaleBytes = 6'881'280;
  static constexpr std::uint64_t kOfficialBf16Bytes = 880'803'840;

  static Result<QwenInt4LinearShapeLedger> CreateOfficial();

  [[nodiscard]] const std::vector<QwenInt4LinearRecordPlan>& records()
      const noexcept {
    return records_;
  }
  [[nodiscard]] const QwenInt4LinearRecordPlan* find_first(
      QwenInt4LinearShapeFamily family) const noexcept;
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;

 private:
  explicit QwenInt4LinearShapeLedger(
      std::vector<QwenInt4LinearRecordPlan> records)
      : records_(std::move(records)) {}

  std::vector<QwenInt4LinearRecordPlan> records_;
};

}  // namespace pih
