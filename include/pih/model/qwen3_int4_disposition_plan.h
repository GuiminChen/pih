#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

enum class QwenInt4DispositionKind : std::uint8_t {
  kQuantizeW4A16G128 = 1,
  kIdentityCopyBf16 = 2,
  kEqualSourceDedupToAlias = 3,
};

struct QwenInt4SourceDisposition final {
  std::string source_name;
  std::vector<std::uint64_t> source_shape;
  std::uint64_t source_bytes;
  QwenInt4DispositionKind kind;
  std::vector<std::string> target_identities;
  std::string equality_owner_source;
};

class QwenInt4DispositionPlan final {
 public:
  static constexpr std::size_t kOfficialSourceCount = 311;
  static constexpr std::size_t kOfficialTargetCount = 507;

  static Result<QwenInt4DispositionPlan> CreateOfficialPureW4();

  [[nodiscard]] const std::vector<QwenInt4SourceDisposition>& records()
      const noexcept {
    return records_;
  }
  [[nodiscard]] const QwenInt4SourceDisposition* find(
      std::string_view source_name) const noexcept;
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;

 private:
  explicit QwenInt4DispositionPlan(
      std::vector<QwenInt4SourceDisposition> records)
      : records_(std::move(records)) {}

  std::vector<QwenInt4SourceDisposition> records_;
};

}  // namespace pih
