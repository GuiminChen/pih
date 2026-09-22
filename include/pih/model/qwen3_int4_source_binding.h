#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/core/dtype.h"
#include "pih/core/result.h"
#include "pih/core/sha256.h"
#include "pih/model/qwen3_int4_disposition_plan.h"
#include "pih/model/qwen3_source_artifact.h"

namespace pih {

class SafetensorsFile;

struct QwenInt4ObservedSourceRecord final {
  std::string name;
  DType dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t file_begin;
  std::uint64_t file_end;
  Sha256Digest payload_sha256;
};

class QwenInt4SourceBinding final {
 public:
  static Result<QwenInt4SourceBinding> Verify(
      const QwenInt4DispositionPlan& plan,
      std::vector<QwenInt4ObservedSourceRecord> observed,
      const Qwen3SourceArtifactReceipt& receipt);
  static Result<QwenInt4SourceBinding> ObserveAndVerifyOfficial(
      const SafetensorsFile& source,
      const Qwen3SourceArtifactReceipt& receipt);

  [[nodiscard]] const std::vector<QwenInt4ObservedSourceRecord>& records()
      const noexcept { return records_; }
  [[nodiscard]] const QwenInt4ObservedSourceRecord* find(
      std::string_view name) const noexcept;
  [[nodiscard]] Result<Sha256Digest> semantic_digest() const;

 private:
  explicit QwenInt4SourceBinding(
      std::vector<QwenInt4ObservedSourceRecord> records)
      : records_(std::move(records)) {}
  std::vector<QwenInt4ObservedSourceRecord> records_;
};

}  // namespace pih
