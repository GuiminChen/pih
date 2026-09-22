#pragma once
// Private to native offline Qwen conversion; not a model/runtime SDK interface.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "pih/io/canonical_extent_writer.h"
#include "pih/model/qwen3_int4_artifact_layout.h"
#include "pih/model/qwen3_int4_artifact_metadata.h"
#include "pih/model/qwen3_int4_linear_shape_ledger.h"
#include "qwen3_int4_tensor_payload.h"

namespace pih {

class QwenInt4ConversionTensorSource {
 public:
  virtual ~QwenInt4ConversionTensorSource() = default;
  virtual Result<std::span<const std::byte>> tensor(
      std::string_view source_name) = 0;
};

struct QwenInt4DigestScanReceipt final {
  std::vector<QwenInt4PayloadDigest> payload_digests;
  std::uint64_t peak_anonymous_workspace_bytes;
};

class QwenInt4ConversionStream final : public CanonicalPayloadReader {
 public:
  static Result<QwenInt4ConversionStream> Create(
      std::vector<QwenInt4ArtifactRecordPlan> artifact_records,
      std::vector<QwenInt4LinearRecordPlan> linear_records,
      QwenInt4ConversionTensorSource& source);

  Result<QwenInt4DigestScanReceipt> scan_payload_digests();
  Result<std::size_t> read(std::string_view identity,
                           std::uint64_t logical_offset,
                           std::span<std::byte> output) override;
  void reset() noexcept;

 private:
  QwenInt4ConversionStream(
      std::vector<QwenInt4ArtifactRecordPlan> artifact_records,
      std::vector<QwenInt4LinearRecordPlan> linear_records,
      QwenInt4ConversionTensorSource* source)
      : artifact_records_(std::move(artifact_records)),
        linear_records_(std::move(linear_records)), source_(source) {}

  Result<std::span<const std::byte>> materialize(
      const QwenInt4ArtifactRecordPlan& record);
  const QwenInt4ArtifactRecordPlan* artifact(std::string_view identity) const;
  const QwenInt4LinearRecordPlan* linear_for(
      const QwenInt4ArtifactRecordPlan& record) const;

  std::vector<QwenInt4ArtifactRecordPlan> artifact_records_;
  std::vector<QwenInt4LinearRecordPlan> linear_records_;
  QwenInt4ConversionTensorSource* source_;
  std::optional<QwenInt4TensorPayload> active_linear_payload_;
  std::string active_linear_source_;
  std::uint64_t peak_anonymous_workspace_bytes_ = 0;
};

}  // namespace pih
