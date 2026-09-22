#include "pih/model/qwen3_int4_artifact_io.h"

namespace pih {

Result<std::vector<CanonicalExtentVerificationPlan>>
qwen_int4_verification_plan(const QwenInt4ArtifactMetadata& metadata,
                            const QwenInt4ArtifactLayout& layout) {
  if (metadata.records().size() != layout.records().size()) {
    return Status::FailedPrecondition("Qwen INT4 metadata/layout count mismatch");
  }
  std::vector<CanonicalExtentVerificationPlan> plan;
  plan.reserve(layout.payload_record_count());
  for (std::size_t index = 0; index < layout.records().size(); ++index) {
    const auto& expected = layout.records()[index];
    const auto& observed = metadata.records()[index];
    if (observed.layout.identity != expected.identity ||
        observed.layout.kind != expected.kind ||
        observed.layout.file_offset != expected.file_offset ||
        observed.layout.logical_bytes != expected.logical_bytes ||
        observed.layout.extent_bytes != expected.extent_bytes) {
      return Status::FailedPrecondition("Qwen INT4 metadata/layout record mismatch");
    }
    if (expected.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    plan.push_back({expected.identity, expected.file_offset,
                    expected.logical_bytes, expected.extent_bytes,
                    observed.payload_sha256});
  }
  if (plan.size() != layout.payload_record_count()) {
    return Status::Internal("Qwen INT4 verification plan payload count drifted");
  }
  return plan;
}

Result<QwenInt4ArtifactVerificationReceipt> verify_qwen_int4_artifact(
    std::span<const std::byte> file) {
  if (file.size() != QwenInt4ArtifactLayout::kOfficialFileBytes) {
    return Status::FailedPrecondition("Qwen INT4 artifact file length mismatch");
  }
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  if (!layout.ok()) return layout.status();
  auto metadata = QwenInt4ArtifactMetadata::ParseAndVerify(
      file.first(QwenInt4ArtifactLayout::kMetadataBytes), *layout);
  if (!metadata.ok()) return metadata.status();
  auto plan = qwen_int4_verification_plan(*metadata, *layout);
  if (!plan.ok()) return plan.status();
  auto receipt = verify_canonical_extent_file(
      file, QwenInt4ArtifactLayout::kMetadataBytes, *plan);
  if (!receipt.ok()) return receipt.status();
  return QwenInt4ArtifactVerificationReceipt{*receipt, metadata->roots()};
}

}  // namespace pih
