#include "qwen3_int4_artifact_writer.h"

namespace pih {

Result<std::vector<CanonicalExtentWritePlan>> qwen_int4_write_plan(
    const QwenInt4ArtifactLayout& layout) {
  std::vector<CanonicalExtentWritePlan> plan;
  plan.reserve(layout.payload_record_count());
  for (const auto& record : layout.records()) {
    if (record.kind == QwenInt4ArtifactRecordKind::kAlias) continue;
    plan.push_back({record.identity, record.file_offset, record.logical_bytes,
                    record.extent_bytes});
  }
  if (plan.size() != layout.payload_record_count()) {
    return Status::Internal("Qwen INT4 write plan payload count drifted");
  }
  return plan;
}

Result<CanonicalExtentWriteReceipt> write_qwen_int4_artifact(
    const QwenInt4ArtifactMetadata& metadata,
    CanonicalPayloadReader& reader, CanonicalSequentialSink& sink,
    std::size_t maximum_workspace_bytes) {
  auto layout = QwenInt4ArtifactLayout::CreateOfficialPureW4();
  if (!layout.ok()) return layout.status();
  auto region = metadata.serialize_fixed_region();
  if (!region.ok()) return region.status();
  auto reparsed = QwenInt4ArtifactMetadata::ParseAndVerify(*region, *layout);
  if (!reparsed.ok()) return reparsed.status();
  auto plan = qwen_int4_write_plan(*layout);
  if (!plan.ok()) return plan.status();
  return write_canonical_extents(*region, QwenInt4ArtifactLayout::kMetadataBytes,
                                 *plan, reader, sink,
                                 maximum_workspace_bytes);
}

}  // namespace pih
