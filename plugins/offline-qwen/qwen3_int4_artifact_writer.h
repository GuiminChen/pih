#pragma once

#include "pih/io/canonical_extent_writer.h"
#include "pih/model/qwen3_int4_artifact_metadata.h"

namespace pih {
Result<std::vector<CanonicalExtentWritePlan>> qwen_int4_write_plan(
    const QwenInt4ArtifactLayout& layout);
Result<CanonicalExtentWriteReceipt> write_qwen_int4_artifact(
    const QwenInt4ArtifactMetadata& metadata,
    CanonicalPayloadReader& reader, CanonicalSequentialSink& sink,
    std::size_t maximum_workspace_bytes);
}  // namespace pih
