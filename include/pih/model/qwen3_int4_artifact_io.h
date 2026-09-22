#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "pih/io/canonical_extent_verifier.h"
#include "pih/model/qwen3_int4_artifact_metadata.h"

namespace pih {

struct QwenInt4ArtifactVerificationReceipt final {
  CanonicalExtentVerificationReceipt file;
  QwenInt4ArtifactRoots roots;
};

Result<std::vector<CanonicalExtentVerificationPlan>>
qwen_int4_verification_plan(const QwenInt4ArtifactMetadata& metadata,
                            const QwenInt4ArtifactLayout& layout);
Result<QwenInt4ArtifactVerificationReceipt> verify_qwen_int4_artifact(
    std::span<const std::byte> file);

}  // namespace pih
