#pragma once
#include "generation_promote.h"
#include "prepare_generation.h"

namespace pih::offline_deepseek {
struct GenerationCommit final {
  Status status = Status::Ok();
  GenerationPromotion promotion;
  // Conservative on failure: creation may have happened even when readback or
  // fsync failed. Inspect instead of overwriting/retrying automatically.
  bool receipt_may_exist = false;
  bool receipt_committed = false;
  Sha256Digest receipt_root;
  Sha256Digest verification_projection_sha256;
};
// Commits a verified generation and read-only receipt, without activation.
// staging_name is a direct child of store/.staging. Requires exclusive caller
// control of source/store mutation. Fresh source-bound verification occurs both
// before promotion and after sealing, followed by exclusive receipt creation.
// Never overwrites, removes, rolls back or changes pointers/current.json.
GenerationCommit CommitGeneration(const SourceArtifact& source,
    const SourcePreparationAuthority& authority, const std::filesystem::path& store_path,
    std::string_view staging_name, const Sha256Digest& artifact_root,
    const VerificationProgress& progress = {});
}  // namespace pih::offline_deepseek
