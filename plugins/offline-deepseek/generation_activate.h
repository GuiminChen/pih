#pragma once
#include "generation_pointer.h"
#include "prepare_generation.h"
#include "generation_verify.h"

namespace pih::offline_deepseek {
struct GenerationActivation final {
  Status status = Status::Ok();
  bool temporary_may_exist = false;
  bool pointer_replaced = false;
  bool directories_synced = false;
  bool activated = false;
  Sha256Digest pointer_root;
};
// Freshly revalidates source equivalence and the sealed receipt before updating
// current.json. Requires caller-admitted catalog/receipt/previous roots and an
// exact next ordinal. Null expected_previous requires an empty current pointer.
// Holds a nonblocking advisory store lock shared with CommitGeneration. External
// writers must still be excluded. Failed phases retain files; no rollback.
// Does not reload a running Worker, admit catalog bytes or grant immutable leases.
GenerationActivation ActivateGeneration(const SourceArtifact& source,
    const SourcePreparationAuthority& authority, const std::filesystem::path& store_path,
    const Sha256Digest& artifact_root, const Sha256Digest& receipt_root,
    const Sha256Digest& catalog_root, std::uint64_t ordinal,
    const std::optional<Sha256Digest>& expected_previous,
    const VerificationProgress& progress = {});
}  // namespace pih::offline_deepseek
