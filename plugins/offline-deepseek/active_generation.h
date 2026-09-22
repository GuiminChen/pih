#pragma once
#include "generation_pointer.h"
#include "generation_verify.h"
namespace pih::offline_deepseek {
struct ActiveGenerationObservation final {
  GenerationPointer pointer;
  GenerationObservation generation;
};
// Resolves an externally pinned pointer/catalog pair and verifies its sealed
// receipt and all target bytes. Holds a shared advisory store lock during the
// operation; returned metadata is not a retained storage lease. Consumers must
// independently admit/pin artifact bytes before execution. No writes or source
// checkpoint required; catalog contents are not admitted here.
Result<ActiveGenerationObservation> ObserveActiveGeneration(const std::filesystem::path& store_path,
    const Sha256Digest& expected_pointer_root, const Sha256Digest& expected_catalog_root,
    const VerificationProgress& progress = {});
}  // namespace pih::offline_deepseek
