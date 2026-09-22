#pragma once
#include "generation_verify.h"

namespace pih::offline_deepseek {
struct GenerationPromotion final {
  Status status = Status::Ok();
  bool renamed = false;
  bool directories_synced = false;
  // All mode changes and their fsyncs completed. This is not fs-verity or an
  // immutable lease; an owner may restore write permissions afterwards.
  bool read_only_sealed = false;
  std::filesystem::path destination;
};
// Low-level verified content-addressed move, NOT full store publication or
// activation. Destination is generations_directory / ("sha256-" + root.hex()).
// Both parents must already exist on one filesystem and be caller-controlled;
// no symlink components or overwrite fallback. On post-rename error, renamed
// remains true: callers must inspect the destination, not repeat conversion.
// Seals members 0444 and generation directory 0555 before final verification.
// A sealing failure can leave partially changed permissions at destination.
// No immutable/source-provenance admission, receipt writing or cleanup.
GenerationPromotion PromoteVerifiedGeneration(const std::filesystem::path& staging,
    const std::filesystem::path& generations_directory, const Sha256Digest& expected_root,
    const VerificationProgress& progress = {});
}  // namespace pih::offline_deepseek
