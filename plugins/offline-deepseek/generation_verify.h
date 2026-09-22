#pragma once
#include <filesystem>
#include <functional>
#include <string_view>
#include <string>
#include "pih/core/sha256.h"
namespace pih::offline_deepseek {
class SourceArtifact;
struct SourcePreparationAuthority;
struct GenerationObservation final {
  Sha256Digest artifact_root;
  std::uint32_t tensor_count = 0;
  std::uint64_t tensor_bytes = 0;
  std::uint32_t shard_count = 0;
  std::uint64_t directory_device = 0;
  std::uint64_t directory_inode = 0;
  // True only after a fresh source-bound scan and per-target-tensor comparison.
  bool source_payload_equivalence = false;
  // Canonical existing-format source-bound verification projection. Empty/zero
  // for target-only observations. Hash covers JSON bytes without a newline.
  std::string verification_projection_json;
  Sha256Digest verification_projection_sha256;
  bool receipt_binding_verified = false;
};
using VerificationProgress = std::function<void(std::string_view, std::uint64_t)>;
// Read-only full PP1 observation, not an immutable lease or source-provenance
// receipt. Retains member descriptors through final checks. No console I/O
// except through the optional synchronous callback; no writes/publication.
Result<GenerationObservation> VerifyGeneration(
    const std::filesystem::path& directory, const Sha256Digest& expected_root,
    const VerificationProgress& progress = {});
// Parses receipt bytes against external expected roots, then joins its exact
// 47-object ledger to all actual generation files. No source re-scan or immutable
// admission is implied. Retains target descriptors through final checks.
Result<GenerationObservation> VerifyReceiptBoundGeneration(
    const std::filesystem::path& directory, std::string_view receipt_json,
    const Sha256Digest& artifact_root, const Sha256Digest& receipt_root,
    const VerificationProgress& progress = {});
// Independently derives source semantics, inventory, payloads, disposition and
// target metadata from retained admitted source files. Also compares every
// target tensor's bytes to its observed source payload digest. Read-only; not
// a publication receipt, immutable lease or numerical qualification.
Result<GenerationObservation> VerifySourceBoundGeneration(
    const SourceArtifact& source, const SourcePreparationAuthority& authority,
    const std::filesystem::path& directory, const Sha256Digest& expected_root,
    const VerificationProgress& progress = {});
}  // namespace pih::offline_deepseek
