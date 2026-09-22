#pragma once
#include "generation_copy.h"
#include "source_payload.h"
#include "source_artifact.h"
namespace pih::offline_deepseek {
// Read-only, cheap preflight before source hashing. Repeated immediately before
// writes; this is not a lock or a guarantee against concurrent directory edits.
Status ValidatePreparationStaging(int staging_directory_fd);
struct SourcePreparationAuthority final {
  Sha256Digest model_digest;
  Sha256Digest semantic_root;
  Sha256Digest inventory_root;
  Sha256Digest expected_payload_root;
  Sha256Digest converter_identity_root;
};
// Native PP1 conversion from an already-admitted complete source plan. The
// 50 descriptors include config, index and 48 shards; all stay borrowed/open.
// Requires an empty owned staging directory. Scans source payloads, compares
// the trusted payload root, derives disposition/layout, copies shards and writes
// metadata. Failure retains files. No provenance admission, publication or
// activation is inferred; final all-member verification is a separate step.
Result<CopiedGeneration> PreparePp1Generation(int staging_directory_fd,
    std::span<const int> admitted_source_descriptors,
    std::span<const SourcePayloadShardInput> shards,
    const SourcePreparationAuthority& authority);
// Owner-bound entry derives the entire source plan and compares semantic and
// inventory roots, then revalidates retained paths before/after conversion.
Result<CopiedGeneration> PreparePp1Generation(int staging_directory_fd,
    const SourceArtifact& source,
    const SourcePreparationAuthority& authority);
}  // namespace pih::offline_deepseek
