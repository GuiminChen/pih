#pragma once
#include "generation_metadata.h"
#include "pih/model/deepseek_runtime_artifact_manifest.h"

namespace pih::offline_deepseek {
struct CopiedGeneration final {
  GenerationMetadata metadata;
  DeepSeekRuntimeArtifactManifest::Encoded manifest;
};
// Copies all 44 PP1 shards sequentially into a caller-owned staging directory.
// Metadata is built before any write. Only real successful copy receipts feed
// the manifest. Caller must admit source roots/descriptors and keep them open.
// Returns metadata in memory, not a published generation; metadata file writes,
// final all-member verification and atomic publication remain caller work.
// Failure retains created shards; never removes or replaces existing members.
Result<CopiedGeneration> CopyGenerationShards(int staging_directory_fd,
    std::span<const int> admitted_source_descriptors,
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> layout_authorities,
    std::span<const TensorDispositionAuthority> disposition_authorities,
    const GenerationLayoutAuthority& authority,
    const Sha256Digest& converter_identity_root);
// Validates the in-memory manifest/sidecar before writing index, runtime records
// and finally manifest. Still staging only: shard re-verification and publication
// are separate; failures retain any files already created.
Status WriteGenerationMetadata(int staging_directory_fd, const CopiedGeneration& generation);
}  // namespace pih::offline_deepseek
