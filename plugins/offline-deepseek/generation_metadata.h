#pragma once

#include "generation_layout.h"
#include "pih/model/deepseek_runtime_records_manifest.h"

namespace pih::offline_deepseek {
struct TensorDispositionAuthority final {
  std::string name;
  Sha256Digest disposition_record_root;
};
struct GenerationMetadata final {
  BoundGenerationLayout bound_layout;
  DeepSeekRuntimeRecordsManifest::Encoded runtime_records;
};
// Offline PP1 / DSpark-disabled metadata assembly. All three input arrays must
// have identical strict name order and cover the complete target inventory.
// Source and disposition roots require prior caller admission. No payload I/O,
// provenance admission, artifact-manifest production or publication occurs.
Result<GenerationMetadata> BuildGenerationMetadata(
    std::span<const IdentityTensor> tensors,
    std::span<const TensorLayoutAuthority> layout_authorities,
    std::span<const TensorDispositionAuthority> disposition_authorities,
    const GenerationLayoutAuthority& authority);
}  // namespace pih::offline_deepseek
