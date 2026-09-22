#pragma once
#include "generation_metadata.h"
#include "source_payload.h"
namespace pih::offline_deepseek {
struct Pp1DispositionRecords final {
  std::vector<IdentityTensor> selected;
  std::vector<TensorLayoutAuthority> layout_authorities;
  std::vector<TensorDispositionAuthority> selected_dispositions;
  std::vector<TensorDispositionAuthority> excluded_dispositions;
};
// Complete frozen source inventory -> PP1 selection plus explicit MTP exclusion.
// Derives logical/disposition record roots, not aggregate disposition/owner roots.
// Supplied source roots and payload observations must already be admitted.
Result<Pp1DispositionRecords> BuildPp1DispositionRecords(
    std::span<const ObservedSourceTensor> source);
struct Pp1Disposition final {
  Pp1DispositionRecords records;
  GenerationLayoutAuthority layout_authority;
  std::vector<Sha256Digest> namespace_roots;
};
// Rebuilds records and derives all namespace/owner/global roots. Supplied source
// inventory/payload roots still require independent admission.
Result<Pp1Disposition> BuildPp1Disposition(std::span<const ObservedSourceTensor> source,
    const Sha256Digest& inventory_root, const Sha256Digest& payload_root);
}  // namespace pih::offline_deepseek
