#pragma once
#include <optional>
#include <string>
#include "pih/core/sha256.h"

namespace pih::offline_deepseek {
class GenerationStore;
struct GenerationPointerInput final {
  Sha256Digest artifact_root;
  Sha256Digest receipt_root;
  Sha256Digest catalog_root;
  std::uint64_t activation_ordinal = 0;
  std::optional<Sha256Digest> previous_pointer_root;
};
struct GenerationPointer final {
  GenerationPointerInput input;
  Sha256Digest pointer_root;
  std::string json;
};
// Canonical metadata only. Null predecessor is valid exactly at ordinal 1.
// The integer JSON parser bounds ordinals to INT64_MAX. No file mutation,
// receipt validation, catalog admission, signature or immutable lease.
Result<GenerationPointer> EncodeGenerationPointer(const GenerationPointerInput& input);
Result<GenerationPointer> ParseGenerationPointer(std::string_view json);
// Reads through the admitted store descriptor and revalidates file/path/store
// identity. Empty optional means no pointer. Does not admit the pointed-to
// generation, receipt or catalog; caller must still compare expected roots.
Result<std::optional<GenerationPointer>> ReadCurrentGenerationPointer(const GenerationStore& store);
// Revalidates predecessor metadata and enforces exactly the next ordinal.
// An optional predecessor is not proof that it is the store's current pointer.
Result<GenerationPointer> NextGenerationPointer(const Sha256Digest& artifact_root,
    const Sha256Digest& receipt_root, const Sha256Digest& catalog_root,
    std::uint64_t ordinal, const std::optional<GenerationPointer>& previous);
}  // namespace pih::offline_deepseek
