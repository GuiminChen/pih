#pragma once
#include "generation_verify.h"
#include <vector>

namespace pih::offline_deepseek {
struct EncodedGenerationReceipt final {
  std::string json;
  Sha256Digest receipt_root;
  Sha256Digest body_sha256;
  Sha256Digest object_sha256;
  // Sum of the 47 generation members, not the encoded receipt's byte length.
  std::uint64_t generation_object_bytes = 0;
};
struct GenerationReceiptObject final {
  std::string name;
  std::uint64_t bytes = 0;
  Sha256Digest sha256;
};
struct GenerationReceipt final {
  Sha256Digest artifact_root;
  Sha256Digest receipt_root;
  Sha256Digest verification_projection_sha256;
  Sha256Digest conversion_root;
  Sha256Digest layout_root;
  Sha256Digest disposition_root;
  Sha256Digest converter_identity_root;
  std::vector<GenerationReceiptObject> objects;
  std::uint64_t object_bytes = 0;
};
// Bounded canonical parsing and typed-root verification against independently
// expected artifact/receipt roots. Does not read or admit generation bytes,
// prove source equivalence, authenticate a signature or grant storage leases.
Result<GenerationReceipt> ParseGenerationReceipt(std::string_view json,
    const Sha256Digest& expected_artifact_root, const Sha256Digest& expected_receipt_root);
// Metadata encoding only, not publication or immutable admission. Callers must
// obtain a fresh VerifySourceBoundGeneration result themselves, not deserialize
// or trust a caller-authored observation. Rechecks canonical projection/hash,
// PP1 geometry and object table; cannot authenticate an observation's origin.
Result<EncodedGenerationReceipt> EncodeGenerationReceipt(const GenerationObservation& observation);
// Exact canonical byte comparison against a freshly rebuilt expected receipt.
// No filesystem reads, generation promotion, activation or signature checking.
Status ValidateGenerationReceipt(std::string_view json, const GenerationObservation& observation);
}  // namespace pih::offline_deepseek
