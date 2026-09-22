#pragma once
#include "weight_source_file.h"

namespace pih::deepseek_v41 {
struct SourceNameBinding final {
  std::string canonical_name;
  std::string source_name;
  // Index into Create's original file order. Caller retains the same admitted
  // file objects/order; this metadata inventory does not own their descriptors.
  std::uint32_t shard = 0;
};
// Checks exact membership against an independently authenticated HF index and
// rejects normalization collisions. Keeps every source name, including vision,
// MTP and auxiliary tensors: downstream conversion must explicitly classify them.
// This is index closure, not a claim of reference-model geometry completeness.
class WeightSourceInventory final {
 public:
  static Result<WeightSourceInventory> Create(std::string_view index_json,
      const Sha256Digest& expected_index_digest,
      std::span<const WeightSourceFile* const> files);
  std::span<const SourceNameBinding> bindings() const noexcept { return bindings_; }
  const SourceNameBinding* Find(std::string_view canonical_name) const noexcept;
 private:
  std::vector<SourceNameBinding> bindings_;
};
}  // namespace pih::deepseek_v41
