#pragma once
#include "config.h"
#include "weight_source_inventory.h"
#include "weight_inventory.h"

namespace pih::deepseek_v41 {
// Owns all authenticated source shards and their stable index order. The trusted
// expectation document binds config/index/shard byte identities in one digest.
// Caller admits the source directory descriptor and excludes concurrent writers.
class WeightSourceCheckpoint final {
 public:
  static constexpr std::size_t kMaximumExpectationBytes = 128 * 1024;
  static Result<std::unique_ptr<WeightSourceCheckpoint>> Open(int source_directory,
      std::string_view expectations_json, const Sha256Digest& trusted_digest);
  ~WeightSourceCheckpoint();
  WeightSourceCheckpoint(const WeightSourceCheckpoint&) = delete;
  WeightSourceCheckpoint& operator=(const WeightSourceCheckpoint&) = delete;
  const FlashConfig& config() const noexcept;
  // Owned authenticated snapshot; valid for this checkpoint object's lifetime.
  std::string_view config_json() const noexcept;
  const Sha256Digest& expectation_digest() const noexcept;
  const WeightSourceInventory& inventory() const noexcept;
  Status Revalidate() const;
  Status ConvertWoA(std::uint32_t layer, const WoATensorWriter& write) const;
  // Exact main-backbone inventory names only. No vision/MTP/auxiliary discard
  // policy is implied by converting an individual tensor.
  Status ConvertBackboneTensor(std::string_view canonical_name, const WoATensorWriter& write) const;
  // Every source tensor must be consumed by a backbone conversion (including
  // wo_a scales) or individually listed by normalized name in excluded_names.
  // Required inputs cannot be excluded; no wildcard/namespace exclusions.
  Status ValidateBackboneConversion(std::span<const std::string> excluded_names) const;
  using BackboneWriter = std::function<Status(const RuntimeWeight&, std::uint64_t,
      std::span<const std::byte>)>;
  // Admits the complete metadata disposition before any write, then converts
  // each TP1 target in inventory order. Caller retains exclusions in provenance
  // and must discard all unpublished output on any failure. No file publication.
  Status ConvertBackbone(std::span<const std::string> excluded_names, const BackboneWriter& write) const;
 private:
  struct Impl;
  explicit WeightSourceCheckpoint(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace pih::deepseek_v41
