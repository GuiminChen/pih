#pragma once
#include "weight_catalog.h"
#include <filesystem>
#include <memory>

namespace pih::deepseek_v41 {
// Bounded read-only metadata snapshot; authenticate bytes before parsing.
Result<std::vector<std::byte>> ReadAuthenticatedMetadata(const std::filesystem::path& path,
    const Sha256Digest& expected, std::uint64_t maximum_bytes);
struct ExpectedWeightShard final {
  std::string name;
  std::uint64_t bytes = 0;
  Sha256Digest sha256;
};
// Linux-only owned descriptors. Expected identities must come from a trusted
// rank/config-bound manifest, not from files alongside untrusted weights.
class BackboneWeightFiles final {
 public:
  // Read fixed member weights.manifest.json, hash before parsing, bind its
  // config/world/rank, then authenticate every referenced shard through Open.
  static Result<std::unique_ptr<BackboneWeightFiles>> OpenManifest(
      const std::filesystem::path& directory, const Sha256Digest& manifest_digest,
      const FlashConfig& config, std::uint32_t world, std::uint32_t rank, std::uint64_t device_budget);
  static Result<std::unique_ptr<BackboneWeightFiles>> Open(
      const std::filesystem::path& directory, const FlashConfig& config,
      std::uint32_t world, std::uint32_t rank,
      std::span<const ExpectedWeightShard> expected, std::uint64_t device_budget);
  ~BackboneWeightFiles();
  BackboneWeightFiles(const BackboneWeightFiles&) = delete;
  BackboneWeightFiles& operator=(const BackboneWeightFiles&) = delete;
  const BackboneWeightCatalog& catalog() const noexcept;
  Status Revalidate() const;
  // Bounded, caller-owned staging (1..1 MiB). Resolve name internally rather
  // than accepting forgeable LocatedWeight offsets. Failure leaves destination
  // unusable; never upload it. Does not allocate or return raw file descriptors.
  Status Read(std::string_view tensor, std::uint64_t offset,
              std::span<std::byte> destination) const;
 private:
  struct Impl;
  explicit BackboneWeightFiles(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace pih::deepseek_v41
