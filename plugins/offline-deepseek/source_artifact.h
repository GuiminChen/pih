#pragma once
#include <filesystem>
#include <memory>
#include <vector>
#include "pih/core/sha256.h"
#include "pih/model/safetensors_header.h"
namespace pih::offline_deepseek {
struct SourceArtifactObject final {
  std::string name;
  std::uint64_t bytes = 0;
  Sha256Digest sha256;
  Sha256Digest object_root;
};
// Owns the exact 50 regular-file descriptors. No symlink path components;
// expected model digest must originate outside the untrusted source directory.
// Byte identity admission only, not safetensors semantic/inventory admission.
class SourceArtifact final {
 public:
  static Result<std::unique_ptr<SourceArtifact>> Open(
      const std::filesystem::path& directory, const Sha256Digest& expected_model_digest);
  ~SourceArtifact();
  SourceArtifact(const SourceArtifact&) = delete;
  SourceArtifact& operator=(const SourceArtifact&) = delete;
  const std::vector<int>& descriptors() const noexcept;
  // Order: config, index, numeric shards 1..48; matches descriptors().
  const std::vector<SourceArtifactObject>& objects() const noexcept;
  const Sha256Digest& model_digest() const noexcept;
  Status Revalidate() const;
  // Reads actual headers, joins every name/shard to the admitted index, checks
  // complete counts/bytes and full source tensor geometry. Typed semantic and
  // inventory roots are not derived here. Order is numeric shards.
  Result<std::vector<SafetensorsHeader>> ReadShardHeaders() const;
 private:
  struct Impl;
  explicit SourceArtifact(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};
}  // namespace pih::offline_deepseek
