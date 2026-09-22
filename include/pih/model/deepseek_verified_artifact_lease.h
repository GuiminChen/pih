#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>

#include "pih/contracts/verified_artifact_v1.h"
#include "pih/core/result.h"
#include "pih/model/safetensors_header_receipt.h"

namespace pih {

class DeepSeekVerifiedArtifactMapping final {
 public:
  DeepSeekVerifiedArtifactMapping(const DeepSeekVerifiedArtifactMapping&) =
      delete;
  DeepSeekVerifiedArtifactMapping& operator=(
      const DeepSeekVerifiedArtifactMapping&) = delete;
  DeepSeekVerifiedArtifactMapping(DeepSeekVerifiedArtifactMapping&&) noexcept;
  DeepSeekVerifiedArtifactMapping& operator=(
      DeepSeekVerifiedArtifactMapping&&) noexcept;
  ~DeepSeekVerifiedArtifactMapping();

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept;
  Result<std::span<const std::byte>> slice(std::uint64_t offset,
                                          std::uint64_t bytes) const;
  [[nodiscard]] std::uint64_t file_offset() const noexcept {
    return mapping_.file_offset;
  }

 private:
  DeepSeekVerifiedArtifactMapping(
      const pih_verified_artifact_api_v1& api,
      pih_verified_artifact_mapping_v1 mapping) noexcept
      : api_(&api), mapping_(mapping) {}
  void release() noexcept;

  const pih_verified_artifact_api_v1* api_{};
  pih_verified_artifact_mapping_v1 mapping_{};

  friend class DeepSeekVerifiedArtifactLease;
};

class DeepSeekVerifiedArtifactLease final {
 public:
  static Result<DeepSeekVerifiedArtifactLease> OpenDevelopment(
      const pih_verified_artifact_api_v1& api,
      const std::filesystem::path& trusted_root,
      std::string_view member_basename, std::uint64_t maximum_bytes);

  DeepSeekVerifiedArtifactLease(const DeepSeekVerifiedArtifactLease&) = delete;
  DeepSeekVerifiedArtifactLease& operator=(
      const DeepSeekVerifiedArtifactLease&) = delete;
  DeepSeekVerifiedArtifactLease(DeepSeekVerifiedArtifactLease&&) noexcept;
  DeepSeekVerifiedArtifactLease& operator=(
      DeepSeekVerifiedArtifactLease&&) noexcept;
  ~DeepSeekVerifiedArtifactLease();

  [[nodiscard]] std::uint64_t file_bytes() const noexcept {
    return lease_.file_bytes;
  }
  Status read_exact(std::uint64_t offset, std::span<std::byte> output) const;
  Status poll_identity_unchanged() const;
  Result<DeepSeekVerifiedArtifactMapping> map_read_only(
      std::uint64_t file_offset, std::uint64_t bytes) const;

 private:
  DeepSeekVerifiedArtifactLease(const pih_verified_artifact_api_v1& api,
                                pih_verified_artifact_lease_v1 lease) noexcept
      : api_(&api), lease_(lease) {}
  void release() noexcept;

  const pih_verified_artifact_api_v1* api_{};
  pih_verified_artifact_lease_v1 lease_{};
};

// Reads only the bounded safetensors prefix through the storage capability.
Result<SafetensorsHeaderFileReceipt> load_safetensors_header_capability(
    const DeepSeekVerifiedArtifactLease& lease);

}  // namespace pih
