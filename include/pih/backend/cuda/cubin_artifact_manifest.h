#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "pih/backend/cuda/verified_cubin.h"
#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

class CubinArtifactManifest final {
 public:
  static constexpr std::uint64_t kMaximumManifestBytes = 4096;

  static Result<CubinArtifactManifest> Parse(std::string_view json);
  static Result<CubinArtifactManifest> Load(
      const std::filesystem::path& path,
      std::uint64_t maximum_bytes = kMaximumManifestBytes);

  Result<VerifiedCubin> load_cubin(
      const std::filesystem::path& path,
      std::uint64_t configured_maximum_bytes) const;

  [[nodiscard]] std::uint32_t target_sm() const noexcept { return target_sm_; }
  [[nodiscard]] std::string_view producer_toolkit() const noexcept {
    return producer_toolkit_;
  }
  [[nodiscard]] const Sha256Digest& producer_flags_digest() const noexcept {
    return producer_flags_digest_;
  }
  [[nodiscard]] const Sha256Digest& cubin_digest() const noexcept {
    return cubin_digest_;
  }
  [[nodiscard]] std::uint64_t cubin_bytes() const noexcept {
    return cubin_bytes_;
  }

 private:
  CubinArtifactManifest(std::uint32_t target_sm, std::string producer_toolkit,
                        Sha256Digest producer_flags_digest,
                        Sha256Digest cubin_digest, std::uint64_t cubin_bytes)
      : target_sm_(target_sm),
        producer_toolkit_(std::move(producer_toolkit)),
        producer_flags_digest_(producer_flags_digest),
        cubin_digest_(cubin_digest),
        cubin_bytes_(cubin_bytes) {}

  std::uint32_t target_sm_ = 0;
  std::string producer_toolkit_;
  Sha256Digest producer_flags_digest_;
  Sha256Digest cubin_digest_;
  std::uint64_t cubin_bytes_ = 0;
};

}  // namespace pih
