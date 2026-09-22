#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

class RuntimeProfileInheritedFdManifest final {
 public:
  static Result<RuntimeProfileInheritedFdManifest> Create(
      std::array<std::int32_t, 5> authority_fds,
      std::array<std::int32_t, 7> reference_fds);

  [[nodiscard]] const std::array<std::int32_t, 5>& authority_fds()
      const noexcept { return authority_fds_; }
  [[nodiscard]] const std::array<std::int32_t, 7>& reference_fds()
      const noexcept { return reference_fds_; }
  [[nodiscard]] const Sha256Digest& manifest_root() const noexcept {
    return manifest_root_;
  }
  [[nodiscard]] std::span<const std::byte> canonical_bytes() const noexcept {
    return canonical_bytes_;
  }

 private:
  RuntimeProfileInheritedFdManifest(
      std::array<std::int32_t, 5> authority_fds,
      std::array<std::int32_t, 7> reference_fds,
      Sha256Digest manifest_root,
      std::vector<std::byte> canonical_bytes) noexcept;

  std::array<std::int32_t, 5> authority_fds_{};
  std::array<std::int32_t, 7> reference_fds_{};
  Sha256Digest manifest_root_{};
  std::vector<std::byte> canonical_bytes_;
};

}  // namespace pih
