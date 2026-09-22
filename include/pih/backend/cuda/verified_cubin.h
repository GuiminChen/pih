#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

#include "pih/core/result.h"
#include "pih/core/sha256.h"

namespace pih {

class VerifiedCubin final {
 public:
  static Result<VerifiedCubin> Load(const std::filesystem::path& path,
                                    std::uint64_t maximum_bytes,
                                    const Sha256Digest& expected_digest);
  static Result<VerifiedCubin> Load(const std::filesystem::path& path,
                                    std::uint64_t maximum_bytes,
                                    std::string_view expected_sha256);

  [[nodiscard]] std::span<const std::byte> bytes() const noexcept {
    return bytes_;
  }
  [[nodiscard]] const Sha256Digest& digest() const noexcept { return digest_; }

 private:
  VerifiedCubin(std::vector<std::byte> bytes, Sha256Digest digest)
      : bytes_(std::move(bytes)), digest_(digest) {}

  std::vector<std::byte> bytes_;
  Sha256Digest digest_;
};

}  // namespace pih
