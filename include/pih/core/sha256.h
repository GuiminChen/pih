#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

#include "pih/core/result.h"

namespace pih {

struct Sha256Digest final {
  std::array<std::byte, 32> bytes{};

  static Result<Sha256Digest> ParseHex(std::string_view value);
  [[nodiscard]] std::string hex() const;
  friend bool operator==(const Sha256Digest&, const Sha256Digest&) = default;
};

class Sha256 final {
 public:
  Sha256();

  Status update(std::span<const std::byte> input);
  Result<Sha256Digest> finalize();

 private:
  void compress(const std::byte* block) noexcept;

  std::array<std::uint32_t, 8> state_{};
  std::array<std::byte, 64> buffer_{};
  std::uint64_t total_bytes_ = 0;
  std::size_t buffered_bytes_ = 0;
  bool finalized_ = false;
};

Result<Sha256Digest> sha256(std::span<const std::byte> input);

}  // namespace pih
