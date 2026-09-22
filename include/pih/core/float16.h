#pragma once

#include <cstdint>

namespace pih {

struct Float16 final {
  std::uint16_t bits = 0;

  static Float16 FromFloat(float value) noexcept;
  [[nodiscard]] float to_float() const noexcept;

  friend bool operator==(const Float16&, const Float16&) = default;
};

static_assert(sizeof(Float16) == 2);

}  // namespace pih
