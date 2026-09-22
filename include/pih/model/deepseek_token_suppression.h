#pragma once

#include <cstdint>

namespace pih {

struct DeepSeekSuppressedTokenSet final {
  static constexpr std::uint32_t kMaximumTokenIds = 17;
  std::uint32_t token_ids[kMaximumTokenIds]{};
  std::uint32_t token_count = 0;
};

}  // namespace pih
