#pragma once

#include <array>
#include <cstdint>
#include <span>
#include <vector>

#include "pih/core/result.h"

namespace pih {

struct DeepSeekExpertRoute final {
  std::uint16_t expert_id = 0;
  std::uint32_t token_index = 0;
  std::uint8_t route_ordinal = 0;
  float weight = 0.0F;
  bool operator==(const DeepSeekExpertRoute&) const = default;
};

class DeepSeekExpertSubwavePlan final {
 public:
  static constexpr std::uint32_t kRoutesPerToken = 6;
  static constexpr std::uint32_t kExpertCount = 256;

  static Result<DeepSeekExpertSubwavePlan> Create(
      std::uint32_t token_count, std::vector<DeepSeekExpertRoute> routes);
  static Result<DeepSeekExpertSubwavePlan> Reserve(
      std::uint32_t maximum_token_count);
  Status materialize(std::uint32_t token_count,
                     std::span<const DeepSeekExpertRoute> routes);

  [[nodiscard]] std::uint32_t token_count() const noexcept {
    return token_count_;
  }
  [[nodiscard]] const std::vector<DeepSeekExpertRoute>& routes() const noexcept {
    return routes_;
  }
  [[nodiscard]] const std::array<std::uint32_t, kExpertCount + 1>&
  expert_offsets() const noexcept {
    return expert_offsets_;
  }
  [[nodiscard]] std::uint64_t workspace_bytes() const noexcept {
    return workspace_bytes_;
  }
  [[nodiscard]] std::uint32_t maximum_token_count() const noexcept {
    return maximum_token_count_;
  }

 private:
  std::uint32_t token_count_ = 0;
  std::uint32_t maximum_token_count_ = 0;
  std::vector<DeepSeekExpertRoute> routes_;
  std::array<std::uint32_t, kExpertCount + 1> expert_offsets_{};
  std::uint64_t workspace_bytes_ = 0;
};

}  // namespace pih
