#pragma once
#include "pih/core/result.h"
#include <array>
#include <chrono>
#include <cstddef>
#include <span>

namespace pih::deepseek_v41 {
// Private inherited AF_UNIX SOCK_SEQPACKET connection. One 144-byte ID packet,
// then no application traffic: the broker stays alive until process retirement.
class NcclBootstrapChannel final {
 public:
  using Clock = std::chrono::steady_clock;
  using Id = std::array<std::byte, 128>;
  static Result<std::array<std::byte, 144>> Encode(std::span<const std::byte> id);
  static Result<Id> Decode(std::span<const std::byte> packet);
  static Status ValidateSocket(int fd);
};
}  // namespace pih::deepseek_v41
