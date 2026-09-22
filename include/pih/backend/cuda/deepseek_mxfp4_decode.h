#pragma once

#include <cstdint>

#include "pih/core/status.h"

namespace pih {

struct DeepSeekMxfp4Tile final {
  std::uintptr_t packed = 0;
  std::uintptr_t scale_bits = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t error_flag = 0;
  std::uintptr_t stream = 0;
  std::uint64_t logical_k = 0;
  std::uint64_t tile_begin = 0;
  std::uint32_t tile_values = 0;
};

Status launch_deepseek_mxfp4_decode_tile(DeepSeekMxfp4Tile tile);

}  // namespace pih
