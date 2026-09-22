#pragma once

#include <cstdint>

#include "pih/core/sha256.h"
#include "pih/model/safetensors_header.h"

namespace pih {

struct SafetensorsHeaderFileReceipt final {
  SafetensorsHeader header;
  std::uint64_t file_bytes = 0;
  std::uint64_t prefix_bytes = 0;
  Sha256Digest prefix_sha256;
};

}  // namespace pih
