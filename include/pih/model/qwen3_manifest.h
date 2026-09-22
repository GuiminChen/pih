#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pih/model/safetensors_header.h"

namespace pih {

struct ExpectedTensor final {
  std::string name;
  std::vector<std::uint64_t> shape;
};

class Qwen3Manifest final {
 public:
  static constexpr std::uint64_t kOfficialSourcePayloadBytes = 1'503'264'768;
  static constexpr std::size_t kOfficialTensorCount = 311;

  static const std::vector<ExpectedTensor>& expected_tensors();
  static Status Validate(const SafetensorsHeader& header);
};

}  // namespace pih
