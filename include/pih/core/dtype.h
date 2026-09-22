#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class DType : std::uint8_t {
  kFloat32 = 0,
  kFloat16,
  kBFloat16,
  kInt8,
  kUInt8,
  kInt32,
  kInt64,
  kBool,
  // Opaque one-byte checkpoint storage formats. Numerical interpretation is
  // format-specific; loaders must preserve their bits.
  kFloat8E4M3,
  kFloat8E8M0,
  // Runtime-only metric and indexing types. Appended to preserve the stable
  // numeric values of checkpoint-facing dtypes above.
  kFloat64,
  kUInt32,
};

inline Result<std::uint64_t> dtype_size(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
    case DType::kInt32:
    case DType::kUInt32:
      return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kInt64:
    case DType::kFloat64:
      return 8;
    case DType::kInt8:
    case DType::kUInt8:
    case DType::kBool:
    case DType::kFloat8E4M3:
    case DType::kFloat8E8M0:
      return 1;
  }
  return Status::InvalidArgument("unknown dtype");
}

}  // namespace pih
