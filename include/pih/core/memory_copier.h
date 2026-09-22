#pragma once

#include <cstdint>

#include "pih/core/device.h"

namespace pih {

class MemoryCopier {
 public:
  virtual ~MemoryCopier() = default;
  virtual Status copy(void* destination, Device destination_device,
                      const void* source, Device source_device,
                      std::uint64_t bytes) = 0;
};

class CpuMemoryCopier final : public MemoryCopier {
 public:
  Status copy(void* destination, Device destination_device,
              const void* source, Device source_device,
              std::uint64_t bytes) override;
};

}  // namespace pih
