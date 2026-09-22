#pragma once

#include <cstdint>
#include <memory>

#include "pih/core/memory_copier.h"

namespace pih {

// Startup-only synchronous host-to-device transfer authority. Successful
// return is the publication barrier: all source bytes are resident and visible
// before ResidentWeightSet exposes its immutable catalog.
class CudaMemoryCopier final : public MemoryCopier {
 public:
  static Result<std::unique_ptr<CudaMemoryCopier>> Create(
      std::int32_t device_index);

  Status copy(void* destination, Device destination_device,
              const void* source, Device source_device,
              std::uint64_t bytes) override;

  [[nodiscard]] std::int32_t device_index() const noexcept {
    return device_index_;
  }

 private:
  explicit CudaMemoryCopier(std::int32_t device_index)
      : device_index_(device_index) {}

  std::int32_t device_index_;
};

}  // namespace pih
