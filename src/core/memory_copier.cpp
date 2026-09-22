#include "pih/core/memory_copier.h"

#include <cstring>
#include <limits>

namespace pih {

Status CpuMemoryCopier::copy(void* destination, Device destination_device,
                             const void* source, Device source_device,
                             std::uint64_t bytes) {
  if (destination_device != Device::Cpu() || source_device != Device::Cpu()) {
    return Status::InvalidArgument("CPU copier requires CPU source and destination");
  }
  if (bytes == 0) return Status::Ok();
  if (destination == nullptr || source == nullptr) {
    return Status::InvalidArgument("nonempty copy requires source and destination");
  }
  if (bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::ResourceExhausted("copy exceeds address space");
  }
  std::memcpy(destination, source, static_cast<std::size_t>(bytes));
  return Status::Ok();
}

}  // namespace pih
