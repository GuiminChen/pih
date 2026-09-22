#pragma once

#include <cstdint>

#include "pih/core/result.h"

namespace pih {

enum class DeviceType : std::uint8_t {
  kCpu = 0,
  kCuda,
};

class Device final {
 public:
  static Result<Device> Create(DeviceType type, std::int32_t index) {
    switch (type) {
      case DeviceType::kCpu:
        if (index != 0) {
          return Status::InvalidArgument("CPU device index must be zero");
        }
        return Device(type, index);
      case DeviceType::kCuda:
        if (index < 0) {
          return Status::InvalidArgument("CUDA device index must be nonnegative");
        }
        return Device(type, index);
    }
    return Status::InvalidArgument("unknown device type");
  }

  static Device Cpu() { return Device(DeviceType::kCpu, 0); }

  [[nodiscard]] DeviceType type() const noexcept { return type_; }
  [[nodiscard]] std::int32_t index() const noexcept { return index_; }

  friend bool operator==(const Device&, const Device&) = default;

 private:
  Device(DeviceType type, std::int32_t index) : type_(type), index_(index) {}

  DeviceType type_;
  std::int32_t index_;
};

}  // namespace pih
