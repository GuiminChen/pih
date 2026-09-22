#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

#include "pih/model/qwen3_numerical_run.h"

namespace pih {

struct QwenObservedDevice final {
  std::uint32_t visible_device_count;
  std::uint32_t current_ordinal;
  std::string name;
  std::uint32_t compute_major;
  std::uint32_t compute_minor;
  std::uint64_t total_global_memory_bytes;
  std::array<std::byte, 16> uuid;
  std::uint32_t pci_domain;
  std::uint32_t pci_bus;
  std::uint32_t pci_device;
  std::uint32_t driver_version;
  std::uint32_t runtime_version;
};

class QwenTargetObservation final {
 public:
  static Result<QwenTargetObservation> Create(
      QwenNumericalRunIdentity identity, QwenObservedDevice device);

  [[nodiscard]] QwenTargetGpu target_gpu() const noexcept {
    return identity_.target_gpu();
  }
  [[nodiscard]] std::uint32_t target_sm() const noexcept {
    return identity_.target_sm();
  }
  [[nodiscard]] const QwenNumericalRunIdentity& identity() const noexcept {
    return identity_;
  }
  [[nodiscard]] const Sha256Digest& semantic_digest() const noexcept {
    return semantic_digest_;
  }

 private:
  QwenTargetObservation(QwenNumericalRunIdentity identity,
                        QwenObservedDevice device, Sha256Digest digest)
      : identity_(std::move(identity)), device_(std::move(device)),
        semantic_digest_(digest) {}

  QwenNumericalRunIdentity identity_;
  QwenObservedDevice device_;
  Sha256Digest semantic_digest_;
};

}  // namespace pih
