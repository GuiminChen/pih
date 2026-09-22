#include "pih/model/qwen3_target_observation.h"

#include <algorithm>
#include <span>
#include <string>
#include <utility>

namespace pih {
namespace {

std::string uuid_hex(const std::array<std::byte, 16>& uuid) {
  constexpr char digits[] = "0123456789abcdef";
  std::string result(32, '0');
  for (std::size_t i = 0; i < uuid.size(); ++i) {
    const auto value = std::to_integer<unsigned int>(uuid[i]);
    result[i * 2] = digits[value >> 4];
    result[i * 2 + 1] = digits[value & 15];
  }
  return result;
}

}  // namespace

Result<QwenTargetObservation> QwenTargetObservation::Create(
    QwenNumericalRunIdentity identity, QwenObservedDevice device) {
  const bool uuid_nonzero =
      std::any_of(device.uuid.begin(), device.uuid.end(),
                  [](std::byte value) { return value != std::byte{0}; });
  if (device.visible_device_count != 1 || device.current_ordinal != 0 ||
      device.name.empty() || device.total_global_memory_bytes == 0 ||
      !uuid_nonzero || device.driver_version < 13020 ||
      device.runtime_version < 13020) {
    return Status::InvalidArgument("Qwen target device observation is invalid");
  }
  const bool rtx = identity.target_gpu() == QwenTargetGpu::kRtx4090D &&
                   device.name == "NVIDIA GeForce RTX 4090 D" &&
                   device.compute_major == 8 && device.compute_minor == 9 &&
                   device.total_global_memory_bytes >= 24ULL * 1000 * 1000 * 1000;
  const bool h100 = identity.target_gpu() == QwenTargetGpu::kH100Pcie80Gb &&
                    device.name == "NVIDIA H100 PCIe" &&
                    device.compute_major == 9 && device.compute_minor == 0 &&
                    device.total_global_memory_bytes >= 80ULL * 1000 * 1000 * 1000;
  if (!rtx && !h100) {
    return Status::FailedPrecondition(
        "observed NVIDIA device differs from Qwen target identity");
  }
  auto identity_digest = identity.semantic_digest();
  if (!identity_digest.ok()) return identity_digest.status();
  const std::string canonical =
      "pih.qwen_target_observation.v1\n" + identity_digest->hex() + "\n" +
      device.name + "\n" + std::to_string(device.compute_major) + "." +
      std::to_string(device.compute_minor) + "\n" +
      std::to_string(device.total_global_memory_bytes) + "\n" +
      uuid_hex(device.uuid) + "\n" + std::to_string(device.pci_domain) + ":" +
      std::to_string(device.pci_bus) + ":" + std::to_string(device.pci_device) +
      "\n" + std::to_string(device.driver_version) + "\n" +
      std::to_string(device.runtime_version);
  auto digest = sha256(std::as_bytes(std::span(canonical.data(), canonical.size())));
  if (!digest.ok()) return digest.status();
  return QwenTargetObservation(std::move(identity), std::move(device),
                               digest.value());
}

}  // namespace pih
