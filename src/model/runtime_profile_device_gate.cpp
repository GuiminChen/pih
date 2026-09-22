#include "pih/model/runtime_profile_device_gate.h"

#include <algorithm>
#include <unordered_set>

#include "pih/core/canonical_hash.h"

namespace pih {
namespace {

bool uuid_nonzero(const std::array<std::byte, 16>& uuid) {
  return std::any_of(uuid.begin(), uuid.end(),
                     [](std::byte byte) { return byte != std::byte{0}; });
}

bool matches(RuntimeProfileGpuFamily family,
             const RuntimeProfileDeviceObservation& device) {
  if (family == RuntimeProfileGpuFamily::kRtx4090D24GiB) {
    return device.name == "NVIDIA GeForce RTX 4090 D" &&
           device.compute_major == 8 && device.compute_minor == 9 &&
           device.total_memory_bytes >= 24'000'000'000ULL &&
           device.total_memory_bytes < 30'000'000'000ULL;
  }
  return family == RuntimeProfileGpuFamily::kH100Pcie80GiB &&
         device.name == "NVIDIA H100 PCIe" && device.compute_major == 9 &&
         device.compute_minor == 0 &&
         device.total_memory_bytes >= 80'000'000'000ULL &&
         device.total_memory_bytes < 90'000'000'000ULL;
}

}  // namespace

VerifiedRuntimeProfileDevices::VerifiedRuntimeProfileDevices(
    RuntimeProfileGpuFamily gpu_family, std::vector<std::int32_t> ordinals,
    Sha256Digest observation_root) noexcept
    : gpu_family_(gpu_family), ordinals_(std::move(ordinals)),
      observation_root_(observation_root) {}

Result<VerifiedRuntimeProfileDevices> verify_runtime_profile_devices(
    const VerifiedRuntimeProfile& profile,
    std::span<const std::int32_t> ordered_ordinals,
    RuntimeProfileDeviceProbe& probe) {
  const auto expected_count = profile.payload().world_size();
  if (ordered_ordinals.size() != expected_count || ordered_ordinals.empty() ||
      ordered_ordinals.size() > 4) {
    return Status::FailedPrecondition(
        "runtime device count differs from signed profile");
  }
  std::unordered_set<std::int32_t> ordinal_set;
  std::unordered_set<std::string> uuid_set;
  auto hash = CanonicalHashBuilder::Create(
      "pih:runtime-profile-device-observation:v1", 2 + expected_count * 6);
  if (!hash.ok()) return hash.status();
  auto status = hash->add_u32(1, static_cast<std::uint32_t>(
                                  profile.payload().gpu_family()));
  if (status.ok()) status = hash->add_u32(2, expected_count);
  std::vector<std::int32_t> frozen_ordinals;
  frozen_ordinals.reserve(expected_count);
  for (std::size_t index = 0; status.ok() && index < expected_count; ++index) {
    const auto ordinal = ordered_ordinals[index];
    if (ordinal < 0 || !ordinal_set.insert(ordinal).second) {
      return Status::InvalidArgument("runtime device ordinals are invalid");
    }
    auto observed = probe.observe(ordinal);
    if (!observed.ok()) return observed.status();
    if (observed->ordinal != ordinal || !uuid_nonzero(observed->uuid) ||
        !matches(profile.payload().gpu_family(), *observed)) {
      return Status::FailedPrecondition(
          "observed NVIDIA device differs from signed profile");
    }
    std::string uuid(reinterpret_cast<const char*>(observed->uuid.data()),
                     observed->uuid.size());
    if (!uuid_set.insert(uuid).second) {
      return Status::FailedPrecondition("runtime device UUID is duplicated");
    }
    const auto base = static_cast<std::uint32_t>(3 + index * 6);
    status = hash->add_u32(base, static_cast<std::uint32_t>(ordinal));
    if (status.ok()) status = hash->add_ascii_utf8(base + 1, observed->name);
    if (status.ok()) status = hash->add_u32(base + 2, observed->compute_major);
    if (status.ok()) status = hash->add_u32(base + 3, observed->compute_minor);
    if (status.ok()) status = hash->add_u64(base + 4, observed->total_memory_bytes);
    if (status.ok()) status = hash->add_bytes(base + 5, observed->uuid);
    frozen_ordinals.push_back(ordinal);
  }
  if (!status.ok()) return status;
  auto root = hash->finalize();
  if (!root.ok()) return root.status();
  return VerifiedRuntimeProfileDevices(profile.payload().gpu_family(),
                                       std::move(frozen_ordinals), *root);
}

}  // namespace pih
