#include "pih/model/engine_allocation_lease.h"

#include <array>

namespace pih {
namespace {

bool zero(const Sha256Digest& value) noexcept {
  return value == Sha256Digest{};
}

}  // namespace

Result<Sha256Digest> engine_allocation_lease_digest(
    const EngineAllocationLeaseManifest& manifest) {
  const auto& devices = manifest.sorted_physical_gpu_identities;
  if (zero(manifest.deployment_instance_digest) || devices.empty() ||
      devices.size() > 4)
    return Status::InvalidArgument("engine allocation lease manifest is invalid");
  std::array<std::byte, 4 * sizeof(std::uint64_t)> encoded{};
  for (std::size_t index = 0; index < devices.size(); ++index) {
    if (devices[index] == 0 ||
        (index != 0 && devices[index - 1] >= devices[index]))
      return Status::InvalidArgument(
          "engine allocation GPU identities are not canonical");
    for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
      encoded[index * sizeof(std::uint64_t) + byte] =
          static_cast<std::byte>(devices[index] >> (byte * 8U));
  }
  auto builder = CanonicalHashBuilder::Create(
      "pih:engine_instance_lease_v1", 3);
  if (!builder.ok()) return builder.status();
  auto status = builder->add_hash(1, manifest.deployment_instance_digest);
  if (status.ok())
    status = builder->add_u32(2, static_cast<std::uint32_t>(devices.size()));
  if (status.ok())
    status = builder->add_bytes(
        3, std::span<const std::byte>(encoded).first(
               devices.size() * sizeof(std::uint64_t)));
  if (!status.ok()) return status;
  return builder->finalize();
}

Status verify_engine_allocation_lease(
    const EngineAllocationLeaseExpectation& expectation,
    const EngineAllocationLeaseObservation& observation) {
  if (zero(expectation.lease_digest) || expectation.filesystem_identity == 0 ||
      expectation.file_identity == 0)
    return Status::InvalidArgument(
        "engine allocation lease expectation is incomplete");
  if (!observation.descriptor_open ||
      !observation.exclusive_ofd_lock_held ||
      observation.token_digest != expectation.lease_digest ||
      observation.filesystem_identity != expectation.filesystem_identity ||
      observation.file_identity != expectation.file_identity)
    return Status::FailedPrecondition(
        "engine allocation lease observation drifted");
  return Status::Ok();
}

}  // namespace pih
