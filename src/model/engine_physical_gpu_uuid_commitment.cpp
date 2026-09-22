#include "pih/model/engine_physical_gpu_uuid_commitment.h"

#include <string_view>

namespace pih {

Result<Sha256Digest> engine_physical_gpu_uuid_commitment(
    std::span<const std::byte> raw_uuid) {
  if (raw_uuid.size() != 16)
    return Status::InvalidArgument(
        "physical GPU UUID must contain exactly 16 bytes");
  bool nonzero = false;
  for (const auto value : raw_uuid) nonzero = nonzero || value != std::byte{0};
  if (!nonzero)
    return Status::InvalidArgument("physical GPU UUID cannot be zero");
  constexpr std::string_view domain =
      "PIH.Engine.PhysicalGpuUuidCommitment.v1";
  Sha256 hash;
  auto status = hash.update(std::as_bytes(std::span(domain)));
  if (!status.ok()) return status;
  status = hash.update(raw_uuid);
  return status.ok() ? hash.finalize() : Result<Sha256Digest>(status);
}

}  // namespace pih
