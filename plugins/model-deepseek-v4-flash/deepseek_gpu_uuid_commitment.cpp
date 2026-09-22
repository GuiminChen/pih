#include "pih/model/deepseek_gpu_uuid_commitment.h"

#include <array>
#include <string_view>

namespace pih {

Result<Sha256Digest> deepseek_gpu_uuid_commitment(
    std::span<const std::byte> raw_uuid) {
  if (raw_uuid.size() != 16)
    return Status::InvalidArgument("DeepSeek GPU UUID must contain exactly 16 bytes");
  bool nonzero = false;
  for (const auto byte : raw_uuid) nonzero = nonzero || byte != std::byte{0};
  if (!nonzero)
    return Status::InvalidArgument("DeepSeek GPU UUID cannot be zero");
  constexpr std::string_view domain =
      "PIH.DeepSeek.PhysicalGpuUuidCommitment.v1\0";
  Sha256 hash;
  auto status = hash.update(std::as_bytes(std::span(domain.data(), domain.size())));
  if (!status.ok()) return status;
  status = hash.update(raw_uuid);
  return status.ok() ? hash.finalize() : Result<Sha256Digest>(status);
}

}  // namespace pih
