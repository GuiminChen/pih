#pragma once

#include <vector>

#include "pih/model/engine_owned_resource_baseline.h"

namespace pih {

enum class EngineOwnedResourceLifecycleState : std::uint8_t {
  kActive,
  kRetired,
  kPassive,
};

struct EngineOwnedResourceManifestRecord final {
  EngineOwnedResourceKind kind = EngineOwnedResourceKind::kShm;
  Sha256Digest owner_identity{};
  Sha256Digest resource_identity{};
  std::uint64_t backing_bytes = 0;
  std::uint64_t object_count = 0;
  EngineOwnedResourceLifecycleState state =
      EngineOwnedResourceLifecycleState::kActive;
};

Result<std::vector<std::byte>> encode_engine_owned_resource_manifest_record(
    const EngineOwnedResourceManifestRecord& record);

}  // namespace pih
