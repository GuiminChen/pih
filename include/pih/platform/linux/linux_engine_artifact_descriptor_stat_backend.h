#pragma once

#include "pih/model/bound_engine_artifact_descriptor_operations.h"

namespace pih {

class LinuxEngineArtifactDescriptorStatBackend final
    : public EngineArtifactDescriptorStatBackend {
 public:
  Result<EngineArtifactDescriptorStat> stat(
      std::int32_t descriptor) override;
};

}  // namespace pih
