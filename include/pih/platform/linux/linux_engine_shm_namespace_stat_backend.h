#pragma once

#include "pih/model/bound_engine_shm_namespace_operations.h"

namespace pih {

class LinuxEngineShmNamespaceStatBackend final
    : public EngineShmNamespaceStatBackend {
 public:
  Result<EngineShmNamespaceStat> stat_at(
      std::int32_t directory_descriptor,
      std::string_view basename) override;
};

}  // namespace pih
