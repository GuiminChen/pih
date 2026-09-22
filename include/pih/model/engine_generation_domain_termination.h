#pragma once

#include "pih/core/result.h"

namespace pih {

class EngineGenerationDomainDriver {
 public:
  virtual ~EngineGenerationDomainDriver() = default;
  virtual Status force_kill_domain() = 0;
  virtual Result<bool> domain_empty() = 0;
};

class EngineGenerationDomainTermination final {
 public:
  static Result<EngineGenerationDomainTermination> Create(
      EngineGenerationDomainDriver& driver);
  Status force_kill();
  Result<bool> poll_empty();
  [[nodiscard]] bool force_kill_issued() const noexcept {
    return force_kill_issued_;
  }

 private:
  explicit EngineGenerationDomainTermination(
      EngineGenerationDomainDriver& driver) noexcept : driver_(&driver) {}
  EngineGenerationDomainDriver* driver_ = nullptr;
  bool force_kill_issued_ = false;
};

}  // namespace pih
