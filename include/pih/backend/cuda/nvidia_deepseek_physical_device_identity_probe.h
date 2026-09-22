#pragma once

#include "pih/platform/linux/linux_deepseek_rank_worker_handshake_operations.h"

namespace pih {

class NvidiaDeepSeekPhysicalDeviceIdentityProbe final
    : public DeepSeekPhysicalDeviceIdentityProbe {
 public:
  static Result<NvidiaDeepSeekPhysicalDeviceIdentityProbe> Create(
      std::int32_t startup_device_ordinal);
  Result<Sha256Digest> current_physical_device_uuid_commitment() override;
  Result<std::int32_t> startup_device_ordinal() override { return ordinal_; }

 private:
  explicit NvidiaDeepSeekPhysicalDeviceIdentityProbe(
      std::int32_t ordinal) noexcept : ordinal_(ordinal) {}
  std::int32_t ordinal_ = -1;
};

}  // namespace pih
