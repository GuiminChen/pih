#include "pih/model/deepseek_boundary_health_state.h"

namespace pih {

Status DeepSeekBoundaryHealthState::report_device_error(
    std::uint32_t error_code) noexcept {
  if (error_code == 0) {
    return Status::InvalidArgument(
        "DeepSeek boundary device error code is zero");
  }
  std::uint32_t expected = 0;
  if (!device_error_code_.compare_exchange_strong(
          expected, error_code, std::memory_order_acq_rel,
          std::memory_order_acquire) && expected != error_code) {
    return Status::FailedPrecondition(
        "DeepSeek boundary device error was already recorded");
  }
  poison();
  return Status::Ok();
}

}  // namespace pih
