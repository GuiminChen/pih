#pragma once

#include <cstdint>

#include "pih/backend/cuda/verified_kernel_launcher.h"
#include "pih/core/result.h"
#include "pih/core/tensor_view.h"

namespace pih {

class QwenBf16DeviceErrorClearDriver {
 public:
  virtual ~QwenBf16DeviceErrorClearDriver() = default;
  virtual Status clear_u32_async(const TensorView& target,
                                 std::int32_t owning_rank,
                                 DriverStreamHandle stream) = 0;
};

class QwenBf16ExecutionPrelude final {
 public:
  static Result<QwenBf16ExecutionPrelude> Create(
      const TensorView& device_error, std::uint64_t request_generation,
      std::int32_t owning_rank);

  Status submit(QwenBf16DeviceErrorClearDriver& driver,
                DriverStreamHandle stream);

  [[nodiscard]] bool submitted() const noexcept { return submitted_; }
  [[nodiscard]] std::uint64_t request_generation() const noexcept {
    return request_generation_;
  }

 private:
  QwenBf16ExecutionPrelude(TensorView device_error,
                           std::uint64_t request_generation,
                           std::int32_t owning_rank)
      : device_error_(device_error),
        request_generation_(request_generation),
        owning_rank_(owning_rank) {}

  TensorView device_error_;
  std::uint64_t request_generation_;
  std::int32_t owning_rank_;
  bool submitted_ = false;
};

}  // namespace pih
