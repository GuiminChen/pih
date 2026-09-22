#include "pih/model/qwen3_bf16_execution_prelude.h"

namespace pih {

Result<QwenBf16ExecutionPrelude> QwenBf16ExecutionPrelude::Create(
    const TensorView& device_error, std::uint64_t request_generation,
    std::int32_t owning_rank) {
  if (request_generation == 0 || owning_rank < 0 ||
      device_error.dtype() != DType::kUInt8 || device_error.rank() != 1 ||
      device_error.dim(0) != 4 || device_error.stride(0) != 1 ||
      device_error.device().type() != DeviceType::kCuda ||
      device_error.device().index() != owning_rank ||
      device_error.generation() != request_generation) {
    return Status::InvalidArgument(
        "Qwen device error prelude identity is invalid");
  }
  return QwenBf16ExecutionPrelude(device_error, request_generation,
                                  owning_rank);
}

Status QwenBf16ExecutionPrelude::submit(
    QwenBf16DeviceErrorClearDriver& driver, DriverStreamHandle stream) {
  if (stream == 0) {
    return Status::InvalidArgument(
        "Qwen device error prelude requires an explicit stream");
  }
  if (submitted_) {
    return Status::FailedPrecondition(
        "Qwen device error prelude was already submitted");
  }
  submitted_ = true;
  return driver.clear_u32_async(device_error_, owning_rank_, stream);
}

}  // namespace pih
