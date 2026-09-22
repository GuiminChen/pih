#pragma once

#include "pih/model/deepseek_nccl_endpoint_warmup_runner.h"

namespace pih {

class NvidiaDeepSeekNcclWarmupPayloadOperations final
    : public DeepSeekNcclWarmupPayloadOperations {
 public:
  Status prepare(DeepSeekNcclRole role, void* device_buffer,
                 std::uint64_t bytes, DriverStreamHandle stream,
                 std::uint64_t pattern_identity) override;
  Result<Sha256Digest> digest(const void* device_buffer,
                              std::uint64_t bytes) override;
};

}  // namespace pih
