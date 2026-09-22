#pragma once

#include "pih/backend/cuda/deepseek_indexer_projection.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

class DeepSeekIndexerProjectionOperations {
 public:
  virtual ~DeepSeekIndexerProjectionOperations() = default;
  virtual Status validate_host_error(std::uint32_t* host_error) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status project(DeepSeekIndexerProjectionLaunch launch) = 0;
  virtual Status copy_error_d2h_async(std::uint32_t* host,
                                      std::uintptr_t device,
                                      std::uintptr_t stream) = 0;
};

class DeepSeekIndexerProjectionCoordinator final {
 public:
  static Result<DeepSeekIndexerProjectionCoordinator> Create(
      DeepSeekIndexerProjectionOperations& operations,
      std::uint32_t* host_error_flag);
  Status launch(const DeepSeekIndexerProjectionLaunch& submission,
                DeepSeekAttentionSequenceTransaction& transaction);

 private:
  Status run(Status status);
  DeepSeekIndexerProjectionOperations* operations_ = nullptr;
  std::uint32_t* host_error_flag_ = nullptr;
  bool poisoned_ = false;
};

}  // namespace pih
