#pragma once

#include <cstddef>
#include <cstdint>

#include "pih/backend/cuda/deepseek_sparse_attention.h"
#include "pih/model/deepseek_attention_index_assembler.h"
#include "pih/model/deepseek_attention_sequence_transaction.h"

namespace pih {

struct DeepSeekSparseAttentionHostStaging final {
  std::int32_t* indices = nullptr;
  std::uint32_t index_capacity = 0;
  std::uint32_t* error_flag = nullptr;
  std::uint32_t* page_slots = nullptr;
  std::uint32_t page_slot_capacity = 0;
};

struct DeepSeekSparseAttentionSubmission final {
  const DeepSeekSparseIndexMatrix* index_matrix = nullptr;
  std::uintptr_t query_bf16 = 0;
  std::uintptr_t latent_kv_bf16 = 0;
  std::uintptr_t attention_sink_f32 = 0;
  std::uintptr_t device_indices_i32 = 0;
  std::uintptr_t output_bf16 = 0;
  std::uintptr_t device_error_flag_u32 = 0;
  std::uintptr_t stream = 0;
  std::uint32_t head_count = 0;
  std::uint32_t kv_count = 0;
  std::uintptr_t compressed_kv_bf16 = 0;
  std::uintptr_t device_page_slots_u32 = 0;
  std::int32_t recent_physical_offset = 0;
  std::int32_t compressed_physical_offset = 0;
  std::uint32_t compressed_slot_count = 0;
};

class DeepSeekSparseAttentionOperations {
 public:
  virtual ~DeepSeekSparseAttentionOperations() = default;
  virtual Status validate_host_staging(
      const DeepSeekSparseAttentionHostStaging& staging) = 0;
  virtual Status zero_u32_async(std::uintptr_t device,
                                std::uintptr_t stream) = 0;
  virtual Status copy_h2d_async(std::uintptr_t device, const void* host,
                                std::size_t bytes,
                                std::uintptr_t stream) = 0;
  virtual Status attention(DeepSeekSparseAttentionLaunch launch) = 0;
  virtual Status copy_d2h_async(void* host, std::uintptr_t device,
                                std::size_t bytes,
                                std::uintptr_t stream) = 0;
};

class DeepSeekSparseAttentionDriver final {
 public:
  static Result<DeepSeekSparseAttentionDriver> Create(
      DeepSeekSparseAttentionOperations& operations,
      DeepSeekSparseAttentionHostStaging staging);

  Status validate(const DeepSeekSparseAttentionSubmission& submission,
                  const DeepSeekSparseIndexMatrix& index_matrix,
                  const DeepSeekAttentionSequenceTransaction& transaction)
      const;
  Status launch(const DeepSeekSparseAttentionSubmission& submission,
                DeepSeekAttentionSequenceTransaction& transaction);
  Status launch_paged(const DeepSeekSparseAttentionSubmission& submission,
                      std::span<const std::uint32_t> page_slots,
                      std::uint32_t physical_page_count,
                      DeepSeekAttentionSequenceTransaction& transaction);

 private:
  DeepSeekSparseAttentionOperations* operations_ = nullptr;
  DeepSeekSparseAttentionHostStaging staging_;
  bool poisoned_ = false;

  Status launch_impl(const DeepSeekSparseAttentionSubmission& submission,
                     std::span<const std::uint32_t> page_slots,
                     std::uint32_t physical_page_count,
                     DeepSeekAttentionSequenceTransaction& transaction);
};

}  // namespace pih
