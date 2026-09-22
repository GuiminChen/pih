#pragma once

#include <utility>

#include "pih/backend/cuda/nvidia_deepseek_compressed_page_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_compressor_state_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_index_selection_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_indexer_projection_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_fixed_state_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_recent_state_operations.h"
#include "pih/backend/cuda/nvidia_deepseek_sparse_attention_operations.h"
#include "pih/model/deepseek_attention_runtime_resources.h"
#include "pih/contracts/deepseek_kernels_v1.h"

namespace pih {

class NvidiaDeepSeekAttentionOperationSet final {
 public:
  static Result<NvidiaDeepSeekAttentionOperationSet> Create(
      const pih_deepseek_kernels_api_v1& kernels,
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1& async_api);

  [[nodiscard]] DeepSeekAttentionRuntimeOperations borrow() noexcept;
  [[nodiscard]] DeepSeekFixedStateBankOperations& fixed_state() noexcept {
    return fixed_state_;
  }

 private:
  static Result<NvidiaDeepSeekAttentionOperationSet> CreateImpl(
      const pih_deepseek_kernels_api_v1* kernels,
      std::uintptr_t retained_context,
      const pih_nvidia_cuda_async_api_v1* async_api);

  NvidiaDeepSeekAttentionOperationSet(
      NvidiaDeepSeekRecentStateOperations recent,
      NvidiaDeepSeekCompressorStateOperations compressor,
      NvidiaDeepSeekCompressedPageOperations page,
      NvidiaDeepSeekIndexerProjectionOperations indexer_projection,
      NvidiaDeepSeekIndexSelectionOperations index_selection,
      NvidiaDeepSeekSparseAttentionOperations sparse_attention,
      NvidiaDeepSeekFixedStateOperations fixed_state) noexcept
      : recent_(std::move(recent)), compressor_(std::move(compressor)),
        page_(std::move(page)),
        indexer_projection_(std::move(indexer_projection)),
        index_selection_(std::move(index_selection)),
        sparse_attention_(std::move(sparse_attention)),
        fixed_state_(std::move(fixed_state)) {}

  NvidiaDeepSeekRecentStateOperations recent_;
  NvidiaDeepSeekCompressorStateOperations compressor_;
  NvidiaDeepSeekCompressedPageOperations page_;
  NvidiaDeepSeekIndexerProjectionOperations indexer_projection_;
  NvidiaDeepSeekIndexSelectionOperations index_selection_;
  NvidiaDeepSeekSparseAttentionOperations sparse_attention_;
  NvidiaDeepSeekFixedStateOperations fixed_state_;
};

}  // namespace pih
