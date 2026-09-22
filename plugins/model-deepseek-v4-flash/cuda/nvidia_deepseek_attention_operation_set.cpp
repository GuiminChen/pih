#include "pih/backend/cuda/nvidia_deepseek_attention_operation_set.h"

namespace pih {

Result<NvidiaDeepSeekAttentionOperationSet>
NvidiaDeepSeekAttentionOperationSet::Create(
    const pih_deepseek_kernels_api_v1& kernels,
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1& async_api) {
  return CreateImpl(&kernels, retained_context, &async_api);
}

Result<NvidiaDeepSeekAttentionOperationSet>
NvidiaDeepSeekAttentionOperationSet::CreateImpl(
    const pih_deepseek_kernels_api_v1* kernels,
    std::uintptr_t retained_context,
    const pih_nvidia_cuda_async_api_v1* async_api) {
  if (kernels == nullptr || async_api == nullptr || retained_context == 0) {
    return Status::InvalidArgument("DeepSeek attention requires plugin capabilities");
  }
  auto recent = NvidiaDeepSeekRecentStateOperations::Create(retained_context,
                                                             *async_api);
  if (!recent.ok()) return recent.status();
  auto compressor = NvidiaDeepSeekCompressorStateOperations::Create(
      retained_context, *async_api, *kernels);
  if (!compressor.ok()) return compressor.status();
  auto page = NvidiaDeepSeekCompressedPageOperations::Create(
      retained_context, *async_api, *kernels);
  if (!page.ok()) return page.status();
  auto indexer_projection = NvidiaDeepSeekIndexerProjectionOperations::Create(
      retained_context, *async_api, *kernels);
  if (!indexer_projection.ok()) return indexer_projection.status();
  auto index_selection = NvidiaDeepSeekIndexSelectionOperations::Create(
      retained_context, *async_api, *kernels);
  if (!index_selection.ok()) return index_selection.status();
  auto sparse_attention = NvidiaDeepSeekSparseAttentionOperations::Create(
      retained_context, *async_api, *kernels);
  if (!sparse_attention.ok()) return sparse_attention.status();
  auto fixed_state =
      NvidiaDeepSeekFixedStateOperations::Create(retained_context, *async_api);
  if (!fixed_state.ok()) return fixed_state.status();
  return NvidiaDeepSeekAttentionOperationSet(
      std::move(*recent), std::move(*compressor), std::move(*page),
      std::move(*indexer_projection), std::move(*index_selection),
      std::move(*sparse_attention),
      std::move(*fixed_state));
}

DeepSeekAttentionRuntimeOperations
NvidiaDeepSeekAttentionOperationSet::borrow() noexcept {
  return {&recent_, &compressor_, &page_, &indexer_projection_,
          &index_selection_, &sparse_attention_};
}

}  // namespace pih
