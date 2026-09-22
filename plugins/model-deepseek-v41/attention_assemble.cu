#include "attention_assemble.h"
#include <cuda_runtime.h>

namespace pih::deepseek_v41 {
namespace {
template<class T> T* Ptr(EngramDeviceRegion x) { return reinterpret_cast<T*>(x.address); }
__global__ void Indices(const int* selected, int* output, unsigned* error, unsigned start,
    unsigned ratio, unsigned window_rows, unsigned window_columns, unsigned compressed_columns) {
  const unsigned token = blockIdx.x, width = window_columns + compressed_columns;
  const auto row = static_cast<unsigned long long>(token) * width;
  for (unsigned column = threadIdx.x; column < width; column += 256) {
    int index = -1;
    if (column < window_columns) {
      const unsigned position = start ? (start % 128 + 1 + column) % 128 : (token >= 127 ? token - 127 : 0) + column;
      if (position <= (start ? start : token)) index = int(position);
    } else {
      index = selected[static_cast<unsigned long long>(token) * compressed_columns + column - window_columns];
      const unsigned visible = (start + token + 1) / ratio;
      if (index != -1 && (index < int(window_rows) || std::uint64_t(index) >= std::uint64_t(window_rows) + visible)) {
        atomicOr(error, 1U); index = -1;
      }
    }
    output[row + column] = index;
  }
}
Status Cuda(cudaError_t error) { return error == cudaSuccess ? Status::Ok() : Status::Internal(cudaGetErrorString(error)); }
}
Status LaunchAttentionAssembly(const AttentionAssemblyLaunch& x) {
  const auto validation = ValidateAttentionAssembly(x); if (!validation.ok()) return validation;
  const auto s = *GetAttentionAssemblyShape(x.start, x.tokens, x.ratio);
  const auto stream = reinterpret_cast<cudaStream_t>(x.stream);
  auto copied = Cuda(cudaMemcpyAsync(Ptr<void>(x.kv), Ptr<const void>(x.window), x.window.bytes, cudaMemcpyDeviceToDevice, stream));
  if (!copied.ok()) return copied;
  if (s.compressed_rows) {
    copied = Cuda(cudaMemcpyAsync(Ptr<unsigned char>(x.kv) + x.window.bytes, Ptr<const void>(x.compressed),
        x.compressed.bytes, cudaMemcpyDeviceToDevice, stream));
    if (!copied.ok()) return copied;
  }
  Indices<<<x.tokens, 256, 0, stream>>>(Ptr<const int>(x.selected), Ptr<int>(x.indices), Ptr<unsigned>(x.error_flag),
      x.start, x.ratio, s.window_rows, s.window_columns, s.compressed_columns);
  return Cuda(cudaGetLastError());
}
}  // namespace pih::deepseek_v41
