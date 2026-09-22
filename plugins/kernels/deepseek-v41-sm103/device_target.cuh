#pragma once

// Forced into every device translation unit owned by this pack. A build flag
// override must not silently produce a different architecture under SM103's ABI.
#if !defined(__CUDACC__) && !defined(__CUDA__)
#error "SM103 device target guard requires CUDA compilation"
#endif
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 1030
#error "DeepSeek V4.1 SM103 Kernel Pack requires native SM103 device code"
#endif
