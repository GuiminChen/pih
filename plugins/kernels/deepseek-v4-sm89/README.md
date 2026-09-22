# DeepSeek-V4 SM89 Kernel Pack

`kernel_pack.cpp` exports the native Kernel Pack contract. Its private
implementations live in `native/`: 17 CUDA translation units and 17 C++ launch,
shape and resource-contract implementations. They were moved from the historical
`kernels/cuda` and `src/backend/cuda` directories without forwarding copies.
The shared `src/backend/cuda/cuda_status.cpp` remains an explicitly selected
dependency because it is also used outside this pack.

`native.cmake` owns the two implementation object targets. It preserves the
SM89-only `89-real` architecture, CUDA C++20 setting, native DSpark exclusion,
hidden symbols, position-independent objects and CUDA direct-link allowlists.
The root admits the CUDA toolkit and includes this definition; it no longer
enumerates this pack's implementation files. The shared pack itself is built by
this directory's `CMakeLists.txt`. Internal header ownership still needs migration.

After the move, all 17 private C++ implementations plus the pack entry and shared
CUDA status implementation passed Linux-target C++20 syntax compilation. All 17
CUDA implementations passed Clang host-only and SM89 device-only syntax checks
with the pinned CUDA 13.2.86 headers. Clang 22.1.8 warns that this SDK is newer
than its latest partially supported version; device parsing explicitly selected
`-Xclang -target-sdk-version=12.9`. No warning was hidden. These are compatibility
syntax observations, not NVCC compilation, PTX/cubin generation, library linking,
kernel execution, numerical qualification or hardware support. No tests were run.

Use the [native inference guide](../../../docs/native-inference.md) to build the
whole DeepSeek native bundle and regenerate its Lock. Do not substitute this
SM89 pack for another architecture or infer B300/SM103 support from it.

本内核包现拥有 `native/` 中的 17 个 CUDA 实现和 17 个 C++ 调用/契约实现，
旧目录不保留转发副本；共用 `cuda_status.cpp` 仍显式引用原位置。构建定义位于
`native.cmake`，根构建仅负责 CUDA 准入和包含该文件，不再枚举这些实现。
固定 SM89、隐藏符号、PIC、DSpark 排除宏及直接链接限制保持不变。

迁移后 19 个相关 C++ 编译单元以及 17 个 CUDA 文件的主机/设备侧语法检查通过。
CUDA 检查使用 Clang 和 13.2.86 头文件，保留版本警告，并明确选择 12.9 兼容
设备解析设置。这不是 NVCC 编译、链接、内核运行、数值或硬件资格验证；未执行
测试，也不能据此宣称 B300 支持。
