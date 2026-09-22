# Host spill implementation owner

`staging_lease.cpp` owns the CPU-side staging-slot lifecycle: monotonic lease
generations, fill/copy identities, producer completion and reuse rules. The file
was moved from `src/backend/cuda` without a forwarding implementation. Its
existing declaration remains under `include/pih/backend/cuda` pending header
boundary migration. This state machine does not allocate CUDA memory itself.

This directory's `implementation.cmake` declares `pih_memory_host_spill_impl`
with absolute source paths; the root build includes that definition.
The host-spill plugin consumes that target. Qwen's current native
assembly now obtains the source list from that target instead of maintaining
another hard-coded staging source path. This preserves the current implementation
but does not yet replace Qwen's direct C++ staging calls with native capability
bindings. Memory allocation/transfer integration remains a separate requirement.

Linux-target C++20 syntax and non-CUDA Windows CMake configuration passed after
the move. No GPU work, lease lifecycle tests or model requests were executed.

暂存区租约实现已移入本插件目录，旧 CUDA 源码路径不保留副本。Qwen 当前原生
组合从 host-spill 目标获取源码清单，不再硬编码旧路径；这仍不是通过能力接口
调用的完整改造。源码语法检查和关闭 CUDA 的配置通过，未运行 GPU 或模型测试。
实现目标现由本目录的 `implementation.cmake` 声明，根构建仅包含其定义。
