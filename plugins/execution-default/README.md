# Default execution implementation

The plugin owns the implementations of the phase-tier scheduler, controller
sequence, mailbox, request arena, output-slot arena, output burst credits,
ingress and runtime in this directory. They were moved from `src/scheduler`
without retaining forwarding files or duplicate implementations. Public/internal
declarations still use `include/pih/scheduler`; moving those headers and the
remaining controller implementations is separate work.

The two remaining implementation files formerly under `src/scheduler` were
Qwen-specific: packed batch driving and KV admission. They now live in
`plugins/model-qwen3`, not in this model-independent owner. The old monolithic
core/CUDA aggregate targets have since been removed.
No implementation `.cpp` files remain under `src/scheduler`.

The shared packed-token plan and metadata arena also now belong here, bringing
the execution implementation target to ten sources. They are declared once on
that target; the execution plugin receives its objects, while the current Qwen
native assembly receives its source list. No monolithic core archive is produced.
Both newly moved sources passed Linux-target syntax checks; the non-CUDA CMake
configuration passed. Full native linking and execution remain unverified.

`implementation.cmake` in this directory declares `pih_execution_default_impl`
and the output-credit archive; the root includes that owner definition. The
execution plugin consumes its objects. Qwen's existing source-selection path
consumes the target's absolute source paths without adding a repository prefix.
The V4.1 output-credit archive and standalone output
contract target use the same moved credit implementation, not a private copy.
This is source ownership progress, not completion of all cross-plugin factory
bindings. The current public execution capability computes capacity only;
model cancellation and retirement are still performed by model-owned code.

Every exported capacity/lifecycle callback now validates the exact provider
context before accessing state, including registration. Null or foreign state
objects are rejected without mutation. Three native regression cases cover
foreign lifecycle state, null lifecycle state and a forged ready-capacity state;
they are added to `pih-native-contracts` but have not been compiled or executed.
The modified entrypoint passed Linux-target C++20 syntax compilation. This is
ABI robustness, not isolation against a malicious in-process plugin, and does
not implement the missing scheduler/request factory bindings.

After the move, all eight source files passed Linux-target C++20 syntax checks;
the Windows non-CUDA CMake configuration generated successfully. Build scripts
and active tool/test sources contain no references to the old eight paths.
Line endings were normalized during the move. No model or runtime tests ran,
and these observations are not link or execution qualification.

默认执行插件现拥有本目录中的 8 个调度实现，旧 `src/scheduler` 路径不保留副本或
转发文件。头文件仍位于 `include/pih/scheduler`，其他控制器实现及跨插件工厂
接入仍待继续迁移。Qwen 当前源码选择、V4.1 输出额度和独立契约目标都已更新为
新路径。8 个文件通过 Linux 目标语法检查，关闭 CUDA 的 Windows CMake 配置
通过；后续旧单体聚合目标已删除，但插件绑定仍未全部完成。

当前执行能力只负责容量计算，模型取消和回收仍由模型内部处理。容量及生命周期
回调现统一验证精确的提供者上下文，拒绝空指针或其他状态对象；修改后的入口
通过 C++ 语法编译。新增三个回归用例尚未编译执行。这是 ABI 健壮性检查，
不是对恶意进程内插件的隔离，也不代表调度器/请求工厂已经接通。

实现目标和输出额度库现由本目录的 `implementation.cmake` 声明，根构建仅包含
该定义。Qwen 直接消费目标的绝对源码路径，不再重复拼接仓库根目录。
