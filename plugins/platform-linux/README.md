# Linux platform implementation boundary

The 26 implementation files previously under `src/platform/linux` now live in
this directory. They implement Linux process launch, rank artifact transfer,
cgroup observation/control, pidfd retirement, shutdown channels, descriptor and
namespace observation, and profile lease probes. The old directory has no
forwarding or duplicate `.cpp` implementations. Header declarations retain their
current `include/pih/platform/linux` paths.

This directory's `implementation.cmake` owns the Linux-only
`pih_platform_linux_impl` object target and its absolute source paths. The root
build includes that definition rather than enumerating platform sources.
The plugin entrypoint's current capability table is unchanged: this move does
not expose all of those operations through the native platform ABI or link the
whole rank-specific implementation into the minimal platform shared object.
Remaining model-dependent interfaces need explicit capability binding; do not
infer complete plugin isolation from the directory move.

Independent Linux-target C++20 syntax compilation of all 26 sources passed after
adding missing direct `<cstdint>` dependencies to the cgroup probe/controller,
termination signal source and pidfd reaping ledger headers. The non-CUDA Windows
configuration generated successfully, but does not exercise Linux target
creation. No processes, cgroups, transfers or model tests were executed.

原 `src/platform/linux` 的 26 个实现已迁入本目录，构建清单更新为新路径，旧目录
不保留转发或重复实现。头文件路径暂不变。此次迁移不改变平台插件的能力表，也
不把全部 rank 专用实现强行链接到最小平台插件；剩余跨模型接口仍需能力绑定。
独立编译补齐了 4 个头文件的 `<cstdint>` 直接依赖后，26 个源码的 Linux 目标
语法检查通过。关闭 CUDA 的 Windows 配置通过，但不能验证 Linux 专用目标；
未启动进程或操作 cgroup，也未运行模型测试。
实现目标及 Linux 平台条件现由本目录的 `implementation.cmake` 管理，根构建
不再枚举这 26 个源码；Windows 配置仍不能验证 Linux 目标创建和链接。
