# Verified artifact storage owner

This directory owns mapped files, controller file leases, dm-verity snapshot
receipts, canonical extent writing, durable file sinks and exclusive publication.
The pure canonical extent verifier lives in core primitives so both storage and
Qwen can use it without linking another plugin's implementation. Declarations
for the older file utilities still live under `include/pih/io`.

`pih_storage_verified_artifact_impl` is the shared implementation archive used
by this plugin and native offline artifact tools. This directory's
`implementation.cmake` declares that target with absolute plugin-owned paths;
the root build includes the definition. It links the core primitives,
not the monolithic model/runtime archive. Directory ownership alone does not
prove every runtime consumer uses the storage capability instead of direct C++
calls; those remaining cross-plugin bindings must still be completed.

The `artifact.authenticated-snapshot.v1` capability is for source bytes whose
expected SHA-256 is supplied by an independently trusted deployment record.
`open_snapshot` opens one basename beneath an absolute root without following
file symlinks, copies bounded bytes to private anonymous memory, hashes the
complete copy, checks descriptor identity, and makes the pages read-only before
returning a handle/address/length. `release_snapshot` revokes that view; the
consumer must not retain borrowed pointers after release. Live snapshots block
provider drain/disposal. A failed consumer with possible asynchronous borrowers
must retain its snapshot and fail-stop rather than releasing it prematurely.
The older `artifact.verified-reader.v1` development lease remains available to
its existing consumers; Qwen's online loader now uses the snapshot capability.

The snapshot provider and Qwen consumer passed Linux-target C++20 syntax checks.
Full Linux linking, file publication, verity operation, model execution and
runtime tests have not been performed for this cutover.

本目录拥有原 `src/io` 的存储实现；纯字节 extent 校验已归入 core，供存储与
Qwen 共用，旧路径不保留副本或转发文件。
插件和原生离线工具继续使用同一实现库；头文件仍位于 `include/pih/io`。
实现库由本目录的 `implementation.cmake` 声明，根构建不再枚举存储源码。
目录迁移不代表所有运行时消费者均已通过能力接口调用，跨插件绑定仍待补齐。
新增 `artifact.authenticated-snapshot.v1` 能力从受限路径复制完整文件、验证
独立可信的 SHA-256，并在发布前把内存页设为只读。快照有显式释放句柄；仍有
借用者时不得释放，未释放快照阻止 provider 卸载。该能力及 Qwen 消费端通过
Linux 目标语法检查；尚未完成 Linux 全链接、文件/模型运行测试。
