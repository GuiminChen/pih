# Native Qwen model owner

This plugin implements `inference.text.v2` for `Qwen/Qwen3-0.6B`, with BF16 and
native PIH INT4 selection. See the [native inference guide](../../docs/native-inference.md)
for environments, artifact preparation, one-command builds, serving and observations.
All runtime/numerical/performance qualification remains unverified.

For source-only Linux-target checking without a GPU, run
`python tools/check_native_syntax.py --qwen-model --sysroot out/linux-syntax-sysroot-openssl --generated-dir /absolute/configured-build/generated --compiler clang++`
after preparing the pinned CUDA/cuBLAS syntax header trees and a CMake-generated
`pih/version.h`. The check reads the explicit Qwen and execution-default CMake
source lists (257 translation units at this revision), verifies the header
receipts, and performs C++20 syntax compilation only. It does not link the
plugin, run inference, or qualify a GPU/model profile.

The GPU-only INT4 shape and metric fixtures are not linked into the production
model plugin. To build their separate native command in an existing Linux Qwen
build tree, reconfigure with `-DPIH_BUILD_QWEN_QUALIFICATION_TOOLS=ON`, then
build target `pih-qwen-cuda-qualify`. Set `PIH_QWEN_CUBIN_PATH` to the absolute
installed cubin and invoke `pih-qwen-cuda-qualify shape|gemm|metric`. The `shape`
command emits the raw five-case JSON receipt previously obtained through
`_pih`; the optional `tests/hardware/qwen_m2_shape_corpus_smoke.py` collector
checks exact GPU identity and artifact/cubin hashes around that command. The
raw CLI output alone is not an authorizing qualification receipt. This command is excluded from normal
builds and production installation. It has not been linked or run on Linux here.

## Current capability bindings

| Required capability | Current use |
| --- | --- |
| `execution.default.v1` | Compile and validate fixed 4096-token/32-sequence capacity before GPU engine creation |
| `execution.controller.v1` | Own admission, scheduling, plan/event ledger and terminal controller retirement in the execution plugin |
| `device.cuda-runtime.v1` | Prepare the exact selected kernel-pack SM |
| `device.cuda-memory.v1` | Device/pinned allocation and release; BF16 weight H2D uploads |
| `device.cuda-resources.v1` | Retain/bind primary contexts; create/retire streams and events |
| `device.cuda-async.v1` | Record/query events, inspect thread errors, typed H2D/D2H/D2D copies and KV scrub/retirement |

Exactly one SM89 or SM90 Qwen kernel pack is required. The model validates the
capability table size/version, required callbacks and returned status/payloads.
The engine loaders require memory, resource and async tables explicitly; there
are no overloads constructing private implementations when a provider is absent.
The bundle builder includes the CUDA backend, its Linux platform dependency and
the execution plugin, and seals their hashes and registered capabilities.

The pre-release async table includes `require_clean_last_error`. Rebuild all
providers and consumers and regenerate old Locks; old table layouts are rejected.
The backend uses `cudaPeekAtLastError` with the exact retained context already
current on the calling thread. It does not clear errors or switch contexts for
that check. Typed copies validate direction, nonempty pointers/stream and address
range, check thread errors before submission and again after the copy, and
preserve a copy failure even when the final probe succeeds.

KV scrubbing uses the same bound API for zeroing, recording and polling
completion, with a nonzero bounded timeout, address-range validation and thread
error checks. Failed clear submission cannot record a successful completion;
pending events cannot release a KV slot as cleared. This driver has no CUDA
header dependency. Per-step four-byte error-buffer clearing also uses a separate
capability adapter with exact shape/rank validation and pre/post error checks.
Linear execution drivers no longer inherit the clear interface or implement a
direct CUDA memset. Matrix/kernel execution and placement still need migration.

Allocation adapters validate generation, kind, device, size, alignment and
address range. Resource destruction retires created handles before their
context, including partial construction. Failed memory/resource retirement
fail-stops the worker without unwinding/unloading uncertain owners; it is not
a recoverable HTTP error.

## Source ownership and remaining work

The model-owned BF16/INT4 engines, planning, packed controller, KV admission,
request/session lifecycle and instrumentation live here, without forwarding
implementations at their former `src/model`/`src/scheduler` paths. Eight shared
artifact implementations in `artifact/` are selected explicitly by
`plugins/offline-qwen`; model-only builds do not instantiate converter targets.

The model uses its explicit `sources.cmake` list, not the removed root
`pih_core`/`pih_cuda` aggregates. It no longer appends the execution provider's
controller/queue/scheduler implementation sources. A narrow packed-plan value
archive is linked by both DSOs so the model adapter can reconstruct and verify
the provider's wire plan before GPU execution. It no longer compiles the
concrete device allocator, pinned allocator, memory copier, runtime-resource,
completion-event or typed-copy drivers replaced by capabilities.

This is not complete plugin isolation. Remaining direct CUDA kernel/placement
operations, shared implementation dependencies and internal header boundaries
still require migration. The controller cutover has source syntax evidence only;
full Linux linking, worker graph validation and inference remain unverified.

## Configuration and verification

Model configuration requires exactly `model_directory`, `precision`,
`artifact_sha256`, `config_sha256`, `maximum_context_tokens` and
`generation_timeout_ms`. Missing or extra fields are rejected. The tokenizer
uses the pinned digest in the plugin; the artifact provider independently
authenticates immutable snapshots of all three model files before use.
The deadline covers request preparation and generation, but does not preempt a
running CPU function or CUDA call.

The earlier 265-unit Qwen Linux-target C++20 syntax check covered the old
model-plus-borrowed-execution source set. The new cutover uses the explicit
model and shared-value source set. These are syntax observations,
not a full link, sealed capability graph check, CUDA qualification or model
execution. Native regression code covers capacity, allocation, resource
retirement, event status and typed-copy error handling; the added tests have not
been compiled or executed. Five typed-copy cases were added to the existing
async contract test target with deferred discovery.

The subsequent KV-scrub change passed syntax compilation without CUDA headers;
both modified engines also passed with the prepared CUDA/cuBLAS headers. Four
additional KV regression cases were added but not compiled or executed.

The later per-step clear separation passed syntax checks for both engines and
both linear execution drivers. Four regression cases cover exact four-byte
clearing, invalid targets, submission failure and dirty-thread rejection; these
tests also remain uncompiled and unexecuted.

## 中文说明

KV 清零/回收现通过能力接口清零、记录及查询完成事件，不再直接调用 CUDA。
校验地址范围并保留有界等待，未完成不能当作清零成功。每步四字节错误缓冲区
清零也已分离为能力适配器，原矩阵驱动不再实现清零；矩阵/内核和放置等操作
仍待迁移。该 KV 驱动通过不带 CUDA 头文件的语法检查，两套引擎语法
通过；新增四个 KV 回归用例未编译执行。之后的每步清零改造也通过两套引擎与
两个矩阵驱动的语法检查，另外四个清零回归用例未编译执行。

本插件提供 Qwen3-0.6B 的 BF16/PIH INT4 原生文本接口，部署和启动命令见
[中文推理指南](../../docs/native-inference.zh.md)。目前所有运行、数值和性能资格
仍未验证，不应作为生产可用声明。

容量检查、设备准备、设备/固定页内存、上下文与流/事件资源，以及事件操作和
类型化复制已接入上表的必需能力。加载接口明确要求能力表，不保留自动构造
旧驱动的回退。一键构建包含执行插件、CUDA 后端及其 Linux 平台依赖，Lock
记录插件哈希和注册能力。必须选择且只选择一个 SM89/SM90 Kernel Pack。

异步能力新增真实的线程错误检查，要求当前上下文匹配，不清除错误、不替调用者
切换上下文。H2D/D2H/D2D 复制校验方向和地址范围，保留前后错误检查；复制失败
不会被后续成功检查覆盖。内存或资源回收失败会直接终止 Worker，避免卸载仍被
使用的提供者。旧二进制和 Lock 必须重新构建、封装，不兼容旧能力表大小。

显式源码现为 252 项，另外仍借用执行器和 host-spill 源码。上述六个具体驱动已从
模型编译清单移除；这不代表全部插件化完成。调度/请求工厂、内核和放置/清零等
直接 CUDA 操作、共享实现依赖及内部头文件边界仍需迁移。

本轮三个 C++ 文件语法通过，新增五个类型化复制回归用例但未编译执行。此前的
258 项全量语法结果属于旧清单，不能冒充当前清单完整构建通过。未做 CUDA
链接、真机推理或模型测试。配置必须提供模型目录、精度、上下文上限和生成超时；
超时不能抢占正在执行的 CPU/CUDA 调用。
