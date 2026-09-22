# Native DeepSeek-V4-Flash model owner

The artifact epoch guard now consumes the model-local capability poller instead
of declaring a removed controller-catalog overload. The engine routes storage
polls through this guard, including topology checks, provider exceptions and
sticky first-failure state. Its six CPU regression cases now use scripted
capability observations and are wired to `pih-native-contracts`; they have not
been compiled or run. These cases do not replace storage-plugin filesystem
integrity tests. Guard, engine and bootstrap passed Linux-target syntax checks;
the refreshed 212-unit model/backend/default selection also passed after removing
the guard's transitive controller-catalog dependency. No linking was performed.
Historical controller catalog/handoff consumers outside this owner remain.

制品 epoch 守卫已改用模型内的能力检查接口，不再声明已删除实现的旧目录重载。
引擎的租约检查统一经过守卫，核对拓扑、捕获提供方异常并保留首次失败状态。
六个 CPU 回归用例已迁移到能力观察结果并接入原生测试目标，尚未编译或执行；
它们不替代存储插件的文件系统完整性测试。守卫、引擎和启动层通过语法编译；
移除间接目录依赖后，212 个模型/后端/默认入口翻译单元的复查也通过，未做链接。
目录外旧控制器目录及交接调用方仍待迁移。

H2D, request-input copy, recent-state and fixed-state adapters now have no
native/legacy switch: twenty-seven
preprocessor decisions, the no-capability constructors, direct CUDA transfer
fallbacks and unused current-context helpers were removed. H2D explicitly binds
either CUDA async or host-spill; the other three require CUDA async. The four
implementations passed Linux-target syntax compilation in two separate groups
without CUDA headers or the native-selection macro. Added static regression
guards were not executed.
The remaining owner-local switches have since been removed: 189 decisions across
34 source/header files, deleting 1,490 lines of selectors and legacy branches.
The owner build and syntax checker no longer define the native-selection macro.
DSpark/NCCL feature guards are unchanged; old code outside this owner still needs
separate migration. This does not establish whole-repository completion.

After this removal, non-CUDA CMake configuration passed and the 161 model
compilation entries no longer contain the retired selector. The combined
model/backend/default selection (212 C++ translation units) passed Linux-target
syntax compilation without that selector. No linking or tests were performed.

H2D、请求输入复制、近期状态和固定状态共删除 27 处条件选择、无能力参数入口
和直接 CUDA 回退。H2D 明确选择异步或 host-spill 能力，其他三者必须绑定异步
能力；四个实现分两组通过不带 CUDA 头文件及原生选择宏的语法编译。新增静态
回归守卫未执行。随后又清理了所属目录及配套头文件中的 189 处条件选择，
涉及 34 个文件、删除 1,490 行选择器及旧分支。所属构建和语法检查工具不再
定义原生选择宏；DSpark/NCCL 功能开关保持不变。目录外历史代码仍需单独迁移，
这不代表全仓库改造完成。

本次清理后，非 CUDA CMake 配置通过，161 个模型编译条目已不含旧选择宏。
模型、后端和默认入口合计 212 个 C++ 翻译单元通过不带该宏的 Linux 目标语法
编译；未进行链接或测试。

The model entrypoint, generation and text-completion adapters live here with
the 161 C++ model implementations selected by the native PP1 build. Those
implementations were moved from `src/model` without forwarding copies. The
selection was taken from the actual CMake compilation database, not inferred
from filenames alone. Core no longer builds a monolithic inference archive.

The rank runtime implementation and declaration now contain only the native
capability path. Seventeen native/legacy preprocessor decisions were resolved:
the no-capability constructor and private native allocator/resource-driver
fallbacks were removed, not hidden behind a build flag. The public constructor
requires memory, resources, async and Kernel Pack tables; host-spill remains an
explicit optional capability, not a legacy inference fallback. The owner-local
adapters now also use only the capability branches, without a native-selection
macro. Historical DeepSeek code outside this owner has not all been retired.

rank runtime 的实现及声明已删除 17 处原生/旧路径条件选择，仅保留能力路径。
无能力参数的构造函数，以及自行创建分配器/资源驱动的回退已从源码删除；
host-spill 仍是明确的可选能力，不是旧推理回退。所属适配器现也只保留能力分支，
不再依赖原生选择宏；目录外历史代码尚未全部清完。

The engine bootstrap has now also resolved its thirteen native/legacy
preprocessor decisions and five catalog-type branches. Public construction
requires an artifact capability/catalog and a runtime factory; no direct
rank-runtime construction remains in this bootstrap. Its internal catalog is
the capability catalog, not a template retaining the old controller-catalog
fallback. The current public path remains development SM89 PP1; removing old
constructors does not implement or qualify the remaining multi-rank/DSpark
plugin paths. Regression guards were added, not executed.

引擎启动层也已清理 13 处原生/旧路径预处理选择和 5 处目录类型分支。公开构建
入口必须传入制品能力/能力目录及运行时工厂，启动层不再自行创建 rank runtime；
内部目录不再通过模板保留旧控制器目录回退。目前公开路径仍为开发级 SM89 PP1，
删除旧构造函数不代表多 rank/DSpark 插件接入或验证完成。新增回归守卫未执行。

The shared runtime artifact manifest, runtime records manifest, tensor ownership
plan and V4 config implementations are also used explicitly by `offline-deepseek`.
That CPU tool does not need to compile the model engine or activate a plugin.
Headers retain their current `include/pih/model` paths pending header migration.

This owner's `host.cmake` assembles the native PP1 source set directly from its
directory and checks its count and path hash; the syntax checker checks the same
migrated selection. The root includes that target definition but no longer scans
legacy model files to subtract optional features. Unused old DSpark/NCCL model
aggregate targets are removed, not substituted with compatibility archives. The
24 native CUDA capability adapters now live in `cuda/`, with their target owned
by `cuda.cmake`. That host-only object target is configurable without CUDA; it
does not contain kernels or link CUDA libraries. The whole model plugin still
requires its GPU backend and Kernel Pack. Other historical DeepSeek model
code, optional DSpark/multi-rank code and qualification oracles remain outside
this directory. This move does not complete those features, capability binding,
the V4.1/B300 model/pack integration, or standalone plugin build configuration.

Use [native inference](../../docs/native-inference.md) for current preparation,
Lock and serving commands. Syntax observations are not proof of a linked CUDA
plugin, numerical correctness, model execution, performance or hardware support.

After migration, non-CUDA CMake configuration passed with the updated 161-file
path closure. The four offline-shared implementations passed Linux-target C++20
syntax compilation without CUDA headers. The combined model/backend/plugin-support
selection also passed: 259 C++ files, using the pinned Linux headers and native
PP1 definitions. No CUDA compilation, complete plugin link or test execution was
performed. The observation is recorded under `out/`, not shipped as qualification.

Following the adapter move, the non-CUDA compilation database contains exactly
the same 24 adapter implementations and native definitions at their new paths.
The refreshed backend syntax selection (51 files including the default entry
and support selection) passed. The Linux CI compile step now names this adapter
object target explicitly; that workflow has not been executed for this change.

本目录收纳原生 PP1 目标实际选入的 161 个模型 C++ 实现，以及原有插件入口、
生成和文本适配层。迁移依据 CMake 编译数据库，旧 `src/model` 路径不保留副本。
四个制品/配置共用实现仍由离线工具显式选取，不要求离线工具编译模型引擎。
模型目标由本目录的 `host.cmake` 直接选择并校验，不再由根构建扫描旧模型目录
后排除可选功能。无消费者的旧 DSpark/NCCL 模型聚合目标已删除；头文件路径、
24 个原生 CUDA 能力适配器现已归入 `cuda/`，由 `cuda.cmake` 定义对象目标，
不依赖 CUDA 工具链即可配置。该目标不包含内核，完整模型插件仍需要 GPU 后端
和 Kernel Pack；头文件及剩余能力接口仍待整理。

其他历史 DeepSeek、DSpark、多 rank 及数值验证实现不在此次迁移范围内。
源码归属迁移不表示这些功能、V4.1/B300 接入或跨插件能力绑定已经完成；
语法编译也不能作为完整链接、模型运行或硬件支持的证明。

迁移后非 CUDA CMake 配置及 161 文件集合校验通过，四个离线共用实现单独通过
无 CUDA 头文件的 C++ 语法编译；包含模型、后端适配器和插件支持的 259 文件
合并检查也已通过。本轮未编译 CUDA、完整链接插件或执行测试。

后续 24 个适配器迁移后，非 CUDA 编译数据库的文件集合和原生宏再次校验通过，
更新后的后端语法选择共 51 个文件也已通过。CI 已显式加入适配对象目标，
但该工作流变更尚未运行。
