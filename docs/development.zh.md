# 开发指南
[English](development.md) | 中文

## 独立原生契约编译

旧 `pih_core`/`PIH::core`、`pih_cuda`/`PIH::cuda` 聚合目标、内部头文件整体安装
路径及单体测试入口已删除。启用 `PIH_BUILD_MONOLITH=ON`、`PIH_BUILD_TESTS=ON`
或旧 `PIH_ENABLE_NCCL=ON` 会明确配置失败。请选择原生插件和
`PIH_BUILD_NATIVE_CONTRACT_TESTS`；V4.1 独立 NCCL 开发目标使用其文档中的另一
开关。CPU 契约 CI 已改为构建原生契约目标，不再构建旧 C++ 聚合集。
历史 C++ 测试源码仍需迁移，较小的原生套件不代表已覆盖全部旧契约，也不代表
跨插件能力绑定完成；不会生成兼容聚合库。

按下文准备固定 Linux 头文件后，可运行
`python3 tools/check_native_syntax.py --compiler clang++ --plugin-support`，
把迁移后的执行、存储、host-spill 和 Linux 平台实现纳入语法编译。该选项在默认
集合中加入各目录当前顶层 `.cpp` 文件（含入口），报告名以
`-plugin-support.json` 结尾并记录源码哈希；不代表完整头文件闭包、链接、能力
绑定或运行正确性。CI 语法任务也已加入该选项，但配置更新不等于 CI 已运行。

HTTP 请求解析、Qwen 停止字符串和 V4.1 输出租约契约测试可独立于历史单体测试集编译。在 Linux 安装 `libgtest-dev`、
CMake、Ninja、C++20 编译器和 OpenSSL 3 开发文件后执行：

```bash
cmake -S . -B out/build/native-contracts -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPIH_DEPLOYMENT_PROFILE=custom \
  -DPIH_BUILD_WORKER=OFF -DPIH_BUILD_PLUGINS=OFF \
  -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF \
  -DPIH_ENABLE_CUDA=OFF -DPIH_ENABLE_NCCL=OFF \
  -DPIH_BUILD_TESTS=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=ON
cmake --build out/build/native-contracts --target pih-native-contracts --parallel 2
```

测试发现延迟到 CTest 执行阶段，构建不会执行测试或加载模型。目标仅链接 GTest 与
请求解析实现、Qwen 停止过滤及有界 JSON/status 基础代码，或 V4.1 发布队列及输出额度/status
基础代码。输出用例覆盖预留、发布、消费者释放、回收后丢弃、过期/外部租约、非法输出和
空可见内容；不证明 rank 已回收，丢弃执行中输出前仍须调用方确认所有 rank 已退出。
不构成完整服务、socket、GPU 或模型资格验证。发行集成环境已使用生产实现依赖完成
原生契约目标的编译和最终链接，但没有执行生成的 Linux ELF 测试。远程只编译 CI
工作流已配置但未运行。后续在 Linux 执行测试时，使用
`ctest --test-dir out/build/native-contracts --output-on-failure`。

## 开发 Worker

使用 Linux、CMake 3.26+、Ninja、支持 C++20 的编译器及 OpenSSL 3 开发头文件。Ubuntu 24.04 示例：

```bash
sudo apt-get update
sudo apt-get install --yes cmake ninja-build g++ libssl-dev
cmake --preset phase1-native
cmake --build --preset phase1-native --parallel 2
```

构建目标会运行最小插件的生命周期 smoke，无需模型、CUDA Toolkit 或 Python 扩展。它不验证文本生成。

已安装 Docker 时也可以构建 CPU 预览容器：

```bash
docker build -f deploy/Dockerfile.preview -t pih-preview .
docker run --rm --network none --read-only --cap-drop ALL pih-preview
```

镜像构建时会编译 Worker 和外部 SDK 示例，运行契约及加载检查；运行时以非 root
用户执行最小插件。smoke 完成后进程退出，不会启动 HTTP 服务。

旧 `_pih` 扩展绑定、运行时包、构建及安装入口已从公开源码树移除；`PIH_BUILD_PYTHON=ON`
现在会明确失败，不再提示启用旧单体。对应的扩展注入测试工具和专属测试也已删除。
[CPU CI](../.github/workflows/cpu-contract.yml) 保留 C++ 合约及纯客户端 wheel 检查，不再生成
扩展或运行旧 Python 套件。C++ 合约使用 `PIH_DEPLOYMENT_PROFILE=custom`，这不是生产兼容入口。
移除旧测试不等于 GPU 或模型资格验证通过。

离线 DeepSeek M5 证据处理已移到 `tools/evidence/deepseek_optimization_evidence.py`，相关工具与测试直接引用该工具包，不再经 `pih` 初始化或加载 `_pih`。旧模块位置不提供转发别名。该模块处理原始观测与证据，不是推理实现，也不进入 `pih_client` wheel；迁移本身没有执行观测采集或模型测试。其余旧包依赖仍待清理。

Qwen teacher-forced 质量编译器也已迁到 `tools/evidence/qwen_teacher_forced_quality.py`，旧包不再导出它，也不保留转发模块。在源码仓库运行 `python tools/compile_qwen_teacher_forced_quality.py --input INPUT.json --output OUTPUT.json` 可处理已有观测，不执行模型或采集观测。证据格式标识保持不变。尚未迁出的收据及部署质量角色模块已显式引用新工具模块，其余依赖仍待迁移。

`python -m pip wheel . --wheel-dir dist` 现在构建 Python 3.11+ 纯客户端 wheel，
只包含 `pih_client`，不构建 CMake/pybind11/CUDA/`_pih`。原有
`from pih import Engine` 必须迁移到 HTTP 客户端，不提供兼容别名。
原生插件单独构建；当前不承诺已发布预构建 wheel。

找不到 CMake 时安装上面的依赖；找不到 OpenSSL 时安装 `libssl-dev`。缓存选错 Profile 时使用新的构建目录。SDK 使用方式见[插件指南](plugins.zh.md)。

## 原生 Qwen 分词器参考比较

后续在 Linux 验证时，比较工具调用模型插件共用的原生分词实现，不再调用旧
Python 分词器：

```bash
cmake -S . -B out/build/tokenizer -G Ninja -DPIH_DEPLOYMENT_PROFILE=custom -DPIH_BUILD_WORKER=OFF -DPIH_BUILD_PLUGINS=OFF -DPIH_BUILD_MONOLITH=OFF -DPIH_BUILD_PYTHON=OFF -DPIH_BUILD_TESTS=OFF -DPIH_ENABLE_CUDA=OFF -DPIH_BUILD_TOKENIZER_TOOLS=ON
cmake --build out/build/tokenizer --target pih-tokenize --parallel 2
python tools/verify_qwen_tokenizer_reference.py --native-tokenizer out/build/tokenizer/plugins/common/pih-tokenize < /models/Qwen3/tokenizer.json
```

可选比较工具需要 Python `tokenizers` 参考库；原生构建还需要 ICU 开发库。
工具只接受脚本固定 SHA-256 的分词制品，将其复制到私有临时目录，逐一比较
8 条编码样例。每条默认超时 60 秒，可通过 `--timeout-seconds` 调整。
只能指定可信本地可执行文件。报告记录运行前后文件 SHA-256，不证明可执行文件
及动态库身份闭包；子进程日志写临时文件，不提供磁盘配额沙箱。
报告不再包含旧 Python `execution_root`。样例通过也不证明特殊 token、解码、
对话渲染、模型生成或 GPU 支持通过验收。本次迁移未执行该命令；其余旧分词器
消费者仍需清理。

## 原生 Qwen 纯文本对话参考比较

服务与 `pih-qwen-format` 现在共用 `plugins/model-qwen3/chat_format.h`。
分段计划保留独立控制 token，但在 BPE 前合并相邻普通文本，不再任意分割角色、
换行和正文。正文使用普通文本编码，特殊 token 字面量不会成为模板控制标记；
只有固定助手前缀识别这些标记。渲染字节上限为 2 MiB。
CPU 工具接受只含 `messages` 的 JSON 文件，返回 JSON `text` 字段，
不分词、不加载权重、不执行推理。

使用上面配置的 CPU 构建目录，后续验证命令为：

```bash
cmake --build out/build/tokenizer --target pih-qwen-format --parallel 2
python tools/verify_qwen_chat_renderer_reference.py --native-renderer out/build/tokenizer/plugins/common/pih-qwen-format < /models/Qwen3/tokenizer_config.json
python tools/verify_qwen_chat_renderer_reference.py --native-renderer out/build/tokenizer/plugins/common/pih-qwen-format --tokenizer /models/Qwen3/tokenizer.json < /models/Qwen3/tokenizer_config.json
```

比较工具需要可选的 Python `jinja2` 参考依赖及固定哈希的对话模板，不再导入
旧 Python 渲染器。它以 `enable_thinking=False` 检查 5 条纯文本样例，包含前导
换行及纯空白。可选 `--tokenizer` 需要 `tokenizers` 参考库和固定哈希的分词
制品，还会通过服务共用的组装函数比较 token ID。对应的原生工具模式为
`pih-qwen-format REQUEST.json --tokens TOKENIZER.json`，此模式现在与服务使用
相同 SHA-256 固定分词器字节，但不认证完整模型或文件系统。工具调用、推理
历史、thinking 模式、正文控制 token 字面量及
生成仍不在此比较范围内，
不能据此宣称已实现，也不能用此样例集替代旧 Python 渲染器更广的测试。
只使用可信可执行文件；临时日志和可执行文件哈希的限制与分词比较工具相同。
每条默认超时 30 秒。本次迁移只做语法及导入检查，未运行该比较或 GPU 测试。

## 从 Windows 执行 Linux 语法检查

没有 Linux 执行环境但已有 Clang 时，可运行以下可选命令。只检查 Linux 目标的
C++20 语法，不链接、不运行程序或模型：

```text
python tools/prepare_linux_syntax_headers.py
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS
```

准备脚本从 Ubuntu 官方归档下载五个固定开发包（含 Linux OpenSSL 头文件），按
`tools/linux-syntax-sysroot.lock.json` 校验长度和 SHA-256，只将头文件提取到新建
的 `out/linux-syntax-sysroot`。需要支持 ar/zstd 的 bsdtar（Windows 系统 `tar`
可用，其他环境可传 `--tar bsdtar`）。不安装系统软件、不执行安装脚本、不提取
运行时二进制，也不涉及 CUDA 或模型文件。已有目标目录不会被覆盖，下载暂存目录
保留在 `out` 便于检查。此包锁仅用于开发期语法检查，不是生产系统版本或安全更新
策略。

包锁更新后请创建新的 sysroot，不要修改旧回执。比如准备时传
`--output out/linux-syntax-sysroot-openssl`，检查时传
`--sysroot out/linux-syntax-sysroot-openssl`。旧回执会按设计拒绝通过新包锁检查。
仅有头文件不代表已提供链接库。

Qwen cuBLAS Lt 适配器还需运行 `python tools/prepare_cublas_syntax_headers.py`，
从固定 NVIDIA 开发包只读取允许列表中的 6 个普通头文件和许可证，写入新目录
`out/cublas-syntax-headers-13.4.1.3` 并记录包及文件摘要。直接语法编译
`src/backend/cuda/gemm_plan.cpp` 时，在 CUDA 头文件目录之外添加
`-I out/cublas-syntax-headers-13.4.1.3/include`。库桩、链接、包内测试脚本均不
提取或执行。这不是 cuBLAS 安装；实际 CUDA 构建仍需要完整工具链和链接库。
该头文件包对应 CUDA 13.2 版本范围。

检查器覆盖其明确列出的默认源文件集合：共用文本工具、Worker、HTTP Surface、
DeepSeek 生成和模型入口、Qwen 入口及转换器、调度器、基础类型、CUDA rank
运行时/启动器及仅依赖能力接口的共享专家适配器。使用原生 PP1
编译宏，生成 `out/native-syntax-report.json`。这不是 CMake 完整构建、全项目
检查、链接验证、CUDA kernel 编译、运行测试或发布资格验证；成功不能替代这些
验收项。

需要同时检查 CMake 为原生 DeepSeek PP1 目标选出的 161 个 C++ 模型源文件时，
在纯插件构建目录导出编译数据库：

```text
cmake -S . -B out/build/preview-local -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-model-build out/build/preview-local
```

此模式使用固定的 Linux 头文件环境检查合并后的编译单元。执行前校验预期源码
列表哈希及 PP1 宏定义，不是逐条重放 CMake 编译命令，也不包含 CUDA 源文件。
结果与完整诊断分别保存为 `out/native-syntax-closure-report.json` 和相邻的
`.log` 文件；任何批次失败都会使命令失败。前述链接、CUDA、运行和发布验证的
限制仍然适用。

添加 `--deepseek-backend` 可同时检查原生 PP1 CUDA 后端中全部 24 个仅依赖能力
接口的 C++ 适配文件。选择规则排除 DSpark、NCCL、直接探测物理设备的实现和
Kernel Pack 契约源文件，并核对 CMake 固定数量/摘要；不需要 CUDA 头文件，
也不编译 `.cu` 文件。

```text
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-backend
python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --deepseek-model-build out/build/preview-local --deepseek-backend
```

报告记录实际选取、去重后的编译单元；前者写入
`out/native-backend-syntax-report.json`，合并模式写入前述 closure 报告。
两者都只是语法检查，不是 CUDA 构建或运行测试。
`native-preview.yml` 的 `native-syntax` 作业已配置上述合并检查，不执行模型。
提交工作流配置不代表 CI 已运行或通过。

native-preview CI 的 tokenizer 任务还会用 Linux 编译器编译实际 PP1 模型
对象、模型自有的 CUDA 能力适配对象目标及共用模型支持库。这些适配器不需要
CUDA 头文件或启用 CUDA 工具链。该任务不链接 CUDA 模型插件、不执行推理；提交 CI 配置
不等于 CI 已运行通过。

显式离线工具、制品和 supervisor 源码集合及转换命令变体可用
`python tools/check_native_syntax.py --compiler PATH_TO_CLANG_PLUS_PLUS --artifact-tools`
检查，输出 `out/native-artifact-syntax-report.json` 及完整诊断日志。该步骤不链接
工具；Linux `artifact-tool-build` CI 任务配置为通过
`PIH_BUILD_ARTIFACT_TOOLS=ON` 构建实际可执行文件。
