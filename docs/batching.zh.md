# 原生多序列批处理

[English](batching.md) | 中文

文本服务升级到 `pih.inference.text.v3` 和 `pih.surface.text-http.v3`。
提供者、消费者和部署 Lock 必须一起重建；SDK 保留 v2 头文件供旧代码使用。
`load`/`close` 仍须独占，多个 `complete` 可并发；输出回调仍在各自调用者线程执行，
响应缓冲区仅在该次调用期间借用，未在旧 ABI 下改变回调语义。

| 模型 | 同一执行计划中的原生计算 | 配置序列上限 | 硬件路径 |
| --- | --- | ---: | --- |
| Qwen3 BF16 | 现有 packed controller 的共享 prefill/decode | 32 | SM89 / SM90，单卡 |
| Qwen3 PIH INT4 | 相同调度器，模型插件拥有 INT4 计算 | 32 | SM89 / SM90，单卡 |
| DeepSeek V4 Flash | 共享 dense/mHC/router；独立 sparse attention、KV、压缩状态 | 已封存 PP1 配置为 4 | SM89 |
| DeepSeek V4.1 | 共享 embedding、FFN/MoE、head；独立 attention、KV、Engram | 32 | SM103，TP 2 / 4 / 8 |

上限是源码准入范围，不是已验证的硬件容量。请求批次与专家批次分别管理。
各模型在共享计划内保留必要的独立注意力通道，不创建请求专属引擎副本，也没有 Python 推理桥接。
V4 旧 `inference.tokens.v1` 入口保持串行；持续合批通过原生文本接口提供。
V4.1 的旧 v1 配置仍用于单请求 supervisor 命令。

## 配置与容量

三个配置项分别限制活动序列数、单计划 token 数和额外排队请求数：
`maximum_batch_sequences`、`maximum_batch_tokens`、`maximum_pending_requests`。
序列数可设为 1、2、4；提示词和输出还必须满足上下文与实际 KV 容量。
容量不足明确拒绝，禁止降级为串行。请求队列满返回 HTTP 429，连接执行槽满返回 503。

Qwen 的 `deploy/native.py serve` / `seal` 和 V4 的 `serve-deepseek` 在原有命令后加：

```bash
--maximum-batch-sequences 4 --maximum-batch-tokens 4096 --maximum-pending-requests 64
```

Qwen 复用原有保守物理资源池；V4 在发布就绪前预留每序列 KV 和独立采样/错误缓冲区。
V4 每计划每序列最多推进一个 token，确保 recent ring 的读写依赖安全。
V4.1 按每 rank 的真实 device/host budget 规划私有缓存区、packed 边界和共享专家工作区。
rank 数与请求批次大小无关。V4.1 的完整 prefill 提示词必须放入单计划
`maximum_batch_tokens`；超出时明确拒绝，不拆分为未支持的 continuation-prefill。

V4.1 文本服务要求 supervisor v2 配置，新增 `batching` 对象包含上述三个字段。
以下命令生成新配置候选文件；审核路径、预算与内容后，以独立可信摘要封存：

```bash
bash deploy/run-v41-native.sh batch-config /absolute/old.json /absolute/new.json 4 4096 64
bash deploy/run-v41-native.sh seal-service /absolute/bundle /absolute/new.json TRUSTED_SHA256
bash deploy/run-v41-native.sh serve /absolute/bundle 8000
```

完整字段见 [SUPERVISOR_CONFIG.md](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md)。
v2 序列数设为 1 即保留 batch=1；重新封存 Lock 后才能启动新版本。

## 调度、并发与退出

每引擎一个执行者负责 GPU 提交、完成事件消费和一次性计划确认。
按请求和 generation 路由事件，允许动态加入、退出和变长输入。
sampling、seed、stop、max_tokens、deadline、取消及输出账本分别维护。
单请求取消只排空自己的在途工作；设备错误停止整个引擎，工作安全退休后才释放资源。

每请求的有界输出缓冲区将回调与调度分开，慢消费者只耗尽自己的输出额度。
服务退出先停止准入，再按退休期限排空。V4.1 在 HTTP 线程启动前创建一个常驻监督子进程，
它独占一组 rank worker 的 cgroup/进程监管；各 rank 看到相同的批次身份、顺序和通信布局。

普通 JSON 与 SSE 均保留。并发 Python HTTP 示例见[英文说明](batching.md#scheduling-and-ownership)。
Python 仅发送 HTTP，不执行模型；每个客户端分别读取自己的 SSE 流。

## 本次验证边界与硬件命令

Windows CPU 验证覆盖真实 packed controller、多序列调度接口、batch=1/2/4、事件路由、
变长提示词、取消、超时、慢消费者和队列拒绝。V4 检查独立设备资源地址、分配计数与释放归零。
V4.1 检查生产 packed 请求账本、计划排列、缓存区隔离和实际 Engram 状态重置。
真实生产 session、collector、commit gate 和输出额度的测试覆盖动态加入与取消，
仅模拟操作系统通道及 tokenizer 边界。这些测试不是 CUDA 数值或性能验收。

2026-10-09 集成验收使用 CMake 4.4.4、Clang 22.1.8 和真实 CUDA 13.2.86 工具，
从当前源码重建四个部署配置：`qwen-4090d`、`qwen-h100`、`deepseek-pp1`、`deepseek-b300`。
全部完成 CUDA 机器码编译、严格未定义符号检查下的生产程序/插件最终链接及全新目录安装。
62 个 ELF/设备产品检查通过，四个配置分别为 16/16/16/14 个。
Qwen 每种架构的 cubin 含 18 个入口，V4/V4.1 Pack 分别含 17/20 个设备镜像。
链接使用真实 Linux CRT、OpenSSL、ICU、CUDA、cuBLAS 和 NCCL 库，没有 stub 或未解析符号豁免。
这证明交叉编译与链接，不代表 Linux 运行通过。保留 Clang 的 CUDA 版本支持警告；原生 NVCC 构建待验收。

本轮 Release CPU 验收 10/10 个可执行测试目标通过，使用真实 GoogleTest 1.17 和 ICU 74.2 库。
Qwen 取消/超时/复用回归另连续运行 10 轮，共 100 个用例通过。
工具测试 125 项及 14 个子测试通过。已安装 SDK 的 26 个头文件审计及外部 C11/C++ 消费者构建、测试通过。
发布包已补齐批处理配置和 HTTP 一致性工具，安装目录内隔离 CLI 检查通过。

本机只有 Intel 显卡，没有 NVIDIA GPU 或 Linux 执行环境。四种模型的真实服务启动、
认证模型推理、batch=1/2/4 数值一致性、实际 GPU 计划成员、多 rank 通信和性能均**未验证**。
不能从构建结果推断硬件容量。

在目标 Linux 主机准备支持的 CUDA 工具链及开发依赖，使用全新的 bundle 目录重建：

```bash
for profile in qwen-4090d qwen-h100 deepseek-pp1 deepseek-b300; do
  python3 deploy/build-release.py "$profile" \
    --build-dir "$PWD/build/acceptance-$profile" \
    --bundle "$PWD/bundles/acceptance-$profile" --jobs 4
done
```

正式顶层 `pih-native-contracts` 干净重建另完成 22 个 Linux x86-64 测试程序链接，
包含 13 个 GoogleTest 程序及 9 个直接测试；这些 Linux 程序未在本机执行。
验收修复了 ICU 依赖作用域/链接、测试 include 路径及两个不可达的 V4.1 CTest 注册。

Linux 安装 GoogleTest、ICU、OpenSSL 开发依赖后执行：

```bash
cmake -S . -B build/batch-contracts -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPIH_ENABLE_CUDA=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=ON \
  -DPIH_BUILD_TOKENIZER_TOOLS=ON -DBUILD_TESTING=ON
cmake --build build/batch-contracts --target pih-native-contracts --parallel 4
ctest --test-dir build/batch-contracts --output-on-failure
```

按上方配置命令，用认证制品及重新封存的 Lock 启动每个重建服务。
对四种模型配置分别执行：

```bash
for batch in 1 2 4; do
  python3 tools/verify_batch_service.py --url http://127.0.0.1:8000 \
    --model Qwen/Qwen3-0.6B --batch-size "$batch" --max-tokens 16 \
    --output "/absolute/new-parity-${batch}.json"
done
```

Qwen 分别启动 BF16 和 INT4；DeepSeek 替换为服务公布的模型 ID：
`deepseek-ai/DeepSeek-V4-Flash-0731` / `deepseek-ai/DeepSeek-V4.1-Flash`。
此工具比较确定性单请求与并发请求的文本、结束原因和 token 数，
明确输出 `execution_plan_verified=false`：HTTP 并发不能单独证明原生合批。
CPU 计划测试证明结构上的合批；实际 GPU 输出与吞吐量仍需硬件记录。
