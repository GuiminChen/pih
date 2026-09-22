# 原生部署
[English](deployment.md) | 中文

四种配置具备原生代码/构建交付路径。**硬件运行未验证**：下列启动服务和处理模型制品的
命令是给操作者的说明，本次交付没有执行这些操作。构建、安装和打包不激活 GPU。

## 构建机与目标环境

使用 Linux x86-64、Bash、编排用 Python 3.11+、CMake 3.26+、Ninja、C++20 编译器、
OpenSSL 3/ICU 开发文件、支持准确 SM 的 CUDA Toolkit 13.2+。
B300 另需 NCCL 2.31.2 头文件和真实匹配动态库。通过 CMake 设置提供依赖位置；脚本不安装
驱动/系统包，也不下载权重。Qwen 对应 SM89/SM90，PP1 对应 SM89，B300 必须为 `103-real`。
公共 cubin 规则区分 NVIDIA 与 Clang 参数；Clang 还需 Toolkit 的 cuRAND 头文件，
独立头目录通过 `PIH_CUDA_DEVICE_INCLUDE_DIRS` 提供。

交叉编译必须提供真实目标 CRT 和运行库。本地制品需要 glibc 2.39 时代的运行环境、
具有 GLIBCXX_3.4.32 的 libstdc++、ICU74、libcrypto.so.3、libcudart.so.13、libcuda.so.1，
按模型还需要 libcublasLt.so.13 或 libnccl.so.2。精确递归依赖以安装 ELF 的 DT_NEEDED
与符号版本表为准。不使用 stub 库，不将构建机 compatibility driver 当成目标驱动安装。
发行归档不附带这些第三方运行库或模型权重。

Worker 拒绝动态加载注入环境变量。通过审查过的系统加载器配置提供运行库，勿使用
`LD_PRELOAD` 或不受控的 `LD_LIBRARY_PATH`。构建机不激活 CUDA/NCCL provider。
认证插件依赖可执行 `memfd_create`（`MFD_EXEC`）、file seals 和可读 `/proc/self/fd`，
策略拒绝时直接失败。B300 运行还要求 supervisor schema 中规定的 cgroup-v2 委派父目录、
clone3、pidfd、execveat 权限。

## 无权重、无 GPU 执行的构建打包

选择 `qwen-4090d`、`qwen-h100`、`deepseek-pp1`、`deepseek-b300` 之一；
每种配置使用独立构建目录和全新安装目录：

```bash
python3 deploy/build-release.py qwen-4090d   --build-dir /absolute/build-qwen89 --bundle /absolute/new-qwen89 --jobs 4
python3 tools/package_release.py native --root /absolute/new-qwen89   --output /absolute/new-qwen89.tar.gz
```

其他配置替换名称和目录。交叉构建追加
`--cmake-arg=-DCMAKE_TOOLCHAIN_FILE=/absolute/toolchain.cmake`；
重复 `--cmake-arg=-DNAME=value` 提供真实依赖，不能覆盖封闭配置的模型选择。
脚本构建 `pih-release`，仅安装同名组件。`NATIVE-INSTALL.json` 记录精确安装成员；
打包拒绝缺失、变化或额外文件，生成并逐成员回读验证 `RELEASE-MANIFEST.json`。
构建打包不会制造模型可信 Lock。

解压到当前用户控制的新目录，保持 bin/lib 和 Qwen 的 `lib/qwen3-smXX/sm_XX`
cubin/manifest 布局。不要混用不同版本 Worker、SDK、模型和 Pack。
当前 Pack ABI 要求 `bind_origin`，旧表被拒绝；更改二进制后重建并重新 seal。
模型 DSO 编入实际生成的 Qwen cubin 摘要，向 CUDA 提交前核对已持有的镜像快照；
同时替换镜像和 sidecar 仍会失败。

## Qwen 4090 D/H100 启动

使用固定 Qwen3-0.6B revision 和独立可信的 artifact/config SHA256。
BF16 使用 `model.safetensors`，INT4 使用原生 `model.xing-int4`，不接受 GGUF/AWQ/GPTQ，
两者都需要同 revision 的 config/tokenizer。显式使用附带的
`pih-qwen-source-verify`、`pih-qwen-int4-convert`、`pih-qwen-int4-verify` 准备制品；
serve 不隐式转换。权重必须放在源码/构建/bundle 之外。
详细输入见[制品与请求指南](native-inference.zh.md)。

```bash
bash deploy/run-native.sh serve --target rtx4090d --precision bf16   --bundle /absolute/new-qwen89 --build-dir /absolute/build-qwen89   --model-dir /srv/models/Qwen3-0.6B   --artifact-sha256 "$TRUSTED_ARTIFACT_SHA256"   --config-sha256 "$TRUSTED_CONFIG_SHA256" --max-context 4096 --port 8000
```

H100 PCIe 使用 `h100-pcie` 与 SM90 bundle。只有从源码使用时才追加 `--build`。
启动器校验 GPU 身份并 seal 二进制/制品身份，最后 exec 原生 Worker，
不会把任意本地文件的自算摘要当成可信模型来源。

## DeepSeek V4-0731 PP1 启动

先使用原生 prepare/verify/generation-store 工具及固定 semantic snapshot。
原始 HF checkpoint 不是 runtime generation。生产 PP1 是单 rank、逐 token 计划，
未交付多 token 并行 prefill，没有恢复旧跨进程 PP 路径。

```bash
bash deploy/run-native.sh serve-deepseek   --bundle /absolute/new-pp1 --build-dir /absolute/build-pp1   --artifact-dir /srv/models/deepseek-v4-0731-pp1 --artifact-root "$ARTIFACT_ROOT"   --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731   --max-context 4096 --port 8000
```

启动器校验并接入外部制品，启动原生 text/SSE surface。缺失或不匹配即失败，不选择
reference 运行时。请求字段和 semantic 准入见[制品与请求指南](native-inference.zh.md)。

## DeepSeek V4.1 B300 启动

先由原生 converter 生成 canonical TP1，再 reshard 成 2/4/8 rank 布局。
TP1 是离线格式，不代表支持单 rank 生产 backbone；不声明支持视觉、MTP 或任意 checkpoint。
预算、placement、tokenizer/map 和逐 rank 摘要遵守
[supervisor schema](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md)。

```bash
bash deploy/run-native.sh v41 seal-rank /absolute/new-b300
# 审查 supervisor.json：worker/helper/rank-lock 摘要、外部制品、委派 cgroup 父目录；获取可信摘要。
bash deploy/run-native.sh v41 seal-service /absolute/new-b300   /absolute/supervisor.json "$TRUSTED_CONFIG_SHA256"
bash deploy/run-native.sh v41 serve /absolute/new-b300 8000
```

已有 Lock 不覆盖。源码的一键服务构建入口是
`bash deploy/run-native.sh v41 build-service BUILD NEW_BUNDLE JOBS`（目录必须绝对路径），
随后按上述顺序 seal。rank Worker、transport/NCCL helper、supervisor 和 SM103 Pack
都是原生实现；构建成功不等于通信或数值验证。

## 运行与故障处理

服务绑定 loopback，串行处理准入文本请求。对外暴露时自行配置认证反向代理；不声明内置
TLS/auth。Qwen/PP1 配置 `--generation-timeout-ms`，B300 使用 supervisor 限额。
SIGTERM、Ctrl-C、断连进入原生取消/退役流程；CUDA/NCCL pending 不是完成证明。
无法证明退役时 fail-stop，不伪报成功或切换运行时。保留 stderr、退出码、Lock 和部署收据，
升级使用新 bundle。`probe` 会实际推理，明确不属于本次无硬件验证范围。
