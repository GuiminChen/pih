# V4.1 canonical 权重重分片

[English](RESHARD.md) | 中文

此命令只接收**已转换为 canonical 格式并通过可信摘要校验的 TP1 文本主干权重**。
它不是原始 Hugging Face checkpoint 转换器，不会生成该前置制品，也不表示
V4.1/B300 已获得运行资格。独立的[源转换命令](SOURCE_EXPECTATIONS.md)可处理已实现的
输入类型并生成 canonical TP1 制品；其他源格式仍不支持。

## 构建

环境为 Linux x86-64、C++20 编译器、CMake、Ninja 及仓库原生 CPU 构建依赖
（包括 OpenSSL 开发头文件）。不需要 CUDA 或 Python 推理运行时。在仓库根目录执行：

```bash
bash deploy/run-v41-native.sh build-weights /absolute/build/v41-weights 4
```

请使用独立于 GPU 构建的目录。此步骤只构建 `pih-v41-reshard`，不执行重分片、
下载或测试。

## 参数与执行

从可信部署记录设置以下变量，不要直接对不可信输入计算摘要后将其当成可信来源：

- `CONFIG`：V4.1 HF 配置 JSON 的绝对路径。
- `CONFIG_SHA256`：配置原始字节的可信、非零、小写 SHA-256。
- `SOURCE`：canonical TP1 权重目录的绝对路径，包含 `weights.manifest.json`
  及其全部分片；清单必须绑定相同配置、world size 1、rank 0。
- `SOURCE_MANIFEST_SHA256`：上述清单的独立可信摘要。
- `WORLD`、`RANK`：输出并行度 1/2/4/8，以及从零开始且小于 WORLD 的 rank。
- `DEVICE_BUDGET_BYTES`：该 rank 对齐后权重占用上限，正整数。它不包含缓存、
  临时缓冲和其他运行开销，不代表整模型显存足够。
- `OUTPUT`：新的输出目录绝对路径。父目录必须已存在、属于当前有效用户，且
  组和其他用户不可写；`OUTPUT` 和 `OUTPUT.staging` 都必须不存在。

路径不得经过符号链接。配置、源清单和源分片必须是只读、单硬链接的普通文件。
整个过程必须保持源数据不变，并排除同用户或 root 的并发修改；这些检查不是
恶意主机隔离机制。

```bash
bash deploy/run-v41-native.sh reshard /absolute/build/v41-weights \
  "$CONFIG" "$CONFIG_SHA256" "$SOURCE" "$SOURCE_MANIFEST_SHA256" \
  "$WORLD" "$RANK" "$DEVICE_BUDGET_BYTES" "$OUTPUT"
```

命令先读取并校验完整源制品，再使用 1 MiB 数据缓冲复制目标 rank，完整读回
输出分片并生成运行端清单。元数据另占内存。磁盘必须容纳整个输出 rank；不提前
预留磁盘空间。Engram 表很大，即使不运行 GPU，这仍是耗时的 CPU/I/O 操作。
每次 rank 调用都会重新校验完整源，不会共享上一次校验结果。

## 发布、回执与失败处理

若要同时组装 Worker 必需的配置和映射，在命令末尾追加
`--runtime-map /absolute/compressed-token-map.bin "$TRUSTED_MAP_SHA256"`。
程序先校验映射，再在发布前写入同一 staging 的 `hf_config.json` 和
`compressed-token-map.bin`；回执增加映射摘要及元数据字节数。未提供此参数时，
输出仍是权重制品，不应当作可直接启动的 Worker 目录。

文件独占创建在 `OUTPUT.staging`，同步并设为 0400。完成后目录设为 0500，
通过 Linux `renameat2(RENAME_NOREPLACE)` 改名为 `OUTPUT`，再同步父目录。
文件系统或内核不支持该操作时直接失败，不回退到可覆盖目标的重命名。

- 退出码 0 且收到 JSON 回执：记录配置、源清单和输出清单摘要、world/rank、
  41 个权重分片及包含清单的总字节数。保存回执，并按部署准入规则决定是否将
  输出清单摘要写入运行配置；回执不是数值等价证明。
- 退出码 2：没有成功发布回执，先检查保留的 staging。
- 退出码 3：目录已经改名，但之后同步或回执输出失败；先检查 OUTPUT，不能盲目重跑。

进程中断也可能留下 staging，或者已改名但没有回执的输出。仅有清单文件不能
证明成功。命令不会自动删除、覆盖、续传或激活模型；清理和恢复须先人工检查。
原子发布仅针对单个 rank 目录，不是整个多 rank 部署事务。

当前仅通过 Linux 目标对象编译和 Bash 语法检查；尚未完成链接、实际重分片或模型测试。
