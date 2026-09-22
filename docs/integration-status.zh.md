# 原生集成状态
[English](integration-status.md) | 中文

本次交付区分代码/构建验收与硬件资格验证。生产矩阵覆盖 Qwen3-0.6B
RTX 4090 D（SM89）/H100 PCIe（SM90）、DeepSeek V4-0731 RTX 4090 D PP1
（SM89）、DeepSeek V4.1 B300 文本 backbone（SM103、2/4/8 ranks）。
全部使用原生 Worker、C ABI 能力和原生 Kernel Pack；不会选择 Python 模型回退。

| 路径 | 代码与构建证据 | 运行状态 |
| --- | --- | --- |
| Qwen 4090 D/H100 | 生产 CMake 模型、原生工具、Pack 最终链接；SM89/SM90 各含18个入口的独立 cubin | 硬件运行未验证 |
| DeepSeek V4-0731 PP1 | 生产模型、三个制品工具和 SM89 Pack 最终链接；17个内嵌 cubin | 硬件运行未验证 |
| DeepSeek V4.1 B300 | 模型、rank Worker、NCCL helper/provider、supervisor、离线工具和 SM103 Pack 最终链接；20个内嵌 cubin | 硬件运行未验证 |
| 共享运行时 | Worker/CLI、能力插件、严格链接和明确的生产安装清单 | 本机未执行 Linux 生命周期检查 |

本地设备编译使用 Clang 22.1.8 与真实 CUDA 13.2.86 libdevice、ptxas、fatbinary，
生成设备机器码，不把 host-only 或 PTX 检查当作完成。保留 Clang 对 CUDA 13.2
超出其部分支持12.9范围的警告。本机没有执行原生 NVCC 路线或远程 CI。
Linux 交叉链接使用真实 CRT、标准库、OpenSSL、ICU、CUDA、cuBLAS、NCCL，
没有伪符号、stub 库或未解析符号豁免。

认证动态库通过密封可执行 memfd 装载。加载器在发布 Pack 契约前显式绑定原始安装位置，
解决 Qwen 外置 cubin 在密封装载后的定位问题。预览 Pack ABI 新增必需的 `bind_origin`
回调；必须同版本重建 SDK、Worker、rank Worker 和所有 Pack。旧的小结构被拒绝，
没有兼容适配。重建后必须刷新插件/Pack 摘要与部署 Lock。

GPU、真实权重、模型推理、数值、性能和多卡实机均为**硬件运行未验证**，不属于本次
代码/构建完成门槛。构建收据不代表硬件支持、容量或数值正确性；CI 配置不代表远程运行通过。

操作见[部署指南](deployment.zh.md)和[发行边界](release.zh.md)。
