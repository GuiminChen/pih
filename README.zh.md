# PIH
[English](README.md) | 中文

PIH 是面向低延迟、高吞吐和高效内存使用的高性能、插件优先 LLM 推理框架。

目标架构由 `pih-worker`、部署 Lock 选择的原生插件能力图以及硬件架构专用
Kernel Pack 组成。`pip install .` 只安装 HTTP 客户端 `pih_client` 和主机诊断，
不再构建或发布 `_pih` 单体。历史运行时代码仍待逐项迁移清除，不作为兼容入口。
部署只加载实际需要的平台、存储、后端、执行、内存、模型和服务插件。

## 开发者预览

当前 `v0.1.0-alpha` 是开发者预览版，API、ABI、插件契约和包布局可能发生
不兼容变化。当前没有任何 GPU/模型 Profile 获得生产支持；RTX 4090 D 和
H100 PCIe 的真实模型、数值、服务及性能证据仍未完成。

## 为什么选择 PIH

- 只加载部署 Lock 选择的平台、存储、后端、执行、内存、模型、服务面和
  Kernel Pack；
- 让部署组合显式化，并在启动时拒绝缺失、重复或不兼容的能力；
- 将可复用运行时契约与模型、硬件架构专用内核分离；
- 模型制品始终位于源码树和构建树之外。

## 快速构建

构建不依赖 CUDA、Python Extension、测试或单体运行时的 Phase 1 原生 Worker：

```bash
cmake --preset phase1-native
cmake --build --preset phase1-native
```

在 Linux RTX 4090 D 开发机上，可以使用外部已验证模型 Generation 构建实验性
DeepSeek PP1 Bundle：

```bash
cmake --preset deepseek-4090d-pp1 \
  -DPIH_DEEPSEEK_4090D_ARTIFACT_DIRECTORY=/absolute/model/generation \
  -DPIH_DEEPSEEK_4090D_ARTIFACT_ROOT_SHA256=<64-lowercase-hex>
cmake --build --preset deepseek-4090d-pp1
cmake --build out/build/deepseek-4090d-pp1 \
  --target pih_deepseek_4090d_pp1_smoke
```

该 Smoke 不会把模型字节复制进构建树，并始终报告
`hardware_evidence_open`。成功组装不代表该部署已经获得支持。

## CI 与目标机预检

`cpu-contract` Workflow 构建 Linux CPU-only C++/Python 制品并运行契约测试。
它有意关闭 CUDA 和 NCCL；CPU CI 通过不能证明 GPU、模型、性能或受支持
Profile。

Linux 运维方可以在复制部署制品前运行只读目标机预检：

```bash
pih-preflight-target-host --hardware-profile rtx4090d --devices 0
# 或检查 H100 PCIe：
pih-preflight-target-host --hardware-profile h100-pcie --devices 0,1
# 重放已保存的 JSON 收据：
pih-preflight-target-host --verify-receipt target-preflight.json
```

预检只确认目标 SKU、可见设备和主机内存等前置条件，仍保持
`hardware_evidence_open`。模型加载、CUDA/NCCL 执行、容量、数值正确性和发布
支持必须通过精确目标资格测试。

## 文档

Qwen 原生插件服务的构建、按 GPU 选择启动方式、HTTP 示例及效果观察步骤，见
[原生推理操作指南](docs/native-inference.zh.md)。该代码已完成 Linux/CUDA 交叉编译与链接，硬件运行未验证；
不通过旧 Python 兼容桥接运行。

- [文档索引](docs/index.zh.md)
- [系统架构](docs/architecture.zh.md)
- [开发指南](docs/development.zh.md)
- [插件开发](docs/plugins.zh.md)
- [部署指南](docs/deployment.zh.md)
- [支持策略](docs/support.zh.md)

参与贡献前请阅读 [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md)。安全问题请按照
[SECURITY.md](SECURITY.md) 私下报告。第三方依赖与参考项目记录在
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 许可证

[Apache License 2.0](LICENSE)
