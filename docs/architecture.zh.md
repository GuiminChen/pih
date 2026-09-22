# 架构
[English](architecture.md) | 中文

原生 Worker 按 sealed Lock 选择 C ABI 能力图；模型插件拥有模型执行代码，通用 provider 拥有资源和传输能力。Kernel Pack 分离设备架构实现。启动校验 ABI、能力 provider、身份、生命周期及安装位置；失败时不回退到旧运行时。

SDK 位于 `include/pih/plugin_sdk`，安装包提供 header-only `PIH::plugin_sdk`。所有预览组件必须使用同一 revision。
构建证据见[集成状态](integration-status.zh.md)，资格边界见[支持说明](support.zh.md)。历史架构记录保留在工作区，不进入默认发行。
