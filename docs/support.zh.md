# 支持边界
[English](support.md) | 中文

新增多序列修订的源码、CPU 执行、CUDA 编译、生产链接及待执行硬件验收，
见[批处理说明](batching.zh.md)。

首版代码与构建验收覆盖 Qwen 4090 D/H100、DeepSeek V4-0731 PP1 和 DeepSeek V4.1 B300。
**硬件运行未验证**；不声明生产硬件资格、数值正确性、容量或性能。
客户端仅 HTTP；Windows/macOS 不提供本地推理。开发期 reference 运行时不进入公开仓库和默认发行。
预览 ABI 需要全部组件使用同一 revision，旧表不兼容。
远程 CI 未执行。具体构建证据见[集成状态](integration-status.zh.md)，交付范围见[发行说明](release.zh.md)。
