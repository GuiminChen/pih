# 文档
[English](index.md) | 中文

先阅读[开发指南](development.zh.md)构建原生 Worker，再按[插件指南](plugins.zh.md)使用安装后的 SDK 构建外部插件。

- [架构](architecture.zh.md)
- [部署与验证](deployment.zh.md)
- [原生插件一键推理与结果观察](native-inference.zh.md)
- [Python HTTP 客户端](client.zh.md)
- [支持矩阵](support.zh.md)
- [发布清单](release.zh.md)
- [DeepSeek V4.1 Flash / B300 接入](deepseek-v41-b300.zh.md)

当前预览验证原生能力组合，尚无通过资格验证的模型/GPU Profile，也没有通用的 `pih serve` 命令。Python wheel 只包含 HTTP 客户端与主机诊断，并非原生插件 Bundle，也不包含旧推理兼容层。
