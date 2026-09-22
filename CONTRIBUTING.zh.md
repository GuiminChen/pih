# 参与 PIH 开发

[English](CONTRIBUTING.md) | 中文

PIH 是早期阶段的插件优先 LLM 推理框架。欢迎提交能够保持能力契约明确、部署
Bundle 精简的改进。

## 提交前

1. 涉及公共 API、ABI、插件契约或整体架构的较大变化，请先创建 Issue 讨论；
2. 不要提交模型权重、生成 Bundle、性能输出、证据收据或凭据；
3. 未满足对应契约中的验证和晋级条件前，不得声称某个 GPU/模型 Profile 已获
   得支持。

## 开发流程

创建独立分支并提交范围明确的 Pull Request，说明受影响的契约和兼容性影响。
依赖较少的原生开发路径为：

```bash
cmake --preset phase1-native
cmake --build --preset phase1-native
```

Linux CPU Contract Workflow 是 Pull Request 的可移植基线。只有变更声明或影响
CUDA、模型或硬件行为时，才需要相应的专项验证。

提交贡献即表示你同意按照本仓库采用的 Apache License 2.0 授权该贡献。
