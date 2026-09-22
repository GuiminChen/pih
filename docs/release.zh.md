# 发行内容与验收
[English](release.md) | 中文

代码/构建验收要求四组生产配置完成编译、最终链接、正确架构的 CUDA 设备镜像生成和安装，
原生能力与部署链无已知实质缺口。**硬件运行未验证**不阻塞本次交付。
远程 CI 未执行，不能将配置完成写成运行通过。

## 三种独立分发

1. 原生包：选择封闭模型配置，构建 `pih-release`，只将 `pih-release` 组件安装到
   **新目录**。正向目标清单为 `cmake/PIHRelease.cmake`，包含所选命令、能力插件、模型、
   Pack、Qwen 外置 cubin/manifest、部署脚本和许可证。不含权重、Python 模型运行时、
   资格验证程序或 reference 后端。CUDA/ICU/NCCL/系统运行库由目标环境提供，不从本机拷贝。
2. 原生源码包：以 `release/source-policy.json` 正向清单为准。
   `tools/package_release.py source --output /absolute/new-source.tar.gz` 生成精确成员、
   字节数和 SHA256 的 `RELEASE-MANIFEST.json`，扫描凭据模式与本机用户路径，并逐成员回读校验。
   测试和最小 SDK 示例是开发源码，不作为生产模型程序安装。历史设计、实现与资格验证记录
   不进入本源码包。
3. Python wheel **及 sdist**：仅含 `pih_client` HTTP 客户端和主机诊断。
   `MANIFEST.in` 与 setuptools 显式包清单排除旧 `python/pih`、`_pih`、reference
   和原生构建目录；客户端不导入模型实现。

开发期使用的 DeepSeek V4.1 Python/TileLang reference 包不进入公开仓库及上述三种发行，
原生部署也没有自动回退路径。旧 Python 运行时与绑定源码、内部构建证据、交付备份、
`.git`、`out`、权重和生成二进制均排除于公开仓库和原生源码包。

```bash
python3 deploy/build-release.py qwen-4090d --build-dir /absolute/build-qwen --bundle /absolute/new-qwen
python3 tools/package_release.py native --root /absolute/new-qwen --output /absolute/new-qwen.tar.gz
python3 tools/package_release.py source --output /absolute/new-source.tar.gz
python3 -m build --wheel --sdist
```

每份原生包包含项目 Apache-2.0 许可证、第三方通知和 DeepSeek MIT 通知。
源码包保留改编 encoding 的原始通知；不再分发权重、tokenizer 或第三方运行库。
模式扫描有明确范围，不声称可以自动发现所有凭据或许可问题。

`native-release-build` 工作流由维护者手动触发，在已配置 CUDA13.2/NCCL 的 Linux
构建机完成编译/链接/打包，不执行 GPU 程序。CPU 契约工作流独立运行。
本次交付不包含自动发布、Git 提交或 push。

Docker preview 是独立 CPU/最小插件开发镜像，不是模型发行包。最终阶段仅拷贝 Worker、最小插件、Lock 和许可证；构建上下文排除历史/reference 运行时。本机未执行 Docker 构建或运行。
