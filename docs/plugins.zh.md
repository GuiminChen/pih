# 插件开发
[English](plugins.md) | 中文

当 `PIH_BUILD_MONOLITH=OFF` 时，普通安装也只发布显式 SDK 头文件，不再复制内部
model/backend/core 头文件或 `status_bridge.h`。从旧 Bundle 迁移时必须使用新的
安装目录：CMake 安装不会自动删除旧目录中残留的头文件。当前顶层 CMake 已拒绝
单体推理模式。此前的本地原生安装输出了最初 10 个公开头文件和 3 个 SDK CMake
文件；当前安装规则已扩展为全部 19 个 C 能力契约和 6 个 ABI 头文件（共 25 个），
包括执行、传输、设备、内存、异步、健康、平台和制品接口，仍不暴露
内部 C++ 实现。外部示例使用安装目录配置并编译、链接成功，
测试关闭。Windows 交叉工具链发出 `.drectve` 链接警告，因此 DLL 加载、导出和 Linux
运行行为仍未验证；这些只是安装与构建证据，不是插件执行资格证明。

安装到新目录后，运行 `python3 tools/verify_plugin_sdk_install.py /absolute/sdk`
对照当前源码检查安装边界。检查器要求恰好 25 个头文件、内容逐一匹配源码，检查
其中 `pih/` 引用未越出公开集合，并要求 3 个 CMake 包文件存在。无需安装时，可运行
`python3 tools/verify_plugin_sdk_install.py --source-only` 核对校验器清单、CMake
安装规则与外部 C 探针是否一致。多余内部头文件和
过期副本均会失败。它拒绝头文件树中遇到的符号链接，但不是签名包或恶意文件系统
验证器；包文件存在不等于其语义正确，仍须做外部消费者配置/构建。旧版检查曾在
本地新安装目录通过；扩展后的 25 头文件边界仍需重新配置、安装并构建消费者。

外部示例还会构建 `sdk_c_header_probe`：一个从安装目录包含全部 25 个公开头文件的
C11 对象文件，不执行代码、不链接模型。`hello_plugin` 构建依赖该探针，CI 的普通
消费者构建因此同时检查 C 兼容性和 C++ 示例。旧版探针曾编译通过；扩展版仍需
重新构建，且这不代表跨编译器 ABI 布局或二进制兼容性已验收。

先按[开发指南](development.zh.md)构建 Worker，然后运行[独立外部插件示例](../examples/external-plugin/README.md)。

```bash
cmake --install out/build/phase1-native --prefix "$PWD/out/sdk" --component pih-plugin-sdk
```

使用方调用 `find_package(PIHPluginSDK 0.1.0 EXACT CONFIG REQUIRED)`，链接 `PIH::plugin_sdk`。SDK 组件仅包含 C ABI 头文件，不包含内部 C++ 状态桥接器。

示例注册一个 activation 范围的能力，在 configure 时解析它，并按 register/configure/start/ready/drain/stop/dispose 顺序执行。它检查宿主 ABI，阻止异常跨越 C 边界，注册失败不会推进生命周期。启动失败后的回滚也需要正确关闭资源。

[原生预览 CI](../.github/workflows/native-preview.yml)将示例复制到源码树外，再构建并加载。开发 Lock 不是生产部署签名，也不隔离不可信原生代码。
