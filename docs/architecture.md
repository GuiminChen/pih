# Architecture
English | [中文](architecture.zh.md)

PIH separates a native worker from a Lock-selected capability graph. Plugins publish typed capabilities; startup resolves providers and checks ABI, lifecycle and deployment compatibility. Kernel Packs separate device-specific code from runtime contracts.

The public C ABI lives in `include/pih/plugin_sdk`. The installed `PIHPluginSDK` CMake package exposes `PIH::plugin_sdk` without linking the runtime. It is a preview interface; use SDK and worker from the same revision.

Native Qwen, DeepSeek PP1 and B300 models own their execution code; shared providers own resource and transport capabilities. See [integration status](integration-status.md) for build evidence and [support boundaries](support.md). Historical architecture records remain workspace development material, outside the default release.
