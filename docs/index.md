# Documentation
English | [中文](index.zh.md)

See [native plugin inference](native-inference.md) for the new Qwen build/launch
path, exact environment matrix, request examples and observation commands.

Start with [development](development.md) to build the native worker. Continue with [plugins](plugins.md) to build an external plugin using the installed SDK.

- [Architecture](architecture.md)
- [Deployment and qualification](deployment.md)
- [Python HTTP client](client.md)
- [Support matrix](support.md)
- [Release checklist](release.md)
- [DeepSeek V4.1 Flash / B300 onboarding](deepseek-v41-b300.md)

This preview exercises native composition. There is no qualified model/GPU profile and no general-purpose `pih serve` command yet. The Python wheel contains only an HTTP client and host diagnostics, not the native bundle or an old inference compatibility layer.
