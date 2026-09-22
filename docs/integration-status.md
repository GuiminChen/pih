# Native integration status
English | [中文](integration-status.zh.md)

This delivery separates code/build acceptance from hardware qualification.
The production matrix covers Qwen3-0.6B on RTX 4090 D (SM89) and H100 PCIe
(SM90), DeepSeek V4-0731 on RTX 4090 D PP1 (SM89), and DeepSeek V4.1 text
backbone on B300 (SM103, 2/4/8 ranks). All use native Workers, C ABI capabilities
and native Kernel Packs. No Python model runtime is selected or used as fallback.

| Path | Code and build evidence | Execution status |
| --- | --- | --- |
| Qwen 4090 D / H100 | Production CMake model, native tools and Pack final links; separate SM89/SM90 cubins, 18 entrypoints each | Hardware execution unverified |
| DeepSeek V4-0731 PP1 | Production model, three artifact commands and SM89 Pack final links; 17 embedded cubins | Hardware execution unverified |
| DeepSeek V4.1 B300 | Model, rank Worker, NCCL helper/provider, supervisor, offline tools and SM103 Pack final links; 20 embedded cubins | Hardware execution unverified |
| Shared runtime | Worker/CLI, providers, strict linking and explicit production installation | Linux lifecycle execution unverified locally |

Local device compilation used Clang 22.1.8 plus genuine CUDA 13.2.86 libdevice,
ptxas and fatbinary. This is device machine-code generation, not host-only or
PTX-only evidence. Clang warns that CUDA 13.2 exceeds its partially supported
12.9 range; that warning is retained. Native NVCC builds and remote CI have not
been executed in this local environment. The Linux cross-linked products use
real CRT, standard, OpenSSL, ICU, CUDA, cuBLAS and NCCL libraries; no substitute
symbols, stub libraries or unresolved-symbol exemptions were used.

Authenticated shared libraries load from sealed executable memfds. Kernel Packs
receive an explicit original installation location before their contract is
published. Qwen external cubins therefore remain locatable after sealed loading.
The preview Pack ABI gained a required `bind_origin` callback: rebuild the SDK,
Worker, rank Worker and every Pack together. Older, smaller tables are rejected;
there is no compatibility adapter. Plugin/Pack hashes and deployment Locks must
be regenerated after any rebuild.

GPU execution, real weights, inference, numerical correctness, performance and
multi-GPU execution are **unverified** and excluded from code/build acceptance.
A build receipt does not claim target support, resource capacity or numerical
correctness. Remote CI configuration is not a passing CI result.

Use the [deployment guide](deployment.md) and [release boundaries](release.md).
