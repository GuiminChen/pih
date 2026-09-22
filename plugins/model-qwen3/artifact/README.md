# Qwen artifact representation

These eight CPU-only implementations are shared by native Qwen loading and the
offline artifact tools: artifact I/O, layout, metadata, disposition, linear
shape ledger, source binding, model manifest and source-artifact metadata.
They do not implement quantization, model
execution or CUDA allocation. The offline tool target selects these exact files
without compiling the parent directory's engine implementation.

The conversion/publication pipeline remains in `plugins/offline-qwen`; it is not
part of the model's runtime source set. Declarations retain their current
`include/pih/model` paths pending the remaining header migration. No forwarding
implementation is retained in the old `src/model` directory.

这 8 个 CPU 制品实现由模型加载和离线验证共用，包含 INT4 表示及模型清单、
源制品元数据，不承担量化或 GPU 执行。
离线构建显式选择文件，不编译父目录的引擎源码；转换流水线仍独立位于
`plugins/offline-qwen`。旧路径不保留副本，头文件路径暂不变。
