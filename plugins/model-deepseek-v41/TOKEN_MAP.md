# Engram token-map preparation / 映射准备

Build on Linux x86-64 with C++20, CMake, Ninja, OpenSSL and ICU development
dependencies. The CPU tools do not require CUDA or Python inference:

```bash
bash deploy/run-v41-native.sh build-weights /absolute/build/v41-weights 4
bash deploy/run-v41-native.sh token-map /absolute/build/v41-weights \
  /absolute/tokenizer.json "$TRUSTED_MAP_SHA256" /absolute/output/engram-map
```

`TRUSTED_MAP_SHA256` must come from an independently admitted reference map, not
from this generator's own output. The tokenizer must match the repository's fixed
V4.1 tokenizer digest and be a read-only single-link regular file. Paths must be
absolute and traverse no symlinks. No tokenizer/map is downloaded automatically.

映射摘要必须来自独立可信参考记录；不能用本工具生成的摘要自证正确。Unicode/ICU
差异可能导致结果不同，因此即使压缩词表条数正确，只要完整摘要不符就会失败。
当前尚未运行完整映射生成或参考对照，编译通过不是 tokenizer 等价性证明。

The command generates 129280 little-endian U32 entries (517120 bytes), requires
99092 first-appearance compressed IDs, then compares the complete digest. Only
after matching does it create `OUTPUT_DIRECTORY.staging`, write/read back
`compressed-token-map.bin`, set file mode 0400 and directory mode 0500, rename with
`RENAME_NOREPLACE` and sync the parent. The parent must already exist, be owned by
the effective user and not group/other writable; output and staging must not exist.
Exclude concurrent same-user/root writers: these checks are not hostile-host isolation.

退出码 0 和 JSON 回执表示发布流程成功，回执记录 tokenizer 摘要、map 摘要、
字节数以及实际 ICU/Unicode 版本。退出码 2 时先检查保留的 staging；退出码 3
表示目录已改名，但之后同步或回执失败，应检查输出，不能盲目重跑。进程中断也
可能留下临时或已发布文件。工具不会自动删除、覆盖、续传或激活模型。

## Runtime placement / 运行端放置

The native worker reads `compressed-token-map.bin` from each configured artifact
root and validates it against supervisor `model.map_sha256`. During deployment
assembly, place an independent read-only regular copy in each root, alongside the
required configuration/weight artifacts, and retain the matching digest. Do not
hard-link it: runtime admission requires one link. Do not overwrite an existing
deployment to insert this file. The generator publishes a standalone map directory,
not a ready-to-run multi-rank bundle; bundle assembly remains a separate step.

Prefer adding `--runtime-map /absolute/engram-map/compressed-token-map.bin "$TRUSTED_MAP_SHA256"`
to the end of a `convert` or `reshard` command. It copies the admitted map and exact
configuration into the rank's private staging directory before no-replace
publication. The worker expects configuration under `hf_config.json`, not the HF
source filename `config.json`. Receipt totals include both metadata files; save
the map digest for supervisor `model.map_sha256`. No existing rank is modified.

运行端要求每个制品根目录内包含同名文件，并用 `model.map_sha256` 校验。部署组装
时放入独立只读副本，不要使用硬链接，也不要直接覆盖已有部署。这里只生成辅助
映射，不会生成模型配置、权重、插件锁或完整多 rank 部署。

推荐在 `convert` 或 `reshard` 命令末尾追加上述 `--runtime-map` 参数。程序会在
同一 rank staging 内写入 `hf_config.json` 和 `compressed-token-map.bin`，随后
一起发布，避免手工补文件。插件锁、进程配置及整个多 rank 集合仍须另行组装。

Current evidence is object compilation/static checks only. The Linux CI job builds
and inspects this command without execution; that CI has not been observed passing here.
