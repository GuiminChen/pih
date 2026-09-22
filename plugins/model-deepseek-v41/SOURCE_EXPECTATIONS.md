# V4.1 source expectations / 源 checkpoint 可信清单

`WeightSourceCheckpoint::Open` accepts a source directory descriptor, the JSON
bytes below, and an **independently trusted SHA-256 of those exact JSON bytes**.
The `pih-v41-convert` command wraps this native offline API.

这是原生离线 API 和转换命令使用的清单。调用方必须从可信
发布记录取得清单原始字节的 SHA-256，不能对任意本地文件自行算摘要后就宣称来源可信。

## Schema / 格式

The root has exactly five fields. Every file object has exactly `name`, `bytes`
and `sha256`. The following is a schema illustration, **not a usable manifest**;
replace placeholders with independently verified byte counts and lowercase digests.

根对象严格包含五个字段，每个文件对象严格包含三个字段。下面只示意结构，
占位符不是有效参数，也不得用猜测值替代。

```json
{
  "schema": "pih.deepseek-v41.source-expectations.v1",
  "reference_revision": "517ef625df97ec57aadc91b67506a57c20fdc5bb",
  "config": {
    "name": "config.json",
    "bytes": "REPLACE_WITH_POSITIVE_INTEGER",
    "sha256": "REPLACE_WITH_TRUSTED_SHA256"
  },
  "index": {
    "name": "model.safetensors.index.json",
    "bytes": "REPLACE_WITH_POSITIVE_INTEGER",
    "sha256": "REPLACE_WITH_TRUSTED_SHA256"
  },
  "shards": [
    {
      "name": "model-00001-of-00001.safetensors",
      "bytes": "REPLACE_WITH_POSITIVE_INTEGER",
      "sha256": "REPLACE_WITH_TRUSTED_SHA256"
    }
  ]
}
```

Limits: expectation JSON 128 KiB; config 1 MiB; index 64 MiB; 1–256 unique shards,
each at most 512 GiB. All hashes must be nonzero lowercase SHA-256. Source shard
headers retain the 16-MiB/4096-tensor parser limits, and the complete index is
limited to 200,000 tensors. The index must describe exactly these shard names and
their actual tensors/payload total. Extra files not listed in the manifest are not
opened or treated as model inputs. No omitted indexed tensor is silently skipped.

清单绑定配置、索引和全部分片的具体字节。实际源文件必须是只读、单硬链接的普通
文件；叶节点符号链接被拒绝。调用方负责安全取得正确目录描述符，并排除同用户/root
并发修改。目录描述符会固定目录对象，但不会证明取得它的路径可信。

## Conversion scope / 转换范围

After admission, `ConvertWoA(layer, writer)` locates the exact weight/scale pair
through the owned source inventory, validates their formats and streams canonical
BF16 bytes to the caller's unpublished output. No raw source file descriptors are
exposed. Failure invalidates the whole partial tensor output; publication belongs
to the future complete conversion workflow.

`ConvertBackboneTensor(canonical_name, writer)` 也已接入常规主干张量转换：
名称与物理形状必须匹配运行端 TP1 清单；支持 BF16/F32、原类型 FP8/scale 复制、
FP4 打包字节保留，以及可精确表示的 F32→E8M0 scale 转换。未知类型、非有限值和
溢出直接拒绝。不会根据文件内容猜测量化格式，也不会对任意 scale 自动舍入。

共享专家的打包 FP4 输入也已接入：I8/U8 权重形状必须为 `[rows,columns/2]`，
E8M0 scale 必须为 `[rows,columns/32]`。权重和 scale 成对预检查并转为运行端要求
的 FP8 权重和块 scale；路由专家仍保留 canonical FP4。当前每对共享专家执行
两遍有界转换，分别输出 scale 和权重，避免整张结果驻留内存，但会重复读取源数据。
这一路径尚未完成数值对照验证；不支持的 scale 格式仍报错。

`ValidateBackboneConversion(excluded_names)` 会预先检查全部主干必需张量及转换
类型，并核对每个源张量的去向。未被主干转换消耗的源张量必须按规范化名称逐项
列入排除清单；不接受通配符、未知名、重复名或排除必需输入。
`ConvertBackbone(excluded_names, writer)` 先执行这项检查，再按运行端清单顺序
输出全部张量。调用方必须把排除清单保存在转换来源记录中。任意失败都使整个
未发布输出无效；预检查不扫描数值，转换过程中仍可能发现数值错误。

`MaterializeSourceBackbone(staging_fd, source, excluded_names, device_budget)`
已将整套转换接到真实文件输出，生成 canonical TP1 的 41 个分片、运行端权重清单
和 `conversion.provenance.json`。来源记录绑定源可信清单摘要、配置摘要、输出权重
清单摘要及排序后的完整排除列表，上限 16 MiB；排除项大小在写入分片前检查。
文件独占创建、同步、设为只读并读回校验。返回回执包含权重清单和来源记录的摘要，
总字节数包含来源记录；失败仍保留所有临时文件，不自动删除或发布目录。

逐项排除并不代表实现了视觉/MTP/辅助能力。尚不支持的量化转换，以及原始
checkpoint 的未支持量化格式仍会被拒绝。运行端只校验权重清单时不会自动校验
额外的来源记录，发布流程还需保存并核对来源记录摘要。
清单与索引一致，不等于已证明参考模型所有所需张量都具备
正确数值或形状；`reference_revision` 标签也不能单独证明来源。

## CLI / 命令行

On Linux x86-64, build both CPU tools from the repository root (C++20 compiler,
CMake, Ninja and native CPU dependencies including OpenSSL development headers):

```bash
bash deploy/run-v41-native.sh build-weights /absolute/build/v41-weights 4
bash deploy/run-v41-native.sh convert /absolute/build/v41-weights \
  "$SOURCE" "$EXPECTATIONS_JSON" "$EXPECTATIONS_SHA256" \
  "$EXCLUSIONS_JSON" "$EXCLUSIONS_SHA256" "$DEVICE_BUDGET_BYTES" "$OUTPUT"
```

Set variables explicitly before invocation. `SOURCE` is the source directory;
`EXPECTATIONS_JSON` is the expectation document above. `EXCLUSIONS_JSON` is a
separately approved document with exactly this schema (empty only if no source
tensor needs excluding):

```json
{"schema":"pih.deepseek-v41.conversion-exclusions.v1","excluded_source_names":[]}
```

Exclusion names are normalized source names, not wildcard patterns. Both JSON
paths must name read-only single-link regular files and both hashes must come from
independently approved records. Exclusions are bounded to 16 MiB/200,000 names;
unknown, duplicate and required-input exclusions are rejected before conversion.
The command does not synthesize approvals or automatically discard extra tensors.

所有路径必须为绝对路径且不经过符号链接。先设置变量：源目录、源可信清单及其
摘要、经确认的排除清单及其摘要、TP1 对齐权重字节预算、新输出目录。预算不包含
推理缓存和临时显存，不表示 TP1 推理能运行。源文件及两份清单须保持只读且不被
并发修改；程序不能隔离同用户/root 恶意写入。

输出父目录须已存在、归当前有效用户所有，且组/其他用户不可写。`OUTPUT` 和
`OUTPUT.staging` 都必须不存在。先独占写入 staging，全部完成后设为只读，通过
`renameat2(RENAME_NOREPLACE)` 发布，并同步父目录；不支持该操作时不回退到覆盖。
准备足够磁盘空间：源文件、完整 canonical TP1 输出以及后续 rank 输出可同时存在。
大表复制和读回会产生大量 CPU/I/O 开销；命令不下载模型、不启动 GPU、不运行测试。

Exit 0 emits `pih.deepseek-v41.conversion-receipt.v1`, with source/config/exclusion
hashes, output manifest/provenance hashes, TP1 rank 0, shard count and total bytes.
Save this receipt. Exit 2 means no success receipt; inspect retained staging. Exit
3 means rename succeeded but a later sync/receipt step failed; inspect the output
instead of blindly rerunning. Interrupted processes can also leave output without
a receipt. No automatic cleanup, overwrite, resume or model activation occurs.

成功输出可作为 [canonical 重分片命令](RESHARD.zh.md) 的 TP1 输入；后续多卡部署
须分别生成各 rank，不是一次原子发布整个多 rank 集合。此转换输出只覆盖已实现的
文本主干类型，逐项排除视觉/MTP/辅助张量不会自动实现那些能力。

All checks so far are compilation/static checks. No source admission run,
checkpoint conversion or numerical/model test has been performed for this API.

The `weights-linux-link` job in `native-v41-build.yml` builds both commands using
the documented `build-weights` entrypoint and inspects their ELF dependencies
without running them. This workflow has been added, not observed passing here.
The local `check_native_syntax.py --artifact-tools` check explicitly compiles the
conversion macro variant as well as the default reshard variant; syntax results
are not Linux linker or execution evidence.
