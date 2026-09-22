# Canonical V4.1 rank resharding

English | [中文](RESHARD.zh.md)

This CPU-only developer command consumes an **already canonical, authenticated
TP1 text-backbone artifact**. It is not a raw Hugging Face checkpoint converter,
does not create that prerequisite artifact, and does not establish model/B300
qualification. The separate [source conversion command](SOURCE_EXPECTATIONS.md)
can create canonical TP1 artifacts for its implemented input types; other source
formats remain unsupported.

## Build

Use Linux x86-64, a C++20 compiler, CMake, Ninja and the repository's native CPU
build dependencies (including OpenSSL development headers). No CUDA or Python
inference runtime is required. From the repository root:

```bash
bash deploy/run-v41-native.sh build-weights /absolute/build/v41-weights 4
```

Use a separate build directory from GPU inference builds. This command builds
`pih-v41-reshard`; it does not execute it, download checkpoints or run tests.

## Inputs and invocation

- `CONFIG`: absolute path to the pinned V4.1 HF configuration JSON.
- `CONFIG_SHA256`: independently trusted nonzero lowercase hash of its raw bytes.
- `SOURCE`: absolute directory containing canonical TP1 `weights.manifest.json`
  and all its shards; the manifest binds world size 1, rank 0 and this configuration.
- `SOURCE_MANIFEST_SHA256`: independently trusted digest, not an automatically
  generated hash that treats untrusted input as authentic.
- `WORLD`, `RANK`: output TP1/2/4/8 and zero-based rank below world size.
- `DEVICE_BUDGET_BYTES`: positive integer limit for this rank's aligned weights;
  it excludes runtime caches/scratch and is not a whole-model memory guarantee.
- `OUTPUT`: absolute new directory. Its existing parent must be owned by the
  effective user and not group/other writable. Neither `OUTPUT` nor
  `OUTPUT.staging` may exist. Paths must not traverse symlinks.

Configuration, source manifest and source shards must be read-only single-link
regular files. Keep the source immutable and exclude concurrent same-user/root
writers throughout. This is not a hostile-host isolation mechanism.

```bash
bash deploy/run-v41-native.sh reshard /absolute/build/v41-weights \
  "$CONFIG" "$CONFIG_SHA256" "$SOURCE" "$SOURCE_MANIFEST_SHA256" \
  "$WORLD" "$RANK" "$DEVICE_BUDGET_BYTES" "$OUTPUT"
```

Set those variables from your trusted deployment record first. The command reads
and authenticates the complete source, copies the selected rank with a 1-MiB
payload workspace, reads back every output shard, and emits the runtime manifest.
Metadata memory is separate. Disk space must accommodate the complete output rank;
no disk-space reservation is made. Large Engram tables make this a substantial
CPU/I/O operation even though no GPU executes. Each rank invocation reauthenticates
the full source; run ranks separately rather than assuming shared verification.

## Publication and failure handling

Append `--runtime-map /absolute/compressed-token-map.bin "$TRUSTED_MAP_SHA256"`
to include `hf_config.json` and `compressed-token-map.bin` in the unpublished
rank directory. The map is admitted before conversion; both metadata files are
read back and synced before publication. Receipts add map digest/metadata byte
count. Without this option, the output remains weights-only, not a ready worker root.

Files are exclusively created in `OUTPUT.staging`, synced and made mode 0400.
The completed staging directory becomes mode 0500 and is renamed to `OUTPUT` with
Linux `renameat2(RENAME_NOREPLACE)`, then its parent is synced. Unsupported
filesystem/kernel no-replace behavior fails without an overwrite fallback.

Exit 0 plus the JSON receipt reports the source/config bindings, output manifest
digest, rank/world, 41 shard count and total bytes including the manifest. Save the
receipt; use its manifest digest in the subsequent runtime configuration only
after applying your deployment admission policy. It is not numerical parity proof.

Exit 2 means no successful publication receipt; inspect retained staging. Exit 3
means the directory was renamed but a later sync/receipt step failed; inspect the
output instead of rerunning blindly. Interruptions can also leave staging, or an
already-renamed output without a receipt. Manifest presence alone is not success.
No automatic deletion, overwrite, resume or model activation occurs. Cleanup and
recovery require explicit operator inspection. Atomicity covers one rank directory,
not a multi-rank deployment transaction.

Current verification: Linux-target object compilation and Bash syntax checks only.
Full linking, actual resharding and model tests have not been performed here.
