# Native deployment
English | [中文](deployment.zh.md)

The four profiles have a native code/build delivery path. **Hardware execution
is unverified**: the commands that start services or process model artifacts
below are instructions for the operator, not tests executed during delivery.
Build, install and package commands require no GPU activation.

## Build host and target requirements

Use Linux x86-64, Bash, Python 3.11+ for orchestration, CMake 3.26+, Ninja,
a C++20 compiler, OpenSSL 3 and ICU development files, and CUDA Toolkit 13.2+
with support for the exact SM. B300 additionally requires NCCL 2.31.2 headers
and a real matching shared library. Supply toolchain/dependency locations with
CMake settings; scripts do not install drivers/packages or download weights.
Qwen requires SM89 or SM90; PP1 requires SM89; B300 requires `103-real`.
The public cubin rule supports NVIDIA and Clang CUDA driver syntax. Clang builds
also require the toolkit's cuRAND headers; use `PIH_CUDA_DEVICE_INCLUDE_DIRS`
when headers live outside the toolkit include tree.

A cross build must provide real target CRT and runtime libraries. The local
artifacts require glibc 2.39-era runtime, libstdc++ with GLIBCXX_3.4.32,
ICU74, libcrypto.so.3, libcudart.so.13, libcuda.so.1, and, where selected,
libcublasLt.so.13 or libnccl.so.2. Exact transitive requirements are the ELF
DT_NEEDED and version tables of the installed product. Do not substitute
stub libraries or copy the build-host compatibility driver onto the target.
Third-party libraries and weights are not included in the archive.

The Worker rejects loader injection environment variables. Install libraries
through a reviewed system loader configuration; do not use `LD_PRELOAD` or an
uncontrolled `LD_LIBRARY_PATH`. Do not activate GPU providers on a build host.
Authenticated plugins require Linux executable `memfd_create` (`MFD_EXEC`),
file seals and readable `/proc/self/fd`; denied operations fail closed.
B300 execution additionally requires the explicitly delegated cgroup-v2 parent,
clone3, pidfd and execveat permissions described in the supervisor schema.

## Build and package without weights or GPU execution

Choose one profile: `qwen-4090d`, `qwen-h100`, `deepseek-pp1`, `deepseek-b300`.
Use separate build and new installation directories for each:

```bash
python3 deploy/build-release.py qwen-4090d   --build-dir /absolute/build-qwen89 --bundle /absolute/new-qwen89 --jobs 4
python3 tools/package_release.py native --root /absolute/new-qwen89   --output /absolute/new-qwen89.tar.gz
```

Repeat with the other profile names and distinct directories. For a cross
build append `--cmake-arg=-DCMAKE_TOOLCHAIN_FILE=/absolute/toolchain.cmake`;
additional `--cmake-arg=-DNAME=value` settings select real dependencies.
The closed profile options cannot be overridden by these extra arguments.
The script builds `pih-release` and installs component `pih-release` only.
`NATIVE-INSTALL.json` records the precise installed members; packaging rejects
changed, missing or extra files and creates a read-back-verified archive with
`RELEASE-MANIFEST.json`. Neither step creates a model-specific trusted Lock.

Unpack in a new owner-controlled directory; retain bin/lib and Qwen's
`lib/qwen3-smXX/sm_XX` cubin+manifest layout. Do not mix Workers, SDK, models or
Packs from different revisions. The current Pack ABI requires `bind_origin`;
old tables are rejected. Rebuild and reseal after changing any binary. The
model DSO embeds the built Qwen cubin digest and verifies the owned image
snapshot before it reaches CUDA; replacing the image and its sidecar fails.

## Qwen 4090 D / H100 launch

Use a pinned Qwen3-0.6B revision and independently trusted artifact/config
SHA256 values. BF16 uses `model.safetensors`; INT4 uses native
`model.xing-int4`, not GGUF/AWQ/GPTQ. Both require matching config/tokenizer.
Use the shipped `pih-qwen-source-verify`, `pih-qwen-int4-convert` and
`pih-qwen-int4-verify` commands explicitly when preparing artifacts; conversion
is not performed implicitly by serving. Keep weights outside source/build/bundle.
The [artifact and request guide](native-inference.md) documents their inputs.

```bash
bash deploy/run-native.sh serve --target rtx4090d --precision bf16   --bundle /absolute/new-qwen89 --build-dir /absolute/build-qwen89   --model-dir /srv/models/Qwen3-0.6B   --artifact-sha256 "$TRUSTED_ARTIFACT_SHA256"   --config-sha256 "$TRUSTED_CONFIG_SHA256" --max-context 4096 --port 8000
```

For H100 PCIe select `h100-pcie` and its SM90 bundle. Add `--build` only when
using the source distribution. The launcher validates GPU identity and seals
binary/artifact identities before exec of the native Worker. It cannot mint
trusted model provenance from arbitrary local files.

## DeepSeek V4-0731 PP1 launch

Use native artifact prepare/verify/generation-store tools and the pinned semantic
snapshot. A raw HF checkpoint is not a runtime generation. The production PP1
path is single-rank with sequential token plans; concurrent multi-token prefill
is not delivered. No old interprocess PP path is restored.

```bash
bash deploy/run-native.sh serve-deepseek   --bundle /absolute/new-pp1 --build-dir /absolute/build-pp1   --artifact-dir /srv/models/deepseek-v4-0731-pp1 --artifact-root "$ARTIFACT_ROOT"   --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731   --max-context 4096 --port 8000
```

The launcher validates/stages external artifact authority and starts the native
text/SSE surface. Missing or mismatched artifacts fail; no reference runtime is
selected. Supported request fields and semantic admission are in the
[artifact and request guide](native-inference.md).

## DeepSeek V4.1 B300 launch

Prepare canonical TP1 artifacts with the native converter, then reshard into
2/4/8 rank placements. Canonical TP1 is an offline format, not an admitted
single-rank production backbone. Vision/MTP and arbitrary checkpoints are not
claimed. Use the exact [supervisor schema](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md)
for budgets, placements, tokenizer/map and per-rank digests.

```bash
bash deploy/run-native.sh v41 seal-rank /absolute/new-b300
# Review supervisor.json with matching worker/helper/rank-lock hashes,
# external artifacts and a delegated cgroup parent; obtain its trusted digest.
bash deploy/run-native.sh v41 seal-service /absolute/new-b300   /absolute/supervisor.json "$TRUSTED_CONFIG_SHA256"
bash deploy/run-native.sh v41 serve /absolute/new-b300 8000
```

Existing locks are not overwritten. For a source-only one-command service build,
use `bash deploy/run-native.sh v41 build-service BUILD NEW_BUNDLE JOBS` with
absolute paths, then seal as above. Rank Worker, transport/NCCL helper,
supervisor and SM103 Pack are all native; build success is not communication
or numerical verification.

## Operation and failure handling

Services bind loopback and serialize admitted text requests. Use an authenticated
reverse proxy if exposing a service; no built-in TLS/auth is claimed. Configure
`--generation-timeout-ms` for Qwen/PP1 and supervisor limits for B300. SIGTERM,
Ctrl-C and disconnects enter native cancellation/retirement; pending CUDA/NCCL
work is not completion. Unproven retirement fail-stops the process rather than
reporting success or switching runtime. Inspect exit status and stderr, retain
the sealed Lock and deployment receipts, and use a new bundle to upgrade.
The diagnostic `probe` command performs inference and is intentionally outside
this delivery's no-hardware validation scope.
