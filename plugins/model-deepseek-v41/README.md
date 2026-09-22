# Native DeepSeek V4.1 model implementation

This directory is the native implementation boundary, separate from the vendored
Python reference. It now includes a serial text inference plugin, authenticated
supervisor deployment and the SM103 rank-worker path. The independent SM103
Kernel Pack lives in `plugins/kernels/deepseek-v41-sm103`. These are implemented
code paths, not a claim of model correctness or B300 hardware qualification.
Later sections preserve implementation history and may describe earlier stages;
use `docs/deepseek-v41-b300.md` and the current source for deployment status.

## Build ownership

`SupervisorDeployment` freezes the authenticated supervisor configuration and
retains one shared tokenizer across serialized request admissions. The existing
CLI now uses this path. Requests cannot supply artifact/executable paths, device
placement, resource budgets or deadlines; completion length is bounded by both
the deployment limit and ledger memory, and model stop-token identities cannot
be replaced. Successful admissions allocate increasing sequence/sampling IDs
without wrapping. Request owners retain tokenizer lifetime independently.
This is internal native admission, not HTTP parsing or model registration.

`SupervisorDeployment` 固定已认证配置并复用 tokenizer，现有 CLI 已接入。
请求接口不接收路径、设备放置、资源预算或超时；生成长度受部署上限及账本内存
共同约束，不能替换模型停止 token。成功接纳的请求递增分配身份，耗尽时报错；
请求通过共享所有权保持 tokenizer 存活。这是原生内部接纳层，不是 HTTP 入口
或模型注册完成。五个策略回归用例已加入但未运行。

`SupervisorOutput` now owns publication delivery bookkeeping shared by native
front ends. The supervisor CLI uses it directly: each write offers at most 4096
bytes, zero consumption retains the lease for backpressure, failures latch and
require explicit discard, and completion requires every accepted publication to
have been delivered. It neither launches nor retires processes; the CLI still
requires successful cgroup reconciliation before its completion receipt. This
extraction prepares service integration but does not register a model capability.
Both changed implementations passed Linux-target syntax compilation. Six native
regression cases were added, not compiled or executed; no runtime qualification
is implied.

`SupervisorOutput` 已承接原生命令行入口的输出交付账本：每次最多写入 4096
字节，背压时保留租约，失败后停止输出并显式丢弃，成功收尾要求全部已接受记录
交付完成。它不负责进程启动或回收；CLI 仍须完成 cgroup 核对后才输出成功凭据。
这是服务接入的共用组件，不是模型插件注册完成。两个实现通过 Linux 目标语法
编译，新增六个回归用例尚未编译或执行。

`host.cmake`, included from this model directory, owns the CPU weight preparation,
sampling/output ledger and Linux process-supervision targets. The root project
supplies generic core/model-value support and output credits, then adds this
directory once. Native CLI names and their top-level build output locations are
unchanged. `gpu.cmake` now owns the 93 GPU source entries and their target/link
definitions; the root includes it only after CUDA/toolkit admission.
`worker_commands.cmake` owns the rank-worker and NCCL bootstrap executable
definitions and is included after microkernel/worker support exists. Both are
included in the root build scope to preserve binary paths. After normalizing
source/include paths and whitespace, the GPU definitions match their pre-move
contents. This directory split does not register an executable model plugin.

`tokenizer.cmake` now owns the V4.1 tokenizer, Engram map builder and token-map
command previously defined in the common tooling directory. It is included
before host supervision, under the same Linux/tokenizer-tools condition, and
resolves ICU in the model's directory scope. The shared byte-level tokenizer
implementation remains in `plugins/common`; model-specific sources no longer
appear in that directory's target definitions. The command still writes to the
top-level build directory. Linux/ICU linking remains pending; the non-tokenizer
Windows configure passed after this move.

The GPU module's 56 static targets and both worker/helper executable targets are
now `EXCLUDE_FROM_ALL`: merely enabling CUDA must not compile V4.1 as part of a
Qwen-only default build. Explicitly building a V4.1 target still builds its linked
dependencies. `deploy/run-v41-native.sh build` already names the supervisor,
rank worker, NCCL helper and backend targets, so its build scope is unchanged.
This is a subsequent build-selection change, not part of the byte-equivalent
move described above. CUDA compilation and the generated GPU dependency graph
remain unverified locally.

The move preserves all 48 host source entries. Windows CMake generation passed;
compilation stopped on unavailable C++ standard-library headers. The existing
Linux cross-configuration could not regenerate without its OpenSSL development
dependency. Neither result proves Linux linking or execution.

CPU 权重准备、采样/输出账本及 Linux 进程监督目标现由模型目录的 `host.cmake`
管理；根构建仅提供共用基础依赖并加入该目录一次，CLI 名称及构建输出位置保持不变。
48 条源码条目保持一致。GPU 的 93 条源码及其目标/链接定义也已迁入 `gpu.cmake`，
路径和空白归一化后与迁移前一致；rank-worker 和 NCCL 引导命令由
`worker_commands.cmake` 管理。根构建在相应依赖建立后包含它们，保持二进制路径
不变。此次迁移不代表原生模型插件已注册完成。关闭 CUDA 的 Windows 配置通过，
但编译缺少标准库头文件；Linux 交叉配置缺少
OpenSSL 开发依赖，尚不能证明 Linux 链接或执行通过。

V4.1 分词器、Engram map 构建器和 token-map 命令也已从共用目录迁入模型的
`tokenizer.cmake`，在 host 目标之前按原有 Linux/分词工具条件创建，并在本目录
解析 ICU 导入目标。共用 byte-level 实现仍保留在 `plugins/common`，命令输出
仍位于构建根目录。关闭分词工具的 Windows 配置通过，Linux/ICU 链接尚待验证。

随后将 GPU 模块的 56 个静态目标及 worker/helper 两个可执行目标设为
`EXCLUDE_FROM_ALL`，避免仅启用 CUDA 就在 Qwen 默认构建中编译 V4.1。
显式构建 V4.1 目标仍会带入链接依赖；现有部署脚本已经明确列出所需目标，
因此启动脚本的构建范围不变。该选择规则是迁移后的功能改动，不属于上面的
等价迁移；本机尚未验证 CUDA 编译或生成后的 GPU 依赖图。

### Rank-worker fail-stop lifetime boundary

The rank CLI now catches failures from runtime startup/polling and plugin shutdown
inside the lifetime of the runtime, artifacts, channels and plugin stack. It exits
with `_Exit(1)` before those objects can unwind, preserving storage potentially
borrowed by asynchronous CUDA/NCCL work until OS process reclamation. The previous
outer exception handler alone was too late to preserve local object lifetimes.
Only fixed stage labels are printed. Supervisor peer cancellation and cgroup
reconciliation remain required; a rank exit is not proof of successful retirement.
Linux-target syntax compilation passed with five aggregate-initializer warnings
from existing memory-owner headers. No failure injection or GPU execution was run.

Rank Worker 在 runtime 启动/轮询及插件关闭阶段，现于运行时、制品、通道和插件栈
对象仍存活的作用域内捕获异常，并调用 `_Exit(1)`，避免栈展开提前释放 CUDA/NCCL
异步操作借用的数据。只输出固定阶段标签；仍需 Supervisor 取消其他 rank 并核对
cgroup，进程退出不代表成功回收。Linux 目标语法检查通过，现有内存所有者头文件有
5 条聚合初始化警告；未运行故障注入或 GPU 测试。

## Offline canonical rank partitioning

`pih_deepseek_v41_partition_copy` is a CPU-only offline component.
`CopyCanonicalBackboneRank` validates the complete source tensor metadata against
the config-derived TP1 text-backbone inventory, then derives the requested TP1/2/4/8
rank inventory itself. It admits all copy geometries before invoking any writer.
Whole experts retain their global names and only the assigned rank's experts are
copied; replicated tensors, row slices, strided column slices and Engram row-tail
padding use the same runtime inventory as the reader. Weight padding is zero;
E8M0 scale padding is numeric one (byte 127), matching the reference conversion.
A caller-owned 1-byte
to 1-MiB workspace bounds payload memory independently of tensor size. Metadata
memory is proportional to the inventory, not the number of tensor rows.

Reader callbacks resolve canonical tensor names and must fill the exact requested
span. Writer callbacks receive the derived destination tensor geometry and exact
offset/span; all writes must remain unpublished until the whole operation succeeds.
Any callback failure invalidates the partial rank output. This component does not
open files, authenticate source bytes, generate manifests or publish destinations.
It copies canonical physical bytes, including packed FP4 bytes; it does not convert
raw checkpoint dtypes or rename raw checkpoint tensors. Source conversion,
provenance binding and the file-publication CLI remain required before this can be
advertised as a usable V4.1 weight preparation command. No model test is implied.

`BackboneWeightOutputLayout` derives 41 safetensors members (one endpoint member
and one per main layer), with contiguous physical payloads and 8-byte-padded JSON
headers. FP4 uses the runtime's `U8` packed-byte representation. Generated prefixes
are reparsed by `BackboneWeightCatalog::Create`, including exact inventory and
aligned device-budget admission; independent file-offset rules are not maintained
by the writer. Each member is bounded to 512 GiB including its header.
`WriteCanonicalBackboneRank` joins this layout to the partition copier: it emits
headers and payload spans to a shard-name/file-offset callback, with each span no
larger than the caller's workspace. Source metadata is admitted before any write.
The caller must provide exact random-access writes to unpublished files; member
writes may be interleaved and must not be interpreted as append operations. A
failure leaves all partial output unusable. Success returns the output layout,
not file hashes or a trusted manifest.

The Linux-only `pih_deepseek_v41_materialize` target connects this writer to real
files through `MaterializeCanonicalWeightRank`. Its input is an already-open,
authenticated `BackboneWeightFiles` TP1 source with the same configuration digest.
It requires a borrowed descriptor for an empty private directory owned by the
effective user. Every output is created with `O_EXCL`/`O_NOFOLLOW`; contiguous
per-member writes prevent gaps, duplicate ranges and overruns. A 1-MiB workspace
is reused for copying and full readback. Each completed member is made read-only,
synced and read back against its streaming write hash. Only then is the exact
runtime `weights.manifest.json` generated, parsed with the runtime manifest parser,
written, read back and synced. Source identities are revalidated around the copy
and before returning the manifest digest. The receipt includes total file bytes
(including the manifest) and the number of weight shards (excluding the manifest).

This is canonical resharding, not raw checkpoint numerical conversion. The digest
is an observation of the materialized output, not independent source provenance.
Callers must exclude concurrent same-user/root writes and retain immutable source
storage; chmod and identity checks are not a hostile-host security boundary.
The function does not rename or publish the directory. Any failure retains its
partial contents, possibly including a manifest, and callers must not activate
them or infer success merely from a manifest's presence. No overwrite, cleanup or
resume path is provided. The CPU-only `pih-v41-reshard` CLI now supplies directory
creation, no-replace single-rank publication and a source/config-bound JSON receipt;
see [build, inputs and recovery instructions](RESHARD.md). Raw checkpoint numerical
conversion and independent source provenance still remain outside this operation.
Current evidence is Linux-target object compilation only, not linking or an
executed materialization.

## Offline Engram token-map generation

`BuildEngramTokenMap` authenticates the frozen tokenizer JSON and constructs the
129280-entry little-endian U32 map from single-token decoding. Tokens containing
the Unicode replacement character are keyed by original vocabulary spelling;
other tokens follow the reference NFKC, NFD, Mark removal, scalar-wise lowercase,
ASCII whitespace collapse, sentinel-preserved single space, Unicode trim and
sentinel restoration sequence. Empty normalized keys fall back to decoded text.
The shared tokenizer now exposes original spellings separately from output bytes,
including added tokens, without changing its existing decoding policy.

The CPU/ICU `pih_deepseek_v41_token_map` library is available with tokenizer tools
enabled. It checks the 99092 compressed-ID count, bounds individual keys and
aggregate key bytes, and requires the complete generated map to match an
independently trusted map digest. Matching count alone is not sufficient. ICU
Unicode-version differences can change output; a digest mismatch fails rather
than silently declaring reference equivalence. The implementation was checked
against the reference pipeline and upstream [Mark stripping](https://raw.githubusercontent.com/huggingface/tokenizers/main/tokenizers/src/normalizers/strip.rs),
[Mark classification](https://raw.githubusercontent.com/unicode-rs/unicode-normalization/master/src/lookups.rs)
and [scalar lowercase](https://raw.githubusercontent.com/huggingface/tokenizers/main/tokenizers/src/tokenizer/normalizer.rs)
semantics. No complete-map generation/reference comparison has run. The Linux
`pih-v41-token-map` CLI now writes and reads back the verified map, makes it
read-only and publishes its directory without replacing an existing target.
It is built by `build-weights` and invoked through `token-map`; see
[the map generation/deployment guide](TOKEN_MAP.md). This does not assemble a full
rank artifact directory or qualify inference.

## Offline wo_a and expert numerical conversion

The separate `weight_expert_convert` component implements the reference's packed
FP4-to-FP8 expert operation. `ConvertExpertFp4Block` decodes low nibble first from
32x16 packed bytes, takes 32 E8M0 row scales, selects max(scale)/64 as the shared
E8M0 scale and rounds values to E4M3FN with ties-to-even. It rejects NaN scales and
an unrepresentable shared scale below 2^-127; it does not silently clamp that case.
The reference's two FP4 zero codes become positive zero, while negative nonzero
values retain their sign if rounded to FP8 zero. This is not a losslessness claim.
`ConvertExpertFp4Tensor` handles logical 2304x5120/5120x2304 expert matrices in
32-row bands, emitting separate consecutive weight/scale streams with less than
256 KiB payload scratch. Both outputs become unusable if either writer fails.
Source dtype/shape binding and checkpoint dispatch are now connected for the
canonical shared-expert weight/scale pairs. Packed I8/U8 source weights must have
exact `[rows,columns/2]` geometry with E8M0 scales `[rows,columns/32]`; both canonical
output members are admitted together. Their inputs are marked consumed by the
full-backbone disposition check and cannot be excluded. Already-FP8 shared experts
continue through same-format conversion; routed experts retain their canonical
packed-FP4 storage. Other source scale formats for this operation remain rejected.
The tensor-ordered writer currently performs two bounded passes for each packed
shared-expert pair, emitting only the requested weight or scale each pass, so the
implementation does not retain a whole converted tensor but repeats source I/O.
Only Linux-target object compilation has run; no
numerical comparison, actual conversion or model test has been performed.

`weight_dequantize` provides the bounded numerical primitive used to prepare
canonical BF16 `wo_a.weight` from E4M3FN source blocks. `DequantizeWoABlock`
accepts exactly one 32x32 or 128x128 row-major block and its positive numeric
scale, performs FP32 multiplication, then rounds to BF16 with ties-to-even and
emits little-endian bytes. It rejects NaNs and FP32/BF16 overflow. A 32-KiB local
buffer validates the entire block before changing caller output, including when
input/output overlap. `DecodeWeightE8M0` decodes scale bytes without arithmetic,
including byte zero's 2^-127; byte 255 is rejected as NaN.

The offline target disables fast math and contraction. Conversion requires
round-to-nearest mode and, on x86-64, disabled FTZ/DAZ to retain gradual underflow.
It does not change the caller's floating-point environment. This matches the
operation sequence in the vendored reference converter, but numerical parity has
not been tested.

`DequantizeWoATensor` now gathers blocks from the full unpartitioned backbone
shape `[8192,4096]`. It admits the exact weight/scale geometry and byte extents,
accepts E8M0 or explicit little-endian F32 scale storage, reads contiguous source
row bands, and emits contiguous little-endian BF16 output bands of at most 1 MiB.
Its payload scratch is less than 2 MiB, independent of the full tensor extent.
Every block in an output band must pass conversion before that band is written;
a failure in a later band still invalidates all earlier unpublished output.
The callbacks must perform exact reads/writes and own source stability checks.
No callback is invoked for invalid geometry; callback exceptions become errors.

These scale storage branches are explicit conversion capabilities, not evidence
that a particular source checkpoint uses either format. `WoASourceBinding` now
joins parsed safetensors records for a selected main layer: the source weight must
be E4M3FN `[8192,4096]`, scale storage must be E8M0/F32, and the scale dimensions
must define square 32/128 blocks. It accepts only exact reference prefix aliases
(`model.` optional, `self_attn`/`attn`, `weight_scale_inv`/`scale`), with both records
in the same naming family. Mismatched layers/operators, unknown aliases and payload
extent differences are rejected. The binding owns canonical output name and
geometry, and translates bounded tensor-relative reads to each source file's
offsets; weight and scale can reside in different shards.

This binding validates metadata only. Its headers and file callbacks must come
from independently authenticated stable files; it is not a source digest receipt
or a complete-checkpoint inventory check.

The Linux `WeightSourceFile` owns a duplicate source-directory descriptor and a
read-only shard descriptor. `Open` takes a flat member name, independently trusted
size and SHA-256, requires a read-only single-link regular file (at most 512 GiB),
hashes every byte before parsing the safetensors header, and verifies file identity.
Headers retain the shared reader's 16-MiB/4096-tensor bounds. Payload reads use
1..1-MiB exact spans and recheck descriptor and directory-entry identity before
and after I/O. `ConvertWoASourceFiles` joins two such admitted files to the binding
and stream converter, and checks their identities again before reporting success.
Weight and scale may share one `WeightSourceFile` when stored in the same shard.

This admission relies on independently trusted caller expectations and immutable
source storage; it does not protect against hostile same-user/root writes or
authenticate the path by which the caller obtained its directory descriptor.
Its owned directory remains pinned if renamed. Callers must establish the correct
directory and complete checkpoint identity before using this API.

`WeightSourceInventory` authenticates the supplied HF index JSON against an
independently trusted digest, then checks the exact indexed shard/tensor set
against the admitted source files. It rejects missing/extra tensors, duplicate
shards, wrong shard assignments and `total_size` disagreement. It bounds input to
256 shards and 200,000 tensor records in addition to the shared parser limits.
Each name is normalized by exact path segments: optional leading `model.`,
`self_attn` to `attn`, non-vision `mlp` to `ffn`, `weight_scale_inv` to `scale`, and
`e_score_correction_bias` to `bias`. Distinct source names normalizing to one target
are rejected, rather than selecting one arbitrarily. It keeps all namespaces,
including vision/MTP/auxiliary members; downstream disposition must explicitly
account for those tensors. Its sorted lookup records preserve the input shard
order, which the caller must retain with the admitted file objects.

This proves consistency with the trusted index, not completeness against the
model's reference geometry. `WeightSourceCheckpoint` now binds configuration,
index and every shard expectation in one independently authenticated document,
owns their descriptors and source ordering, and exposes a layer-based `ConvertWoA`
that resolves the admitted source names internally. Configuration/index snapshots
are hashed before parsing; their descriptor identities remain checked after the
index text buffer is released. The original configuration snapshot is retained
for runtime metadata assembly. All files are revalidated before and after
conversion. The revision string is checked against the frozen reference, but its
truth still depends on the independently trusted document digest.

See [source expectation format and trust requirements](SOURCE_EXPECTATIONS.md).
`ConvertBackboneTensor` now resolves only exact TP1 runtime-inventory names and
checks source dimensions against their canonical physical shape. It routes FP8
`wo_a` through the scaled block converter; other tensors use `ConvertScalarWeight`:
BF16/F32 storage conversion, same-format FP8/scale copying, packed I8/U8 expert
bytes preserved as U8, and F32 scales converted to E8M0 only when they are exact
positive representable powers of two. It does not quantize arbitrary F32 scales.
Nonfinite values and BF16 overflow are rejected. Each input/output span is at most
1 MiB, with at most 2 MiB of payload buffers. Output offsets are consecutive and
each chunk is fully validated before its write; later failure invalidates the
entire unpublished tensor. Unknown names, changed geometry and other dtype pairs
are rejected rather than implicitly cast. Source file identities remain checked
on reads and around the individual tensor conversion.

`ValidateBackboneConversion` now checks every required TP1 target using the same
scalar admission or wo_a source binding as execution. It accounts for consumed
wo_a scales and requires every remaining source tensor to be listed individually
by normalized name in the caller's exclusion list. Missing targets, unsupported
conversion types, unknown/duplicate exclusions and exclusion of required inputs
fail before output. There is no namespace/wildcard drop policy. This admission is
metadata-only: malformed numeric payloads may still fail during conversion.
`ConvertBackbone` performs this admission before its first write, then emits every
target tensor in runtime inventory order and revalidates source identities on
completion. The caller must retain the exact exclusions in conversion provenance;
excluding a vision/MTP/auxiliary input does not implement that omitted capability.

`MaterializeSourceBackbone` now connects this source conversion to the same
exclusive file writer/readback/manifest pipeline as canonical resharding. It emits
a canonical TP1 text-backbone directory and a separately hashed
`conversion.provenance.json` binding the source expectation digest, configuration,
runtime manifest, conversion identifier and sorted exact exclusion list. The
provenance is capped at 16 MiB, with the exclusion-size bound checked before shard
writes. It is made read-only, synced and read back before success. The receipt's
total byte count includes provenance; its shard count remains 41. The existing
canonical reshard path has no new provenance member and returns a zero provenance
digest. Both paths retain partial files on any error and do not publish directories.

The `pih-v41-convert` command now supplies source/exclusion document admission,
private staging and no-replace publication using the same native entrypoint source
as the separate canonical reshard command. Build both with `build-weights`, then
invoke the explicit `convert` action described in [the source guide](SOURCE_EXPECTATIONS.md).
Unsupported quantization families still fail; this is not a claim that every
published checkpoint variant is covered. Individual
conversion success or an explicit exclusion policy does not constitute a complete
supported model profile. Runtime manifest admission alone does not validate the
separate provenance record; a publishing caller must retain/check its receipt hash.

Both commands accept `--runtime-map MAP_FILE TRUSTED_MAP_SHA256` to assemble worker
metadata in the same unpublished rank directory. The map digest and dense-ID
geometry are admitted before shard writes; after weight generation the exact
configuration bytes are saved as `hf_config.json` and the map as
`compressed-token-map.bin`, exclusively created, read back and synced read-only.
Their bytes are included in the receipt total and the map digest is recorded.
The pure CPU Engram hash target is shared with CUDA token upload; offline assembly
uses runtime map admission without linking CUDA. This assembles weight/config/map
inputs, not plugins, plugin locks, supervisor settings or an atomic multi-rank deployment.
No source shard or conversion was executed during this change.
Canonical resharding does not invoke these conversion components or silently
convert its inputs. Latest validation is Linux-target object compilation with
`-Wall -Wextra`, not an executed conversion or a reference numerical comparison.

## Native V4.1 multi-residual-stream primitives

The independent `pih_deepseek_v41_mhc_cuda` target implements fixed four-stream,
5120-feature mHC without linking the old model implementation. Its API separates
mix generation from pre-collapse, because V4.1 carries pre-mix across sub-blocks:
attention uses the preceding layer's FFN pre-mix; FFN uses this attention's
pre-mix; the new FFN pre-mix is returned to the next layer. Fusing the newly
computed mix into the current collapse would change the reference semantics.

- `LaunchMhcMix`: BF16 residual `[tokens,4,5120]`, FP32 function weight
  `[24,20480]`, scale `[3]`, base `[24]`; writes FP32 pre/post `[tokens,4]`
  and combination matrix `[tokens,4,4]`. The projection is RMS-normalized over
  all 20480 residual features with epsilon `1e-20`. Pre is sigmoid plus `1e-6`,
  post is twice sigmoid. Combination coefficients use row softmax plus `1e-6`,
  initial column normalization, then 19 row/column normalization pairs.
- `LaunchMhcPre`: explicitly supplied carried pre-mix collapses the four BF16
  streams to `[tokens,5120]`, accumulating FP32 and rounding back to BF16.
  Sublayer RMSNorm is a separate required operation, not folded into this API.
- `LaunchMhcPost`: combines the BF16 sublayer output with the original residual,
  using `comb[source,target]` (not its transpose) and the current post-mix,
  producing BF16 `[tokens,4,5120]`.
- `LaunchMhcInitialPre`: initializes each token's FP32 pre-mix to `[1,0,0,0]`.

Launches require 1–4096 tokens, exact buffer lengths/alignment, an explicit stream
and disjoint writable buffers. They reuse the contiguous device-region contract
from the Engram primitives but do not invoke Engram. The three arithmetic
launches report invalid inputs as device-error bit 1 and non-finite arithmetic or
BF16 overflow as bit 2; callers must zero the flag, retain resources and establish
completion before consuming outputs. No allocation, synchronization, Python
fallback or hidden reuse of the old mHC implementation occurs.

The FP32 projection/reductions are initial functional kernels, not optimized
tensor-core implementations. Their floating-point reduction order still requires
reference numerical qualification. The host launch contract and CUDA host/SM90
device compatibility syntax checks passed; NVCC linking, execution and model
integration remain outstanding.

## Native text RMSNorm and sublayer input chain

`pih_deepseek_v41_norm_cuda` implements the text model's RMSNorm widths 128
(index key), 512 (KV/compressor), 1280 (query low-rank state) and 5120 (hidden
state). `LaunchRmsNorm` accepts contiguous BF16 `[rows,width]` input/output and
one `[width]` weight vector whose BF16/F32 storage must be explicit. It reduces
the square mean in FP32, applies `rsqrt(mean + 1e-20)`, multiplies the normalized
value by the weight in FP32 and rounds to BF16. It is not vision LayerNorm or
an arbitrary-width normalization fallback.

The contract accepts 1–4096 rows, exact byte lengths and aligned, non-overlapping
outputs/error storage. Non-finite inputs/weights set device-error bit 1;
non-finite reductions/results or BF16 overflow set bit 2. It enqueues on an
explicit caller stream without allocation or host synchronization. The caller
must retain buffers and establish completion/error admission before consumption.

`LaunchMhcInput` connects an explicitly supplied carried pre-mix collapse to
5120-wide RMSNorm. It validates matching row counts/streams/error flags,
the intermediate buffer and whole-chain live-buffer aliases before launching.
The BF16 intermediate is intentional: normalizing the FP32 collapse directly
would change the reference's cast order. `pih_deepseek_v41_mhc_cuda` links the
independent normalization target, not the old model implementation. These
primitives pass SDK compatibility syntax checks but still need NVCC/linking,
numerical qualification and attention/FFN engine integration.

## Native text RoPE/YaRN primitives

`pih_deepseek_v41_rope_cuda` supplies per-chunk FP32 complex phases and BF16
rotation for the frozen text configuration. `LaunchRopeTable` accepts explicit
U32 original positions `[tokens]` and writes interleaved cosine/sine
`[tokens,32,2]`. Layers 0/1 and MTP 40–42 use base 10000 without YaRN. Backbone
layers 2–39 use base 160000, original length 65536, factor 16, beta-fast 32 and
beta-slow 1. Host double-precision correction bounds and FP32 frequency/ramp
arithmetic follow the reference formula. No full-million-position table or
Python precomputation is required by this entry point.

Positions must be below 1048576. For compressed KV/index keys, supply the first
original position of the group (for example 0,2,4 for ratio-two groups), not
the compressed cache slot. Queries and inverse-rotated attention outputs use
their own original query positions. Invalid positions set the shared device
error flag; the resulting table must not be consumed as a successful result.
The execution planner still needs to establish these position/role bindings.

`LaunchRopeApply` accepts BF16 `[tokens,heads,width]` with widths 128 or 512,
1–64 heads and 1–4096 tokens. It rotates the final 64 values as adjacent complex
pairs, shares each token's phase across its heads and copies the non-rotary
prefix unchanged. `inverse=true` conjugates the phase for attention output
rotation. Exact in-place operation is allowed; partial overlap and aliases with
phase/error outputs are rejected. Invalid tail/phase values set error bit 1;
non-finite results/BF16 overflow set bit 2. Prefix bytes are copied, not normalized.

These kernels are not vision RoPE, an attention/cache implementation or an
automatic compressed-position mapper. Trigonometric/reduction differences still
require numerical qualification against the reference; host/SM90 compatibility
syntax checks alone do not prove exact numerical agreement or B300 support.

`FlashConfig::Parse` accepts the frozen nested HF configuration at the reference
revision `517ef625df97ec57aadc91b67506a57c20fdc5bb`. It bounds input to 1 MiB,
depth 8, 1,024 JSON nodes and 4,096-byte strings; duplicate/unknown fields,
wrong field types, V4-0731 geometry and changed execution semantics fail. Object
ordering and whitespace do not matter. The configuration includes quantization,
text/vision geometry, YaRN, mHC, routed/shared experts, Engram and DSpark fields.
Its raw-byte SHA-256 is returned as an observation, not proof of revision origin
or weight authenticity. The reduced reference inference `config.json` is not the
nested HF format and is deliberately not accepted here.

The returned 43-entry attention sharing plan covers 40 backbone and three MTP
layers. It derives the latest KV source (2, 8, 14, 20) and index source
(2, 8, 14, 20, 24, 28, 32, 36), candidate production at layer 20, candidate use
by later index owners, Engram at layers 1/14 and distinct backbone/MTP expert
counts. Sliding-window-only layers carry no compressed-cache owners. This is
not a complete execution graph, cache allocation plan, TP plan or capacity claim.

`PIH_BUILD_ARTIFACT_TOOLS=ON` includes the CPU-only
`pih_deepseek_v41_config` static target. It links native model-support primitives,
not Python, CUDA or the monolithic inference archive. The fixed metadata matches
the vendored HF configuration in a static comparison; C++ syntax checks pass.
No config executable, model tests, B300 execution or numerical qualification has
been run. Native CED/CSA2/Engram/vision kernels, weight admission and execution
providers remain to be implemented before a runnable V4.1 plugin can be offered.

## Native Engram hash state

`EngramHashState::Create` takes a parsed `FlashConfig`, exactly 129,280 U32
little-endian compressed-token-map entries and their independently admitted
SHA-256. Entries must use the reference's dense first-appearance ID order and
cover all 99,092 compressed IDs. The mapped raw pad token (ID 2) supplies padding.
This does not generate/validate Unicode normalization semantics: NFKC/NFD,
accent stripping, lowercase, whitespace rules and partial-UTF8 handling still
need native tokenizer integration. Never invent a map/root just to pass admission.

The implementation allocates 48 globally unique prime buckets in reference
layer/ngram/head order, verifies row totals 384,006,168 and 384,016,682, and freezes
the per-layer hash multipliers derived from the reference PCG64 seed formula.
Those constants were generated offline with NumPy 2.5.3; the runtime imports no
NumPy, Torch or Python. Integer bounds are compile-time checked to keep products
within signed 64-bit range before XOR and modulo.

Use one state per sequence. `Append(start_position, token_ids, participation)`
requires the exact current position, accepts 1–4,096 tokens per call and at most
1,048,576 across the sequence. An omitted mask means all tokens participate;
otherwise each mask byte must be 0/1. A zero-mask image/dead position blocks all
older lookbacks, including across chunk boundaries. Only three history entries
are retained; reset clears them and the position. All request validation and
output allocation happen before committing new history. Output is
`[token][layer 1 or 14][2/3/4-gram × eight heads]` U32 table indices.

Static prime-ledger derivation and C++ syntax have been checked, not model
execution or tokenizer equivalence. GPU qualification
and integration into V4.1 execution remain unfinished. The pinned Engram
implementation does not contain a convolution.

## Native Engram weight layout

`EngramWeightPlan::Create` admits layers 1/14, a rank and TP world size dividing
both 384 backbone and 128 MTP experts. This is only a layout constraint, not TP
execution support. The six exact matrix suffixes, dimensions, storage types and
byte lengths are exposed and checked by `Validate`, independent of input order.
Duplicates, missing/extra members, transposes and changed storage types fail.
Payload authenticity and scale contents are not established by metadata checks.

Only `embed.weight` and `embed.scale` are row-sharded: each rank reserves
`ceil(global_rows / world_size)` rows of 256 E4M3FN values and eight E8M0 scales.
The plan distinguishes valid rows from final-rank padding. `Locate` rejects IDs
beyond the real table and returns ownership plus a safe local gather index;
non-owner results must be zeroed before the BF16 sum collective. Converter
padding must use zero weights and numeric-one scales (E8M0 byte 127, not byte 1).

The FP8 projection is replicated: weight `[25600,6144]`, block-32 scale
`[800,192]`. Both gate matrices are replicated `[4,5120]`; callers explicitly
select BF16 or F32 storage because the reference constructs them in its active
default dtype. This does not assert the dtype of an unseen source checkpoint.
There is no padding writer or tensor loader in this layout component. Collective
execution is a separate opt-in component described below.

## CUDA lookup, projection and gate primitives

With `PIH_ENABLE_CUDA=ON`, the independent static target
`pih_deepseek_v41_engram_cuda` builds `engram_cuda.cu` and its launch validation.
It links core primitives and CUDA runtime only, not Python or the legacy model.
It uses the configured CUDA architectures; this does not qualify B300 execution
or register a kernel-pack capability. NVCC compilation/linking is not available
locally. The Clang compatibility syntax observation described below covers the
CUDA host/device source and completion adapter but is not NVCC qualification.

`LaunchEngramLookup` decodes E4M3FN table values and block-32 E8M0 scales to BF16,
writing `[tokens,24,256]` in the reference flattening order. Non-owned rows
contribute zero; out-of-table IDs and non-finite data set the device error flag.
For TP, the caller must complete a BF16 SUM collective before FP8 projection.
No collective, projection or automatic fallback is hidden inside the launcher.

`LaunchEngramProjection` explicitly enqueues block-32 activation quantization
followed by the replicated `[25600,6144]` FP8 weight projection. It consumes the
completed BF16 lookup/collective output and requires caller-owned E4M3FN
`[tokens,6144]` and E8M0 `[tokens,192]` scratch. Quantization uses the reference's
`1e-4` maximum-magnitude floor, power-of-two ceiling scale and E4M3FN rounding.
The tiled CUDA GEMM accumulates each 32-element dot in FP32, applies the activation
and weight scales in that order, accumulates across 192 groups and writes BF16
`[tokens,25600]`. Scratch, output and error flag must be mutually disjoint and
must not overlap any inputs. Launch failure after quantization may leave partial
scratch changes; there is no transactional rollback.

This is a functional scalar tiled CUDA GEMM, not a tensor-core-optimized kernel.
It has not been compiled with NVCC or numerically/performance-qualified here and
must not be advertised as production throughput or B300 support. Its grouping
follows the reference, but its reduction order can differ from tensor-core GEMM.

`LaunchEngramGate` consumes BF16 residuals `[tokens,4,5120]` and projected KV
`[tokens,25600]`: four keys followed by one shared value. It computes FP32
per-copy RMS normalization with epsilon `1e-20`, weighted dot, signed square root
with absolute clamp `1e-6`, sigmoid and BF16 residual output. Optional U8 mask
zero bypasses a token exactly; mask values above one flag invalid data. BF16 and
F32 gate weights are supported. Fused multiply-add contraction is disabled;
parallel reduction order still requires numerical qualification against the
reference and is not claimed bitwise identical.

All three APIs accept 1–4096 tokens and exact contiguous buffer sizes, require aligned
non-null addresses and an explicit stream, reject address wraparound and aliases
between outputs/error flag and inputs, and enqueue without synchronization.
The caller must admit allocations/device ownership, retain all buffers through
completion, zero the U32 error flag beforehand and inspect it after completion
before consuming results. Bit 1 means invalid data/ID/mask; bit 2 means
non-finite arithmetic or BF16 overflow. Successful enqueue does not prove a
successful computation. These primitives are not yet connected to model execution.

`LaunchEngramSingleRank` connects lookup → quantization/projection → gate on one
explicit stream. Before enqueuing any work it validates every stage, matching
token counts, intermediate buffers and error flag, plus whole-chain live-buffer
aliasing. It rejects TP greater than one instead of skipping the collective.
The caller still initializes/checks the shared error flag and owns completion;
an asynchronous error can invalidate the entire result, and later enqueue failure
does not undo earlier work. This callable Engram chain is not a full model engine.

## Independent NCCL reduction component

On Linux, `PIH_ENABLE_CUDA=ON` plus `PIH_BUILD_NATIVE_ENGRAM_NCCL=ON` builds
the model-owned `pih_deepseek_v41_engram_nccl` adapter and the independent
`pih.transport.nccl` provider. Keep the legacy `PIH_ENABLE_NCCL` aggregate
option off. The adapter no longer includes `nccl.h`, links NCCL or interprets a
raw NCCL communicator: its handle is opaque and collective calls cross
`transport.collective.v1`. Only the provider links NCCL 2.31.2. The transport
and TP scheduling C++ sources pass Linux-target syntax checks with pinned CUDA
and NCCL headers. Linux linking and multi-GPU execution remain unverified.

After lookup enqueue, `EngramReduction::Submit(lookup, communicator)` performs
in-place BF16 SUM over exactly `tokens * 6144` values. It checks the runtime NCCL
release, communicator readiness, rank/count, current device and output-pointer
device before submission. Rank IDs must use the same global token/hash order;
these local checks do not establish distributed agreement or Lock admission.
The communicator is borrowed, must have completed initialization, and must be
exclusively operated by its owner. Stream provenance and allocation extent still
require the runtime's resource admission. A nonblocking communicator is required
when the owner needs a bounded host-side submission deadline.

The move-only reduction object distinguishes pending enqueue, enqueued and
failed states. Poll `PollEnqueued()` under the owner's deadline until enqueued;
**do not submit projection or another communication operation while pending**.
Enqueued means ordered on the CUDA stream, not completed on the GPU. After that,
projection/gate may be enqueued on the same stream. The owner must continue
checking communicator errors through completion, retain all resources, and
abort the generation/communicator on errors or timeout. Destruction does not
cancel or destroy the borrowed communicator; failure never triggers a fallback.
This follows [NCCL's asynchronous-state contract](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html).

Distributed bootstrap, rank agreement, supervision, deadline/abort ownership and
model-engine wiring remain to be connected; this component alone does not make
the multi-rank service runnable.

## Tensor-parallel Engram stage scheduling

`EngramLaunch` describes the same lookup/projection/gate buffers for either rank
mode. `ValidateEngramChain` checks all stages, their connections and whole-chain
live-buffer aliasing. `LaunchEngramSingleRank` still rejects TP greater than one;
the common description does not remove the required collective.

`EngramTensorParallel::Start(launch, communicator, completion, deadline)` validates the full
chain and local communicator identity before enqueueing lookup and reduction.
The deadline is an absolute `std::chrono::steady_clock` time point. The owner
must use an initialized nonblocking communicator; deadline checks cannot preempt
a blocking NCCL host call. Start can fail after some work was enqueued, so failure
requires generation cleanup rather than immediate buffer release or blind retry.

Retain the returned move-only operation and poll `Advance()` under the scheduler:

- `kWaitingReduction`: NCCL has not confirmed stream submission; no projection or
  gate is submitted. Yield to other work and poll again within the deadline.
- `kWaitingCompletion`: reduction and projection/gate have been submitted once,
  followed by asynchronous device-error readback and a completion event. Repeated
  polls never replay compute. Output is not consumable yet.
- `kComplete`: the event completed, the read-back error flag was zero and NCCL
  remained successful at retirement, all observed within the deadline. This is
  local operation completion, not distributed service readiness or model quality.
- A returned error makes the operation terminal. Do not retry this object;
  notify the generation supervisor to abort all ranks and retire resources.

Before submitting compute, the operation rechecks communicator rank/device
identity (including when a different host thread polls) and the deadline.
`Advance()` monitors communicator errors while waiting and polls the completion
event without synchronizing the host. Do not consume output before `kComplete`. All
resources and the communicator remain borrowed; destroying or moving the object
does not synchronize, cancel, free buffers or destroy the communicator. Calls
must be serialized. The native scheduling C++ is syntax-checked, but its CUDA/
NCCL linking and execution are not yet qualified, and it is not engine wiring.

## Completion resources and output admission

Supply `EngramCompletionResources` with an exclusively borrowed CUDA event and
exactly four bytes of aligned pinned host memory. Preflight rejects an event
still in use and a non-pinned host slot. The owner must prevent concurrent event
re-recording, host-slot reuse and communicator reuse; handle checks alone cannot
prove lifetime ownership. The device error flag must be initialized to zero
before lookup, and all buffers remain live through completion or safe abort.

`EngramCompletion::Record(gate, resources)` appends a four-byte device-to-host
copy and then an event on the gate's stream. The event covers both compute and
error readback. `Poll()` returns pending without reading host memory while the
event is incomplete; CUDA errors and any nonzero device-error bits are terminal.
The TP pipeline uses this automatically. Single-rank callers can call it after
successful `LaunchEngramSingleRank`, then poll under their own deadline.

The completion object never synchronizes or destroys caller-owned resources.
A failed copy/event submission may leave work in flight; retain buffers and
retire or abort them through the resource owner. Completion proves neither
checkpoint authenticity nor correctness against the reference model. The CUDA
completion adapter has passed a Clang host syntax check against official CUDA
13.2.86 headers, but has not been linked or executed here.

## Reproducible CUDA compatibility syntax checks

The development tools can fetch checksum-pinned official NVIDIA wheel archives
and extract only headers/licenses into a new directory under `out`. No package
installation, package scripts, driver changes or model execution takes place.
CUDA runtime/CRT/CCCL 13.2.86 and cuRAND 10.4.2.66 are pinned in
`tools/cuda-syntax-headers.lock.json`. cuRAND is needed by Clang's automatic CUDA
wrapper, not by these Engram kernels. Failed extraction directories are retained
for inspection; existing destinations are never overwritten.

```sh
python tools/prepare_linux_syntax_headers.py
python tools/prepare_cuda_syntax_headers.py
python tools/check_engram_cuda_syntax.py --compiler clang++
```

Skip preparation for an already prepared directory with a matching receipt.
The CUDA header tree includes an empty `bin/` directory solely because Clang
requires it for header discovery; it contains no NVCC or other tool binaries.
The checker validates the CUDA header ledger, performs a host syntax check of
the completion adapter, and separately parses the kernel's host and SM90 device
sides with no libdevice linking. It emits
`out/engram-cuda-syntax-report.json` with commands, source/header hashes, warnings
and explicit qualification limits.

Clang 22 warns that these CUDA 13.2 headers are newer than its partially supported
SDK level. Device parsing explicitly selects its 12.9 SDK feature mode, matching
the driver's host compatibility choice; the warning is retained. Consequently,
passing this check means **compatibility syntax only**, not NVCC 13.2 compilation,
linking, B300/SM103 support, numerical correctness, performance or model tests.
The NCCL adapter is checked only when its separately pinned headers are supplied:

```sh
python tools/prepare_cuda_syntax_headers.py --nccl
python tools/check_engram_cuda_syntax.py --compiler clang++ \
  --nccl-headers out/nccl-syntax-headers-2.31.2
```

NCCL preparation downloads the approximately 252 MB official
`nvidia-nccl-cu13==2.31.2` archive, checks its locked size/SHA-256, and extracts
only headers/licenses. No shared library is installed or loaded. The extended
check covers the reduction adapter and TP scheduler against the real headers,
in addition to the Engram, mHC, RMSNorm and RoPE CUDA compatibility checks. The emitted report records
the NCCL header version and each check separately; all sixty-one syntax checks passed
locally. This still provides no NCCL linker, communicator bootstrap, timeout,
multi-rank agreement or execution evidence.

## Ordinary attention-step index planning

`BuildAttentionStep(config, layer, start, tokens)` is the CPU-native planning
entry for backbone layers 0–39. It requires admitted frozen configuration,
prefill starting at zero with 1–4096 tokens, or single-token decode at a later
position, within the 1048576-position limit. Nonzero-start multi-token calls are
rejected: chunked prefill and DSpark/MTP blocks need their own execution semantics
and are not implicitly treated as decode here.

The plan contains original query positions, causal sliding-window indices and
ring writes. Prefill indices address the current complete KV tensor (not the
ring); at most its final 128 positions seed the ring at `position % 128`.
Decode indices address all 128 ring slots oldest-first with not-yet-populated
slots marked `-1`. This address-space distinction must survive device upload.

For compressed layers it includes end-of-step cache length and a separate causal
visible count for each query. Only KV owners emit new compressed-cache writes,
each with the group's first original position for RoPE. Ratio-two compressor
tail writes retain incomplete prefill groups or update the appropriate decode
slot; incomplete groups publish no new compressed position. Later top-k owners
can select from the same keys but do not become index-key owners:
`index_key_source` is the KV source, while `topk_source` is the latest selection
source. Mixing those ownership concepts would incorrectly create new key caches.

This is a pure plan: it does not advance sequence state, allocate device cache,
prove that preceding layers/steps completed, upload indices or execute attention.
The runtime must retain one consistent sequence position and only commit cache
progress after successful execution. No model inference was run to qualify it.

## Native sliding-window KV preparation

`pih_deepseek_v41_window_cuda` implements the ordinary prefill/decode window KV
update without the old model implementation. `LaunchWindowKv` takes mutable
BF16 `[tokens,512]` post-RoPE vectors and a BF16 `[128,512]` ring for one sequence.
Every 32-element group, including the rotary tail, undergoes E4M3FN quantization
with the `1e-4` magnitude floor and power-of-two ceiling scale, followed by FP32
dequantization and BF16 rounding. This matches the reference's quantized-value
simulation: the ring is BF16 storage, not a physically packed FP8 cache.

At zero-start prefill all vectors remain available in the mutable KV tensor for
causal attention; only the final `min(tokens,128)` vectors seed the ring at their
absolute-position modulo slots. Later single-token decode updates `start % 128`.
Other slots are left untouched. The attention plan's `-1` entries must mask slots
not yet populated; this launcher does not clear a newly allocated ring or admit
sequence continuity. Nonzero-start multi-token updates are rejected.

`LaunchWindowKvPrepare` connects 512-wide RMSNorm, forward tail-64 RoPE and the
quantization/cache update on one stream. It checks all stage layouts/connections
and whole-chain aliases before submitting any work. RoPE may reuse the norm
output exactly in place; other overlapping writable regions are rejected. The
caller supplies the already-computed KV projection and correctly bound phases.

Device-error bits retain the shared convention: 1 for non-finite input and 2 for
non-finite arithmetic/BF16 overflow. Both the temporary KV tensor and ring are
mutable; failures may leave partial changes. The owner must wait for completion,
inspect the error flag and poison/retire failed state rather than reuse it or
retry blindly. Cache allocation, cross-layer scheduling, sparse attention and
end-to-end model qualification remain outstanding.

## Native sparse attention primitive

`pih_deepseek_v41_attention_cuda` supplies `LaunchSparseAttention` for one
sequence: BF16 query/output `[tokens,heads,512]`, shared BF16 KV `[kv_rows,512]`,
FP32 sink `[heads]` and I32 gather indices `[tokens,picks]`. It accepts 1–4096
queries, 1–64 local heads and 1–640 picks (the window plus compressed selection).
KV is used as both key and value. The caller supplies the concatenated KV address
space and causal indices; the kernel does not produce top-k choices or establish
cross-layer cache ownership.

The implementation processes indices in groups of 64 with an FP32 online
maximum and denominator, scale `512^-0.5`, and BF16-rounded probabilities for
value accumulation. This last rounding is intentional and separate from the
FP32 denominator. The learned sink contributes to the denominator only. All
`-1` rows yield zero output; other negative or out-of-range indices set the
device error flag and are never dereferenced. Repeated valid indices retain
their repeated contributions, as in the supplied reference index sequence.

Invalid input values/indices set error bit 1; non-finite score/accumulator/output
arithmetic sets bit 2. A finite dominant sink may overflow its exponential to
positive infinity and produce zero output, matching the reference convention;
that denominator case is not by itself treated as invalid. Outputs/error storage
must not alias inputs. The caller must initialize the error flag and establish
completion before consuming output. Query/output inverse RoPE and output
projections remain separate required stages.

This is a scalar cooperative CUDA implementation, not a tensor-core-optimized
sparse-attention kernel. It preserves blockwise probability rounding but its
dot-product/reduction order still needs numerical qualification. SDK host/SM90
compatibility syntax checks passed; NVCC linking, latency/throughput, numerical
comparison and full model execution are unverified.

## Attention inverse rotation and grouped output projection

`pih_deepseek_v41_attention_output_cuda` implements the BF16 block-diagonal
`wo_a` projection and connects it to sparse attention. `LaunchGroupedOutput`
takes `[tokens,groups,4096]` input, `[groups,1024,4096]` BF16 weights and writes
`[tokens,groups,1024]` BF16 output, with FP32 accumulation. Each group contains
exactly eight 512-wide heads; local group counts are 1/2/4/8, corresponding to
the frozen eight-group layout. No weights from another group participate in
the local dot product. The converter/weight loader must supply already-dequantized
BF16 `wo_a` weights, as required by the reference runtime.

`LaunchAttentionOutput` submits sparse attention, inverse tail-64 RoPE and
`wo_a` on one stream after validating all stage shapes, head/group relationships,
intermediate connections and whole-chain aliases. Inverse RoPE can reuse the
attention output exactly in place; the grouped projection output is separate.
All error flags are shared and must be zeroed/admitted by the caller. Errors may
leave queued work or modified intermediates; completion/error checking remains
required before consuming the low-rank result.

This output is **not** the final 5120-wide hidden state. The FP8 `wo_b` row-parallel
projection and TP SUM are still required. The tiled scalar BF16 implementation
is not tensor-core optimized or performance-qualified. Its SDK compatibility
syntax passes; numerical comparison, NVCC linking and model execution remain
unverified, and the component registers no inference capability.

## Shared FP8 linear projection and local attention output

`pih_deepseek_v41_fp8_cuda` now owns the block-32 activation quantizer and
scale-corrected GEMM used by both Engram and attention output. The Engram adapter
retains its exact `[25600,6144]` admission but calls the common implementation;
its duplicate specialized CUDA projection was removed.

`LaunchFp8Linear` accepts BF16 input `[rows,K]`, E4M3FN weights `[N,K]`, E8M0
weight scales `[N/32,K/32]`, caller-owned E4M3FN/E8M0 activation scratch and BF16
output `[rows,N]`. It admits 1–4096 rows and positive multiples of 32 for K/N,
each no larger than 32768, with exact byte lengths and disjoint writable storage.
These are primitive bounds, not authorization for an arbitrary model shape:
model/weight admission must establish the actual named tensor dimensions.

`LaunchAttentionLocalOutput` extends the grouped output chain with FP8 `wo_b`:
K is `local_groups * 1024`, N is 5120, and all stages share the same token count,
stream and device error flag. The full live-buffer graph is checked before any
stage is submitted. The resulting BF16 `[tokens,5120]` is a **rank-local
contribution**; when local groups are fewer than eight, a TP SUM is required
before mHC residual expansion. This entry does not silently omit or implement
that collective. With all eight groups on one rank there is no inter-rank sum.

The shared kernel retains the FP32 blockwise scale order and BF16 output
rounding of the earlier Engram primitive. It remains a scalar tiled kernel,
not a tensor-core throughput implementation. SDK compatibility syntax checks
pass, but numerical equivalence, NVCC linking, distributed reduction and complete
model execution are not established by those checks.

## Attention query and window input chains

`pih_deepseek_v41_attention_input` connects the native primitives into the
reference's input-side sequences, without allocating buffers or invoking old
model code. `LaunchAttentionQuery` performs FP8 `wq_a` (5120 → 1280), RMSNorm,
FP8 `wq_b` (1280 → local_heads × 512), and forward tail RoPE. The normalized
1280-wide `qr` buffer stays separate and available for the indexer. Local head
counts are 8/16/32/64, consistent with the eight-group attention layout at TP
8/4/2/1; standalone Engram partition geometry does not imply wider model TP support.

`LaunchAttentionWindow` performs the replicated FP8 `wkv` (5120 → 512), followed
by the already implemented window normalization, RoPE, quantize/dequantize and
ring update. The caller supplies layer/query-position-bound phase tables and a
sequence-owned cache. The query and window APIs each preflight every stage and
the entire live-buffer graph before submission. Only the exact RoPE in-place
reuse is allowed; other scratch/output overlap with still-needed tensors fails.

Both APIs use one explicit stream and one shared device-error flag per chain.
They enqueue work only; failure may leave modified scratch or cache, and successful
enqueue does not authorize output consumption without completion/error checking.
The chains are not yet a complete model engine: compressed KV/index selection,
cross-chain scheduling/ownership, TP output reduction and full release qualification
remain to be connected.

## Native compressor projection, pooling and state

`pih_deepseek_v41_compressor_pool_cuda` implements the pre-RoPE compressor,
independently of the old runtime and the Python reference. `LaunchCompressor`
preflights and submits exactly one of two paths from BF16 `[tokens,5120]` hidden
state to normalized BF16 `[emitted_rows,512]` latent:

- Ratio 1: BF16 `[512,5120]` value weight, FP32 accumulation, BF16 projection
  output and RMSNorm. Gate weights, scores and the pooled branch must be absent.
- Ratio 2: FP32 value and gate weights `[512,5120]`, BF16 hidden values promoted
  to FP32, FP32 projected values/scores, per-feature two-token softmax pooling,
  BF16 rounding and RMSNorm. FP32 weights must already be admitted/promoted by
  the weight loader; the launch does not reinterpret BF16 bytes as FP32.

`LaunchCompressorProjection`, `LaunchCompressorPool` and
`LaunchCompressorNormalize` also expose the individual operations. Projection
uses a scalar shared-memory tiled CUDA kernel, not TF32 or tensor cores; it is
not throughput-qualified and numerical reduction order still needs validation.

Ratio-two state consists of separate FP32 `[2,512]` value and score buffers owned
by one sequence/source. An initial prefill starts at zero with 1–4096 tokens;
subsequent steps accept exactly one token, with positions below 1048576. Full
prefill pairs emit rows; an odd tail writes state slot zero. Decode writes slot
`start % 2`, emitting a row only when slot one completes the pair. An incomplete
step requires an absent output `{0,0}` and no RMSNorm launch. Complete steps
require matching output rows, width 512, stream and error flag. The state is
not an independently resumable checkpoint: the caller must enforce contiguous
positions, sequence identity and stream ordering, and must never start decoding
from uninitialized or foreign state. Unused slots need not be read or cleared.

The full chain rejects overlap among projections, state, pooled output,
normalized output, error storage and live input/weight tensors before submission.
Device error bit 1 reports non-finite inputs, and bit 2 reports non-finite
arithmetic or BF16 overflow. Calls enqueue only; failure can leave partial state,
so the owner must retain resources and retire a failed sequence. Successful
enqueue is not completion or permission to consume output.

The normalized latent deliberately remains unrotated for index-key projection.
Compressed-position RoPE and simulated FP4 cache writes are available through
the separate preparation primitive below; index selection and complete model
state admission/scheduling remain outstanding. Compatibility syntax checks do
not establish NVCC compilation, GPU execution, numerical equivalence or B300
support. No runnable model capability is added by this component.

## Compressed KV preparation and cache writes

`pih_deepseek_v41_compressed_kv_cuda` implements `LaunchCompressedKvPrepare`:
forward RoPE on the last 64 features of normalized BF16 `[rows,512]` latent,
followed by group-16 FP4 quantize/dequantize and a contiguous cache write. The
normalized unrotated input must be separate from the rotated output, preserving
it for index-key projection. Phases must represent each compressed group's
first original token position, not its cache index. The attention-step plan
provides those positions; the caller must bind the matching layer/phase table.

For each 16-element group, the maximum magnitude is floored at `6 * 2^-9`.
The scale is `amax / 6` rounded to E4M3FN with non-saturating overflow. Values
are divided by that scale, clamped to `[-6,6]`, rounded to E2M1 nearest-even,
multiplied by the rounded scale, and rounded back to BF16. All 512 features,
including the rotated tail, are quantized. This matches the reference's
**BF16 cache containing simulated FP4 values**, not physically packed FP4
storage; no packed-storage memory saving is claimed. This differs from the
window path's group-32 E4M3 values/E8M0 scales and the indexer's FP4/E8M0 format.

`LaunchCompressedKv` exposes the post-RoPE operation directly. It accepts
1–4096 rows, a capacity of 1–1048576 rows and an in-bounds `first_slot` write.
The exact-size BF16 cache, mutable input and device error flag cannot overlap.
Preparation additionally rejects writes overlapping the original latent or
phase table. An incomplete compressor step emits no rows and must not invoke
this operation. The owner supplies capacity, source/sequence identity and
`first_slot` from its admitted step; bounds alone do not establish causal cache
validity, contiguous progression or permission to read unwritten cache rows.

The launch allocates nothing and does not synchronize. Non-finite input sets
error bit 1; invalid scale or arithmetic sets bit 2. Failed work can modify both
scratch and cache; outputs require completion/error admission and failed state
must be retired. CUDA host/device compatibility syntax is checked, but numerical
comparison, native NVCC linking, hardware execution and complete model/cache
ownership integration remain unqualified.

## Indexer query projection and quantization

`pih_deepseek_v41_indexer_input_cuda` provides `LaunchIndexerQuery`: FP8
projection of normalized BF16 query low-rank state `[tokens,1280]` to
`[tokens,local_heads,128]`, forward tail RoPE, and in-place simulated FP4
quantization. Local head counts are 4/8/16/32 for TP 8/4/2/1, derived from the
frozen configuration's 32 index heads, not the reference dataclass defaults.
The phase table must correspond to the original query positions and layer.

`LaunchIndexerQuantize` operates on BF16 `[tokens,heads,128]` with 1–4096
tokens; heads=1 is additionally available for index keys. Each 32-element
group uses a maximum floor of `6 * 2^-126`, followed by power-of-two ceiling
of `amax * (1/6)` (E8M0 scale semantics), E2M1 nearest-even quantization and
BF16 dequantized storage. It quantizes all 128 dimensions, including the
64-dimensional rotated tail. The E2M1 rounding helper is shared with compressed
KV, while the scale format and group size remain distinct. Neither path claims
physically packed FP4 cache storage.

The query chain validates projection geometry, row/head counts, stream, shared
error flag and the entire live-buffer graph before launching. Exact projection
output/RoPE output reuse is allowed; partial overlap or overwriting normalized
query state, phases or weights is rejected. Error bits 1/2 retain the existing
invalid-input/arithmetic meanings, and completion/lifetime admission remains
the caller's responsibility. This is not yet the full indexer: distributed
score reduction, candidate
selection and top-k integration remain outstanding. Compatibility syntax
checking does not establish numerical or hardware qualification.

## Index-key projection and cache preparation

The same native indexer target now provides `LaunchIndexerKey`: BF16
`[rows,512]` unrotated compressor latent is projected with BF16 `[128,512]`
`wk`, accumulated in FP32 and rounded to BF16. Width-128 RMSNorm, forward
tail RoPE and group-32 FP4/E8M0 quantize/dequantize follow, before an asynchronous
device-to-device copy into BF16 index-key cache `[capacity,128]`. The original
latent stays read-only for the compressed attention KV path. Projection is an
initial scalar tiled implementation, not a tensor-core performance claim.

`LaunchIndexerKeyProjection` and `LaunchIndexerKeyCache` expose the standalone
stages. The cache stage requires exactly one key head, 1–4096 rows and an
in-bounds contiguous write starting at `first_slot`, with capacity at most
1048576. Empty compressor outputs skip the entire key chain. The model owner
must establish that this layer owns the index keys, bind the shared KV source,
and supply compressed-group-first-position phases and the correct write range.
Cache bounds are not proof of source identity or causal visibility.

All stage dimensions, buffers, streams and error flags are checked before any
work is submitted. Exact normalization-output/RoPE-output reuse is permitted;
other writable overlaps and writes into original latent, weights or phases are
rejected. Cache transfer is ordered after quantization on the same explicit
stream. It is not a transactional publication: device-error-marked values can
already have been copied when completion is observed. The owner must admit
completion/error status before exposing the new range and retire failed state.
Distributed ownership/publication and complete indexer scheduling remain open.

## Indexer head weights and rank-local scoring

`pih_deepseek_v41_indexer_score_cuda` implements `LaunchIndexerWeights` and
`LaunchIndexerScore`, also connected by `LaunchIndexerWeightedScore`. Head
weights use BF16 hidden `[tokens,5120]` and BF16 `[local_heads,5120]` projection
weights, FP32 accumulation, BF16 projection rounding, then multiplication by
`128^-0.5 * 32^-0.5 = 1/64` and another BF16 rounding. The scale uses the
global 32 index heads, not the local head count.

Scoring accepts BF16 quantized/dequantized queries `[tokens,local_heads,128]`,
BF16 shared keys `[positions,128]` and BF16 head weights. Each query/key dot is
accumulated in FP32 then rounded to BF16, rectified and multiplied by the head
weight with another BF16 rounding. Head contributions are accumulated in FP32
and the final `[tokens,positions]` score is BF16. No softmax is applied. These
rounding boundaries follow the reference tensor operations; reduction order
still requires numerical qualification. The cooperative warp implementation
is functional code, not a tensor-core or throughput-qualified implementation.

The API admits 1–4096 query tokens, 4/8/16/32 local heads and 1–1048576 key
positions, exact buffer sizes, explicit stream and disjoint writable storage.
The connected path additionally checks matching heads/tokens/stream/error flag
and all live inputs. Zero visible keys skip scoring entirely. Large score
matrices can consume substantial memory; shape acceptance is not a memory
reservation or proof that the device has enough capacity.

Scores are **rank-local and unmasked**. TP>1 must sum BF16 scores across ranks
before causal masking, candidate selection or top-k; this target neither
performs nor silently omits that collective in a claimed complete-model path.
Only committed key rows may be supplied by the runtime. Device errors retain
the invalid-input/arithmetic bits and require completion admission. Distributed
reduction, selection and full scheduler integration remain outstanding; syntax
checks alone do not establish GPU execution or model correctness.

## Causal index top-k selection

`pih_deepseek_v41_indexer_select_cuda` provides `LaunchIndexerSelect` for the
ordinary or candidate-masked selection path. Input scores must already be
globally reduced BF16 `[tokens,positions]`, where
`positions = (start + tokens) / ratio` and ratio is 1 or 2. Prefill starts at
zero with 1–4096 tokens; nonzero-start decode accepts one token. Each query
can see only `(start + query_index + 1) / ratio` compressed positions.

The kernel applies causal negative-infinity masking while retaining the best
`min(512,positions)` scores, then sorts retained indices by position. Visible
indices receive the caller-supplied concatenated-KV offset (0–4096); invisible
fillers become `-1`. Equal scores deterministically prefer smaller positions.
The reference's unspecified top-k tie choices need not be bit-identical, but
non-tied selection, causal visibility and output ordering follow the reference.
With no compressed positions, scores/output must be absent and no kernel runs.

The initial implementation uses one bounded 512-entry heap per query and no
full-position scratch allocation. Selection is serial within each query;
long-context throughput remains unqualified and requires optimization before
a performance claim. Exact input/output/error extents and disjointness are
checked. Non-finite raw scores set device-error bit 1; the caller must reject
that generation after completion even if an output index was produced.

This entry does not perform TP reduction. An optional U8 candidate mask is
accepted only for ratio-one layers, with exact `[tokens,positions]` size and
no overlap with output/error storage. Values other than 0/1 set the device
error flag. Masked positions receive negative infinity before selection and
cannot become output indices; underfilled externally supplied masks produce
`-1` fillers. Complete model ownership, score completion and cache admission
remain runtime responsibilities.

## Two-level candidate blocks

`LaunchIndexerCandidates` generates the candidate source's U8
`[tokens,positions]` mask from globally reduced, unmasked BF16 scores. It admits
the frozen source's ratio-one geometry (`positions=start+tokens`) and the same
prefill/single-token-decode bounds. Each group of eight positions is scored by
the maximum over causally visible entries, with unavailable entries at negative
infinity. The block containing the newest visible position is pinned at positive
infinity, and at most 2048 blocks are retained. Ties prefer lower block indices.
Unreachable negative-infinity filler blocks are dropped; retained blocks expand
to eight mask bits, clipped to the stored row width.

The last retained block may mark future positions in prefill. This is intentional:
the consumer applies its own causal visibility in addition to the candidate mask.
The source's own top-k uses no candidate mask; later candidate-consuming layers
pass this source mask with their own scores to `LaunchIndexerSelect`. Finite,
source-generated masks have enough reachable positions for the frozen top-512
selection. Arbitrary underfilled masks fail closed to `-1` output fillers rather
than returning excluded positions.

The producer uses a bounded 2048-entry heap per query, shares selection ordering
code with top-k, and does not allocate per-block scratch. It remains serial
within each query and is not throughput-qualified. Buffer shape and aliases
are checked; non-finite raw scores set device-error bit 1. Runtime must bind
the correct candidate-source layer, step, source score completion and mask
lifetime. These launch primitives do not establish cross-layer identity or
perform TP reduction, and no complete-model execution claim is added.

## Index-score NCCL reduction

With `PIH_BUILD_NATIVE_ENGRAM_NCCL=ON`, `pih_deepseek_v41_indexer_nccl`
provides `ValidateIndexerReduction` and `ReduceIndexerScore`. The latter accepts
the validated scoring launch, expected rank and a borrowed communicator. TP
size is `32 / local_heads`; multi-rank reduction supports local heads 4/8/16
(TP 8/4/2). Local heads 32 require no collective and are rejected by this
multi-rank API. It performs in-place BF16 SUM over the score output before
causal masking or candidate/top-k work.

The common BF16 submission path is `EngramReduction::SubmitBf16` in the existing
native transport component; Engram lookup now also uses it. This retains one
NCCL implementation rather than duplicating async handling. The generic buffer
contract admits aligned, nonempty BF16 extents up to `4096 * 1048576` elements;
the model-specific entry additionally checks exact score dimensions. Expected
rank/size, current CUDA device, output pointer device type, idle communicator
and exact NCCL 2.31.2 runtime are checked before submission.

Call only after successful score enqueue on the same explicit stream. Keep the
communicator exclusively owned and all buffers alive. Poll the returned borrowed
operation until `kEnqueued` before submitting further CUDA/NCCL work on that
stream; `kEnqueued` is not GPU completion. The owner must enforce deadlines,
abort failed generations, admit device-error/completion state and ensure all
ranks agree on score shape, sequence and collective order. Pointer attributes
do not authenticate allocation extents or distributed agreement. This primitive
does not bootstrap communication. The state machine below connects query/key
preparation, scoring and selection; cross-layer scheduling remains separate.

## Multi-rank indexer state machine

`IndexerTensorParallel::Start` accepts an admitted `FlashConfig`, backbone layer,
an `IndexerPipelineLaunch`, expected rank,
borrowed nonblocking communicator, pinned four-byte error readback slot, exclusive
CUDA event and steady-clock deadline. It validates the query/weighted-score/selection
connection and every live-buffer overlap before scoring. Optional candidate-source
generation must share the same scores, ratio-one step, stream and error flag;
source selection must be unmasked. Consumers instead provide an existing source
mask to selection. This object supports TP 2/4/8, not the single-rank no-collective
path or zero-position no-score steps.

Start submits optional index-key preparation/cache update, query preparation,
head-weight projection and local scores, then NCCL SUM. Non-waiting
`Advance` polls enqueue status before submitting candidates/top-k exactly once,
records shared error readback and a completion event, and polls until both GPU
completion and a zero error flag are observed. It rechecks communicator status
at retirement and enforces the deadline before authorizing `kComplete`. Pending
NCCL after selection was submitted is treated as unauthorized communicator reuse.
Failed or moved-from objects cannot replay work; repeated completed polls return
the completed state without new submissions.

The completion component now exposes `EngramCompletion::RecordFlag` for a generic
aligned device error flag and explicit stream; the Engram gate-specific entry
delegates to it. No synthetic Engram gate or duplicate completion implementation
is needed for indexer output. All resources remain caller-owned: partial enqueue,
timeouts and failures require generation-wide abort and safe resource retirement,
not immediate buffer reuse. Input readiness, phase/source identity, distributed
agreement and communicator bootstrap remain external obligations. This chain is
not a complete runnable model plugin or hardware-qualified inference engine.

`IndexerPipelineLaunch` requires the query projection/RoPE/quantization chain,
bound to the scoring query buffer, local head count, rows, stream and shared
error flag. The optional key chain is supplied only when an index-key owner
emits new compressed rows. Its write starts at `start / ratio`, emits exactly
`positions - start / ratio` rows, and targets the same cache base whose populated
prefix is read by scoring. Cache capacity must cover that prefix. An incomplete
group must omit the key chain; non-owning layers also omit it and read their
admitted source cache. Layer ownership itself still needs model-level admission.

The whole-graph alias check includes query quantization scratch, original
normalized query state, compressor latent, all weights/phases, key projection
and normalization scratch, the complete writable key cache and score/selection
outputs. Only the explicitly permitted exact RoPE in-place buffers are merged;
partial aliases or scratch reuse across independent stages are rejected. The
new key cache is an internal producer/consumer connection, not a generic bypass
for overlapping inputs. The prior cache prefix must already belong to this
sequence and be ready on the stream; this check cannot prove its contents.

## Single-rank indexer execution without NCCL

`pih_deepseek_v41_indexer_chain` now owns the common launch structure and
whole-graph validation in `indexer_chain.h/.cpp`, separated from the multi-rank
state machine. Its build dependencies contain native CUDA primitives and error
completion but no NCCL target or library. Both execution paths use this one
validator; no duplicated single-rank interpretation of the graph is maintained.

`IndexerSingleRank::Start` requires admitted configuration/layer and all 32 index heads, and accepts the same
query/key/scoring/selection launch graph, pinned error slot, exclusive event and
deadline. It submits optional key updates, query preparation, head weights,
scores, optional source candidates and top-k in stream order, without any
collective. `Poll` returns false while pending and true only after completion
and zero device error, within the deadline. Failed and moved-from operations
cannot replay submissions; completed polls are cached.

The operation still borrows all resources and may have queued partial work when
Start fails. The owner must retain resources through safe retirement and enforce
sequence/cache/source identity. Zero-position steps remain a separate no-score
case for the enclosing attention scheduler; this entry does not invent dummy
keys. No NVCC link, numerical/model execution or GPU throughput qualification
is implied by host/device compatibility syntax checks.

## Frozen layer-role admission

Both indexer execution entry points call `ValidateIndexerLayer` before submitting
any work. A non-default parsed configuration and backbone layer below 40 are
required. Only actual top-k/indexer owners may run the computation; layers that
reuse another layer's selected indices must not duplicate it. Compression ratio
must match the frozen role. Concatenated-KV index offset is the full prefill
token count or 128 for decode, not the number of causal window columns.

Index-key preparation must be present exactly when the layer owns the KV/key
source and its step emits compressed rows. Candidate production must be present
at the candidate-source layer and absent elsewhere; consumers require a source
mask, while other indexers must not supply one. These rules prevent omitting
required work or enabling the wrong branch with otherwise valid tensor shapes.

Layer-role checks do not authenticate buffer provenance: the future sequence
resource owner must bind the actual key/cache/candidate source to that layer
and step, retain it across consumers and reject stale generations. Parsed config
geometry alone is not checkpoint authenticity. Zero-compressed-position steps,
shared-index consumers and complete backbone scheduling remain the enclosing
model runtime's responsibility; no runnable model capability is claimed here.

## Attention KV/index assembly

`pih_deepseek_v41_attention_assemble_cuda` implements `LaunchAttentionAssembly`
and its connection to sparse attention, `LaunchAssembledAttention`. It copies
BF16 window KV followed by the populated compressed prefix into the shared-KV
input and constructs the combined I32 index rows. Prefill window storage is the
full `[tokens,512]` step, whereas decode uses the `[128,512]` ring. Window index
columns are `min(tokens,128)` at prefill and 128 at decode, generated directly
from step geometry using the same sliding-window/ring order as the step planner.

Compressed cache rows are `(start+tokens)/ratio`, with at most 512 selected
indices per query. Supplied selected indices already include the window-storage
offset; each nonnegative entry must address a causally visible compressed row,
not a window row or future position. Violations set device-error bit 1 and write
`-1` so sparse attention cannot follow an invalid address. Existing `-1` fillers
remain masked. This guard does not prove that selected entries are the correct
top-k or candidate/source generation; those remain upstream obligations.

Ratio zero and an empty compressed prefix require absent compressed/selected
regions and still execute window-only attention. All exact dimensions, stream,
output connections and whole-chain buffer aliases are checked before enqueue.
Cache copies and index construction are ordered before attention on that stream.
Inputs are preserved, and failed work must be discarded even when invalid
indices were safely masked. No allocator, synchronization or Python fallback
is introduced. Full-prefix copying follows the current reference storage model;
it is not a long-context bandwidth optimization or a packed-cache claim.

The caller must supply normalized/rotated/quantized KV and query data, bind the
correct layer sources and admit completion before consuming output. This
assembly stage is not the complete attention scheduler or a runnable model.

`LaunchAssembledAttentionOutput` extends this path through inverse tail RoPE,
BF16 grouped `wo_a`, and FP8 `wo_b`. Its `AssembledAttentionOutputLaunch`
connects the assembly buffers directly to the existing local-output chain and
validates every stage plus the complete live-buffer graph before submission.
The only permitted intermediate alias is exact attention-output/inverse-RoPE
output reuse. Assembly scratch cannot overwrite original caches, selected
indices, queries, phase tables or either projection's weights; later projection
scratch/output cannot overlap those inputs or the assembled KV/index storage.

The result is BF16 `[tokens,5120]` for this rank. When the eight attention groups
are partitioned across ranks, a SUM is still required before residual/mHC
expansion. This chain deliberately does not call it implicitly or claim the
local contribution is the final distributed result. Successful enqueue still
requires completion and shared device-error admission, and may leave modified
scratch on failure. Query/cache preparation, source ownership and complete-layer
scheduling remain separate work.

## Attention reduction and residual expansion

`pih_deepseek_v41_attention_nccl`, enabled by the same native NCCL option,
provides `ValidateAttentionReduction` and `ReduceAttentionOutput`. It validates
the local attention-output chain, derives TP size as `8 / local_groups`, and
submits in-place FP32 SUM over separate `[tokens,5120]` transport scratch through
the common NCCL operation. Local `wo_b` output is BF16, promoted to FP32 before
SUM, then rounded back to BF16 after collective enqueue and before mHC.
Local groups 1/2/4 imply TP 8/4/2; eight groups need no collective and are
rejected by the multi-rank API. Rank/device/version/lifetime and enqueue polling
obligations are the same as for index-score reduction. Submit only after `wo_b`
and FP32 promotion enqueue, and wait for `kEnqueued` before rounding back to
BF16 and residual expansion on that stream.

`pih_deepseek_v41_attention_residual` connects assembled local attention output
to `MhcPostLaunch`. `ValidateAttentionResidual` checks matching tokens, stream,
device-error flag and BF16 sublayer output, plus cross-stage aliases involving
original four-stream residuals, post/comb coefficients, cache inputs and all
attention scratch. `LaunchAttentionResidualSingleRank` admits only eight local
groups and launches assembly/attention/output projection followed by mHC. It
does not silently treat a partial rank contribution as the full sublayer output.

Coefficients and original residual state must be prepared for this exact layer;
their provenance is not established by pointer/shape checks. These operations
enqueue work only and require completion/error admission before use. The full
layer scheduling remains to be connected; the multi-rank output state machine
is described below. NVCC linking, numerical equivalence and hardware performance are
not proven by compatibility syntax checks.

`AttentionTensorParallel::Start` now connects assembled attention output,
NCCL SUM and mHC residual expansion in the native attention NCCL target. It
accepts the fully validated `AttentionResidualLaunch`, rank, borrowed nonblocking
communicator, pinned error slot, exclusive event and steady-clock deadline.
The complete attention/residual buffer graph, transport and completion resources
are admitted before submitting attention and reduction.

`Advance` polls reduction enqueue without waiting, rechecks rank/current device
before residual submission, and submits mHC plus error/event recording exactly
once. It returns complete only after the event and zero error flag, a final
communicator-state check and a deadline check. Communicator reuse while waiting
for completion is rejected. Failed/moved-from objects cannot replay work, and
completed polls do not submit anything. Timeouts and partial failures require
the owner to abort the generation and retain resources until safe retirement.

This state machine begins with prepared query, KV, selected indices and mHC
coefficients. It does not yet schedule their producers or authenticate their
layer/sequence identity, and does not create or destroy the communicator. It
is an output-to-residual operation, not a complete model execution loop.

## mHC sublayer input preparation

`MhcSublayerInputLaunch` connects current-residual mix generation and the
existing carried-pre collapse/RMSNorm chain. `LaunchMhcSublayerInput` first
produces the current sublayer's `pre`, `post` and `comb`, then collapses the
same residual using the explicitly supplied **previous sublayer's pre**, and
normalizes the BF16 collapsed result. The newly generated pre remains available
for the next sublayer; current post/comb are retained for the residual return.

`ValidateMhcSublayerInput` checks matching residuals, rows, streams and shared
device-error flag plus the entire live-buffer graph. In particular, new pre
cannot overwrite carried pre, and mix outputs cannot alias collapse/normalization
scratch or weights. The BF16 boundary between collapse and RMSNorm is retained.
The graph applies to attention and FFN with their respective weights; it does
not infer layer identity, initialize the first carried pre or substitute newly
generated coefficients for the carried ones. Submission remains asynchronous,
with caller-owned completion, failure retirement and coefficient lifetimes.

## Shared FP8 expert and clipped SwiGLU

`pih_deepseek_v41_shared_expert_cuda` implements `LaunchSharedExpert`: two
independent FP8 projections from BF16 `[rows,5120]` to `[rows,2304]`, clipped
SwiGLU, and FP8 down projection back to `[rows,5120]`. It reuses the block-32
FP8 linear path; this shared expert is not an FP4 routed expert. Full stage
geometry, stream/error connections and all writable/live-input overlaps are
checked before any work is submitted. Activation scratch is distinct from both
branch outputs, and the two projections currently use separate quantization
scratch. Shared-expert activation must not supply routing weights.

`LaunchExpertActivation` exposes the fixed 2304-wide activation independently.
BF16 gate/up outputs are promoted to FP32, gate is clamped only above at 10,
up is clamped to `[-10,10]`, and `silu(gate) * up` is computed in FP32. Optional
FP32 nonnegative per-row routing weights are multiplied before the BF16 cast
that precedes down projection. This preserves the reference weighting location;
weighting the final BF16 expert output would change the quantized path.

The contract accepts 1–4096 rows, exact disjoint buffers and an explicit stream.
Non-finite inputs or invalid routing weights set device-error bit 1, and
arithmetic/BF16 overflow sets bit 2. These enqueue-only calls require completion
admission and safe failure retirement. Routed FP4 projections, routing/dispatch,
expert accumulation/reduction and complete FFN integration remain outstanding.
No optimized tensor-core throughput or numerical/hardware qualification is claimed.

## Mixed FP8-activation / FP4-weight projection

`LaunchFp4Linear` in the shared projection target accepts BF16 `[M,K]` input,
packed E2M1 `[N,K/2]` weights, per-row E8M0 `[N,K/32]` weight scales, E4M3
activation scratch `[M,K]`, E8M0 activation scales `[M,K/32]`, and BF16 output.
Even K coordinates occupy the low nibble and odd coordinates the high nibble.
Unlike FP8 weight scales, FP4 scales are not shared by 32 output rows.

The reference uses **FP8 activations with FP4 weights**, not FP4 activations.
The implementation reuses the existing activation quantizer and templated
block-32 projection kernel, decoding E2M1 exactly to FP32, accumulating the
unscaled dot in FP32, then applying activation and per-row weight scales in
the same order as FP8 projection. All finite E2M1 values are exactly representable
in FP8, so the reference's intermediate FP4-to-FP8 cast does not change them.
E8M0 byte 255 is rejected through the device-error flag. Output rounds to BF16.

Admission requires 1–4096 rows and positive block-32 K/N dimensions up to 32768,
exact byte lengths and disjoint writable buffers. These are primitive bounds,
not permission to bind arbitrary model tensors. The FP4 contract and validator
are separate from FP8, while CUDA arithmetic is shared in `linear_fp8.cu`.
Numerical reduction order, tensor-core optimization, routed-expert chaining and
model execution remain to be qualified or connected; syntax success does not
establish NVCC linking or B300 support.

`RoutedExpertLaunch` and `LaunchRoutedExpert` now connect the two mixed
FP8-activation/FP4-weight gate/up projections, routed SwiGLU and mixed down
projection. Gathered input/output geometry is `[rows,5120]` with a 2304-wide
intermediate. An exact FP32 routing-weight row vector is required and remains
live through activation; its storage cannot be overwritten by projection or
activation scratch. Weighting is applied to the FP32 activated intermediate
before BF16 rounding and the down projection's FP8 activation quantization.

Shared and routed expert chains use one templated connection/liveness validator
with format-specific projection validators. Shared expert rejects routing
weights; routed expert requires them. The native expert target links both
projection formats from the shared CUDA projection target, without old-model
or Python execution. Each call handles one expert's already gathered tokens;
it does not select experts, authenticate expert/rank ownership, gather/scatter
token rows, accumulate expert outputs or perform the MoE collective. These
remaining dispatch/integration stages are not implied by successful syntax checks.

## Native MoE router

`pih_deepseek_v41_router_cuda` implements `LaunchRouter` from BF16 hidden
`[tokens,5120]` and pre-promoted FP32 gate weights. Backbone layers use 384
experts/top-6; layer indices 40–42 use the frozen 128/top-3 primitive geometry
without claiming a complete MTP/DSpark execution path. The caller supplies
FP32 text correction bias and, when an image mask is present, FP32 image bias.
The optional U8 mask must contain 0/1. Weight promotion/admission is the loader's
responsibility, not a reinterpretation of BF16 weight bytes.

The kernel computes FP32 logits at temperature one, then sqrt-softplus with
the reference's softplus linear branch above 20. Bias-adjusted scores choose
experts; routing weights use the **unbiased** scores of those experts, divided
by their sum plus `1e-20`, then multiplied by 1.5. Outputs are U32 expert IDs
and FP32 weights in descending adjusted-score order. Ties prefer lower expert
IDs; the reference does not guarantee identical tie choices. FP32 projection
reduction order still requires numerical qualification.

Admission covers 1–4096 tokens, exact expert-dependent extents and disjoint
logit/output/error scratch. Non-finite input, bias or invalid mask sets bit 1;
non-finite projection/scoring/weight arithmetic sets bit 2. Results require
completion/error admission before dispatch. This initial warp projection and
serial per-token small top-k implementation is not throughput-qualified. Expert
ownership, gathered token construction, expert result accumulation and MoE
collectives remain to be connected; no full-model capability is registered.

## Per-rank expert dispatch plan

`pih_deepseek_v41_expert_dispatch_cuda` implements `LaunchExpertDispatch` from
the router's U32 `[tokens,picks]` indices. For TP 1/2/4/8 it assigns a contiguous
expert range to the expected rank, using 384/top-6 backbone or 128/top-3 MTP
primitive geometry. Outputs are U32 counts `[local_experts]` and routing slots
`[local_experts,tokens]`. Each live slot is `token*picks+pick`, preserving both
the input row and routing-weight location. Entries are stable in token order;
unused capacity is filled with `UINT32_MAX`, including empty experts.

Every route table is checked for out-of-range expert IDs and repeated selection
of the same expert by one token, including malformed remote-expert routes.
Errors set device-error bit 1. Only the first matching slot is stored, so invalid
duplicates cannot overflow an expert's token-sized row. Consumers must still
discard all output after an error; this is not silent duplicate repair.

The launch validates layer, rank/world geometry, exact buffer extents and
disjoint writes, and submits without host synchronization. Its initial per-expert
scan is not throughput-qualified. Counts remain on device: count admission,
row gathering, expert execution, scatter/FP32 accumulation and MoE reduction
remain to be connected. The caller must bind this rank range to the actual
admitted expert weights; arithmetic partitioning alone is not ownership proof.

`LaunchExpertGather` consumes the dispatch plan for one explicitly owned expert,
the original BF16 `[tokens,5120]` input and FP32 `[tokens,picks]` route weights.
It writes BF16 `[rows,5120]` input and FP32 `[rows]` weights for that expert.
The host-specified row count must not exceed tokens; the kernel additionally
checks that it equals the device count before accepting any row. Each routing
slot must be in bounds, point to the expected expert and have a strictly larger
token index than the previous slot. Routing weights must be finite/nonnegative.

Invalid rows are zero-filled and set device-error bit 1, never followed as
input addresses. Error-marked output is not valid expert input despite safe
zero filling. Zero-row experts require absent output regions and still submit
a count check, without reading slots or writing null output. The launch does
not read counts back to the CPU or infer host counts from stale state; the
enclosing scheduler must admit that observation and preserve plan lifetime.
Gathered buffers and error storage cannot overwrite routing metadata or source
input. Expert execution/scatter and completion-controlled scheduling remain to
be connected.

`LaunchExpertScatter` consumes already routing-weighted BF16 expert output
`[rows,5120]` and adds it to an FP32 `[tokens,5120]` accumulator using the same
dispatch slots. It does not multiply routing weights again. Counts, slot bounds,
expert identity and increasing token order are checked before addressing the
accumulator. Invalid metadata sets bit 1 and skips the row; nonfinite expert
values or accumulation overflow set bit 2. Empty experts still check counts.
Atomics prevent a data race even for malformed non-adjacent duplicate slots;
any flagged result must be discarded, not consumed as partially repaired output.

The owner must zero the accumulator before the first expert, submit each expert
exactly once in a fixed order on the dispatch stream, keep metadata immutable
and exclude concurrent accumulator writers. Exact sizes and disjoint metadata,
expert input, accumulator and error storage are validated. The primitive does
not establish weight ownership, exactly-once scheduling, completion admission,
cross-rank reduction, shared-expert addition or the final BF16 conversion.

## Gathered expert execution chain

`pih_deepseek_v41_expert_chain` exposes `ExpertTokenChainLaunch` and
`LaunchExpertTokenChain`: gather token rows and routing weights, run the routed
FP4 gate/up/weighted activation/down chain, then scatter its BF16 output into
the FP32 token accumulator. The scatter descriptor is derived from the gather
plan and expert output, so it cannot independently select a different expert,
row count or stream. Execution is required for nonempty experts and forbidden
for empty experts; the latter perform count checks without matrix launches.

All component layouts, connections and whole-chain live-buffer aliases are
checked before the first enqueue. The gathered input/weights are internal
producer-consumer edges; projection scratch, activation outputs, gathered
buffers, accumulator and error storage must otherwise be disjoint and must not
overwrite source tokens, routing metadata or any expert weights/scales.

This is an enqueue-only chain, not a complete MoE scheduler. The caller still
admits device counts and actual expert weight identity, initializes the shared
accumulator once, submits every local expert exactly once in fixed stream order,
and retains buffers until completion. An enqueue failure can leave earlier
kernels in flight; do not retry the chain or reuse its accumulator as if nothing
was submitted. Device errors require retirement of the entire result. Rank
reduction, shared-expert merge and layer scheduling remain separate work.

`pih_deepseek_v41_expert_counts` provides the asynchronous count observation
needed before allocating per-expert rows. `ExpertCounts::Start` validates the
dispatch layout, current-device memory domains and disjoint pinned host count
and error slots, then submits dispatch, count readback and error/event recording
on one explicit stream. The caller supplies an exclusive event, exact-sized
pinned count storage and a steady-clock deadline; no hidden allocation or host
synchronization is performed.

`Poll` returns false while pending. Only after successful event/error admission
does it copy counts into owned CPU storage and validate per-expert token bounds
and the total routing capacity. `Rows(global_expert)` rejects pending, failed,
moved-from and out-of-rank requests. Repeated successful polling uses the cached
observation, never re-enqueues dispatch or rereads reusable host storage.

The caller must initialize the device error flag, preserve the dispatch plan
and pinned slots until observation or safe retirement, and keep the event
exclusive. Failure, including deadline expiry, does not cancel queued CUDA work
or release any borrowed storage. This observation is not an expert-weight
identity proof or an immutable lease on the device routing plan; the enclosing
scheduler must bind those resources and use each count for the matching chain.

`pih_deepseek_v41_expert_batch` connects that observation to all local expert
chains. `ExpertBatch::Start` requires exactly the local expert count, in
ascending global expert ID, including empty experts. Every chain must match
the observed dispatch descriptor, cached row count, common source hidden state,
route weights and FP32 accumulator. `ExpertCounts::ValidatePlan` compares all
dispatch regions and step/rank/stream fields; it does not hash device contents
or prevent their modification, so the owner's immutable-plan lease remains
required.

Whole-batch preflight runs before accumulator initialization. It checks every
chain and forbids writes overlapping any expert's weights/scales or common
input/routing metadata. Different experts may reuse scratch because execution
is sequential on one stream. The batch zeroes the accumulator once, submits
each expert chain once in the declared order, and records error readback plus
an exclusive completion event. `Poll` reports success only after zero-error
completion within the deadline; repeated polls do not resubmit work.

An expired or failed batch may have partially submitted GPU work, including
accumulator initialization. It must be retired rather than replayed; callers
must also prevent submitting the same batch through a second `Start`. Pinned
completion storage and all GPU resources remain borrowed. This is rank-local
routed-expert scheduling, not cross-rank reduction, shared-expert addition or
whole-model readiness. Device/weight ownership remains the runtime owner's
responsibility; no Python execution or old-model fallback is introduced.

## Routed sum and shared-expert merge

The vendored reference `MoE.forward` accumulates routed expert outputs in FP32,
performs FP32 all-reduce, adds the replicated shared expert once, then casts to
the hidden-state dtype. `Fp32ReductionLaunch`, `ValidateFp32Reduction` and
`EngramReduction::SubmitFp32` now provide in-place NCCL FP32 SUM without a BF16
intermediate. They share the BF16 transport's exact NCCL release, communicator
rank/device/idle-state admission, bounded storage validation and asynchronous
enqueue state machine, with four-byte alignment and element counts. Enqueued
still does not mean GPU-complete; collective ordering and abort ownership are
unchanged. The generic interface does not itself bind the buffer to a MoE layer.

`LaunchExpertMerge` in the shared-expert CUDA target reads FP32 routed results
and BF16 shared-expert results `[tokens,5120]`, adds in FP32 and rounds once to
a disjoint BF16 output. It accepts 1–4096 tokens, validates exact extents and
write aliases, and flags nonfinite operands, sum overflow or BF16 overflow with
bit 2 while zeroing the invalid element. Flagged output is not valid inference
output. It does not reduce or rescale the replicated shared expert.

The caller must schedule merge only after the routed reduction and shared
expert computation are ready on the same stream, retaining the original FP32
sum until then. The reduction-to-merge state machine and FFN residual integration
are not supplied by these two primitives; no full MoE or model readiness is
claimed from syntax-only validation.

`ExpertSharedMergeLaunch` now binds the shared expert's full FP8 computation
to the final merge. Its whole-chain preflight validates output/count/stream/error
connections and live-buffer aliases, including prohibiting in-place routed
reduction storage from overlapping shared inputs or weights. Single-rank callers
can enqueue `LaunchExpertSharedMerge` after their local accumulation without
linking NCCL, but still own completion admission.

The optional `pih_deepseek_v41_expert_nccl` target supplies
`ExpertTensorParallel` for 2/4/8 ranks. Starting with rank-local FP32 accumulation
already ready on its stream, it submits the FP32 reduction, waits for NCCL enqueue
completion, rechecks communicator/device admission, executes the replicated
shared expert and merge once, and records the final error/event observation.
`Advance` distinguishes waiting-for-reduction, waiting-for-completion, complete
and failed states. A successful event alone is insufficient: zero device errors,
healthy NCCL state and the deadline must all pass. Failed or moved-from operations
cannot replay; repeated completed calls use cached state.

This tail does not itself submit or authenticate `ExpertBatch`, prove cross-rank
token agreement, or own buffers/communicators. Keep all
borrowed resources exclusive and alive through safe retirement, including after
partial submission or timeout. The full model scheduler must connect these stages
and coordinate rank failure; the tail must not be independently retried.

`ExpertResidualLaunch` now makes the FFN return explicit: its shared/merge chain
feeds `MhcPostLaunch`, producing BF16 `[tokens,4,5120]` from the merged sublayer,
original residual and prepared post/comb coefficients. The standalone
`pih_deepseek_v41_expert_residual` target exposes validation and enqueue without
requiring NCCL. Preflight checks the shared output-to-sublayer connection,
stream/token/error identity, and cross-stage aliases. Neither expert scratch nor
in-place routed reduction may overwrite the retained residual or coefficients;
the final output also cannot overwrite expert inputs or scratch.

`ExpertTensorParallel::Start` takes this residual descriptor directly; there is
no compatibility overload for the earlier merge-only descriptor. After reduction
enqueue it executes shared computation, merge and mHC post, then records the
completion event **after the residual update**. Its complete state consequently
covers that entire tail. The enclosing layer must still provide correctly paired
FFN coefficients from its mHC input stage and bind the preceding expert batch;
shape agreement alone is not a coefficient provenance proof.

## FFN input-to-routing chain

`pih_deepseek_v41_ffn_route` exposes `FfnRouteLaunch`: mHC sublayer input
processing, BF16 collapse/RMS normalization, router scoring/selection and expert
dispatch/count observation. `ValidateFfnRoute` requires parsed backbone geometry
(layers 0–39, 384/6 routing), matching layer/token/stream/error descriptors,
normalized input connected to the router, and router indices connected to the
dispatch plan. It rejects whole-chain writable aliases against original
residual, carried pre coefficients, norm/mix/router weights and optional image
mask/bias inputs.

`LaunchFfnRoute` returns the pending `ExpertCounts` observation after enqueueing
input processing and routing followed by dispatch/readback. Poll it successfully
before selecting expert row sizes. Newly computed mHC pre/post/comb, normalized
hidden state and router weights remain available for subsequent stages; new pre
coefficients must not replace the carried pre used by the current input stage.

Caller-owned error storage must be initialized before the chain. Completion and
count storage admission or deadline failure can occur after earlier kernels were
submitted, so any failure requires safe retirement, not replay or immediate
resource reuse. This entry point does not allocate expert workspaces, schedule
the expert batch, bind its weights or authenticate coefficient provenance. MTP
layers are deliberately rejected by this backbone execution chain even though
individual routing primitives admit their geometry.

`pih_deepseek_v41_ffn_nccl` now supplies `FfnContinuation` for the observed-route
to final-residual transition on 2/4/8 ranks. Its `Start` takes the original route
descriptor, admitted counts, all local expert descriptors, the shared/residual
tail and borrowed communication/completion resources. Validation binds normalized
hidden state, route weights, accumulator, residual, post/comb and stream/error
identity across stages. Batch scratch cannot overwrite retained coefficients or
shared weights; final tail writes preserve the next-sublayer pre coefficients
and every persistent expert weight. Route buffers overlapping later expert
weights are rejected too, but this late check cannot undo already submitted
routing: the allocator must enforce weight/workspace separation before routing.

The continuation initializes and submits the local batch, polls its zero-error
completion, and only then starts reduction/shared/merge/residual execution. It
reuses the exclusive event/error slot only after the batch observation completes.
`Advance` never resubmits a completed stage; deadline or stage failure poisons the
continuation. It reports completion only when the entire residual tail completes.
The owner must coordinate rank aborts if any rank fails before entering a
collective, keep the communicator exclusive across both phases, and retain all
borrowed buffers. This does not yet construct variable-row expert descriptors,
own their allocations, or wrap route submission into a full model scheduler.

## Expert workspace and descriptor construction

`pih_deepseek_v41_expert_workspace` provides `ExpertWorkspaceBytes(tokens)` and
`BuildExpertWorkspaceBatch`. Capacity is calculable before routing for 1–4096
tokens. The caller supplies one exact-sized, 256-byte-aligned GPU allocation;
the builder partitions it into 13 aligned segments for gathered hidden/weights,
gate/up/down activation quantization scratch and outputs, activation output,
and a separate persistent FP32 accumulator. Scratch capacity is token-sized and
reused across sequential experts, not allocated once per expert.

After `ExpertCounts::Poll` succeeds, the builder binds the identical dispatch
plan, checks every local expert's FP4 weight and E8M0 scale extents (including
empty experts), and constructs ascending-ID `ExpertTokenChainLaunch` descriptors
using exact observed-row prefixes of the shared segments. Empty expert output
regions are absent. It runs full batch validation before returning the descriptor
vector and accumulator region. It enqueues no GPU work and does not initialize
the accumulator; `ExpertBatch` performs that initialization once.

The whole workspace must be disjoint from source/routing/error storage and all
supplied expert weights. Geometry checks are not weight identity admission; the
loader must map each ordered weight descriptor to the correct global expert.
The runtime still owns device allocation, device identity and safe retirement.
Use the returned accumulator as the tail's routed input and the returned vector
with the continuation, keeping workspace and weights live until its completion.
Allocation must not overlap shared-expert weights, residuals or coefficients;
the full continuation performs those additional cross-stage checks.

`AllocateExpertWorkspace` now calls the admitted
`pih_nvidia_cuda_memory_api_v1::allocate_device` capability, requesting the exact
planned size and 256-byte alignment. Supply an empty ABI-initialized
`pih_cuda_allocation_v1` ledger slot. The returned handle is checked for ABI,
device kind/ordinal, extent, power-of-two alignment, aligned address and nonzero
generation. `ValidateExpertWorkspaceAllocation` can recheck this metadata before
binding its `{address, bytes}` to the descriptor builder; this is not a provider
inventory lookup or proof that a copied handle is still live.

The allocation call preserves the provider's output even if status or allocation
metadata is malformed. Ambiguous ownership is an error requiring quarantine and
provider reconciliation, not an invitation to free an inferred address or retry
into the same ledger slot. On normal success the runtime retains the full handle
and capability activation, waits for all users to retire, then releases through
that provider's `deallocate_device` and checks its status. This adapter has no
destructor-time free, direct CUDA allocation or old-runtime dependency; complete
operation/resource ownership is still the enclosing runtime's responsibility.

`ExpertWorkspaceOwner` adds an explicit allocation/retirement ledger using the
backend memory and async capabilities. Construct it with the admitted APIs,
device/context, stream and a dedicated exclusive retirement event. `Allocate`
admits an idle event and requests the workspace; `BeginUse` returns its region
and prevents release. After every user's final command has been enqueued on
that stream (including completion of any pending NCCL enqueue), call
`RecordRetirementFence`. `PollRetirement` returns false while pending, then makes
the allocation reusable. `Release` is allowed only in the ready state and calls
the original provider with the full original allocation handle.

The retirement event must not be shared with stage-completion events or other
operations. No work may use the region after its retirement fence until another
successful `BeginUse`. Event completion establishes memory lifetime safety, not
numerical validity: device-error admission remains the computation's job. The
owner performs no blocking synchronization and can keep polling retirement even
after an inference deadline has expired.

Failed allocation/fence/query/release paths quarantine the ledger and prohibit
automatic retry or release. The original handle remains accessible for provider
fault reconciliation. The class is noncopyable/nonmovable and does not free on
destruction: its enclosing runtime must explicitly release it or retain/transfer
its ledger to fault retirement while keeping the capability activation alive.
This prevents implicit frees of in-flight storage but does not replace global
rank cancellation or prove that external users obeyed the stream/event lease.

`FfnContinuation::Start` now takes ordered `ExpertWeights` and a ready
`ExpertWorkspaceOwner`, replacing its earlier manually assembled batch argument.
The tail template must leave `experts.merge.routed` absent: the entry constructs
the batch from admitted counts and binds the accumulator itself. It validates
workspace capacity/stream and rejects reuse of the retirement event as the
computation event. Full preflight precedes `BeginUse`; after that, any submission
failure leaves the owner in-use for explicit fault retirement.

The owned allocation's entire capacity, not just active row prefixes, must be
disjoint from retained route/tail storage. In particular the returned residual,
next pre coefficients, shared-expert buffers and persistent weights cannot be
freed accidentally with the expert workspace. After the tail completes, the
continuation records the dedicated retirement fence and enters
`kWaitingRetirement`. Only confirmed retirement produces `kComplete`, with the
workspace ready for reuse or explicit release. No compatibility overload accepts
the earlier manually supplied batch. Keep the owner alive and exclusively bound
through the operation; timeout/failure still requires coordinated rank cleanup,
and retirement polling may be continued separately after an inference deadline.

## Unified backbone FFN operation

`FfnOperation` in the native FFN NCCL target wraps route submission, count
observation and the managed continuation for 2/4/8 ranks. `Start` takes the parsed
configuration, route and unbound tail descriptors, ordered weight regions, a
ready workspace owner, pinned count storage, communicator, completion resources
and deadline. It copies host descriptors/weight-region metadata before enqueueing;
the actual device storage and provider activations remain borrowed.

Before routing, it validates backbone geometry, workspace binding, the tail's
component layouts with the derived accumulator, all local FP4 weight extents and
communication admission. Route writes and the reserved expert workspace must
not overlap any shared/routed expert weights. This moves persistent-weight
protection ahead of route submission; count-dependent batch and full cross-stage
checks still occur before expert execution. `ExpertWorkspaceAccumulator` derives
the fixed accumulator region without requiring route counts.

`Advance` transitions from waiting for counts to managed execution and reports
complete only after the residual tail and workspace retirement complete. It
never retries failed stages. A failure can leave earlier route or expert work
in flight, so the outer rank owner must coordinate abort and retain all buffers.
Keep the workspace owner exclusively reserved for the whole operation, including
the initial count phase when its allocation has not yet entered expert use.
This is a backbone FFN operation, not a whole attention/FFN block, model engine,
weight loader or single-rank execution port.

The unified operation now reserves its workspace **before routing is submitted**,
closing the earlier ready-state gap while counts were pending. `Reserve` returns
a monotonically increasing, nonzero reservation identity and enters `kReserved`;
release, a second reservation and unreserved `BeginUse` are rejected. The same
identity is passed through continuation binding, actual expert use and retirement
fence recording. Old reservation identities cannot acquire a later reservation;
identity exhaustion is an error rather than wraparound.

Successful retirement clears the active reservation. A route failure leaves the
workspace reserved, because earlier work may already have been queued; it is not
silently made reusable. Fault retirement can inspect `reservation_record()` and
record a matching fence only after all applicable work is safely enqueued and no
NCCL enqueue is pending. The ledger/identity checks do not replace exclusive host
access or capability lifetime ownership and are not a security boundary against
callers bypassing the contract. Standalone continuation use without an earlier
route reservation still requires a ready workspace and reservation zero.

## Prepared attention-to-FFN block operation

`pih_deepseek_v41_prepared_block` connects `AttentionTensorParallel` to the unified
FFN through `PreparedBlockOperation` on 2/4/8 ranks. It starts with query, window
and compressed KV, selected indices, RoPE phases, attention residual coefficients
and the attention-stage pre coefficients **already prepared**. The operation does
not compute or authenticate these upstream inputs and is not a complete model
layer input scheduler.

`PreparedBlockLaunch` binds attention residual output to the FFN residual input
and explicitly carries attention pre into FFN collapse. Admission checks the
backbone configuration's compression ratio, rank-local attention group count,
token/stream/error identity and all component layouts. Attention writes cannot
overwrite future FFN coefficients or expert weights; FFN writes cannot overwrite
retained attention window/compressed caches or projection weights. The expert
workspace is reserved before attention submission and the same identity passes
into FFN, without reopening a ready-state gap.

`Advance` waits for attention's zero-error residual completion before reusing the
exclusive computation event/communicator for FFN. Completion includes the FFN
residual and expert workspace retirement. Repeated advancement never reruns a
stage; failure retains partial-state resources for outer rank cancellation and
retirement. Host descriptors are copied, while device buffers, workspace owner,
capability activations and communicator remain borrowed. Single-rank execution,
upstream attention/cache ownership and whole-model scheduling remain separate
unfinished integration work.

## Attention mHC/query/window preparation

`pih_deepseek_v41_attention_prepare` exposes `AttentionPrepareLaunch` and
`LaunchAttentionPrepare`. It connects mHC mixing, carried-pre collapse and input
normalization to both the low-rank/expanded query chain and replicated window KV
projection/normalization/RoPE/quantization/cache update. It admits backbone layers
0–39 and world sizes 1/2/4/8, with `64/world_size` query heads and identical
token/stream/error identity across the three chains.

Query and window RoPE consume the same phase region. The caller must still prove
those phases correspond to this layer and step; region equality does not prove
their contents. Whole-chain alias admission retains normalized hidden state,
normalized low-rank qr, mHC pre/post/comb and original residual for downstream
compression, indexing and residual return. Only the existing exact query
expand/RoPE and window norm/RoPE in-place pairs are allowed; partial overlaps and
other cross-stage writable aliases are rejected before the first enqueue.

The entry submits input processing, query generation and window generation on
one stream without allocating or synchronizing. Enqueue success is not GPU
completion. Compressed KV production, index/candidate selection, upstream Engram
and automatic connection to `PreparedBlockOperation` remain to be integrated;
this preparation primitive does not claim full attention or model execution.

## Owner-layer compressed KV preparation

`pih_deepseek_v41_compressed_prepare` connects compressor projection, optional
ratio-two pooling, normalization and compressed RoPE/quantization/cache write.
`CompressedPrepareLaunch` includes the backbone layer and absolute step start;
admission requires the frozen configuration's owning compressed-KV layer, matching
compression ratio and pooling step. Sharing consumers and ratio-zero layers must
not invoke this producer.

Emitted rows are derived from the step boundaries. The cache stage is present
exactly when at least one complete row is emitted and starts at `start/ratio`.
An incomplete ratio-two step still executes projection/persistent pooling-state
update but must omit cache preparation, normalization and RoPE. This handles the
zero-new-row decode case without passing zero-sized work into nonempty kernels.

The normalized, unrotated compressor latent is preserved for index-key generation;
RoPE output and the full destination cache are separate live regions. Whole-chain
alias checks protect original hidden state, weights, phases and persistent pool
state. The caller supplies phases for each completed group's original position,
not its compressed slot number, and owns cache source identity/capacity/lifetime
and completion admission. Query/window, indexer and layer execution still need
to be joined around this producer; it is not a cache ownership scheduler.

`pih_deepseek_v41_attention_sources` now joins query/window preparation and the
optional compressed producer under `AttentionSourcesLaunch`. Producer presence
must exactly match the frozen layer's compressed-KV ownership: owner layers run
compression even on zero-emission state-update steps; ratio-zero and sharing
consumer layers omit it. Compressor input is bound to the normalized hidden
output from mHC preparation, with identical layer/start/token/stream/error data.

The combined preflight checks both component graphs and cross-graph liveness.
Query/window/mHC writes cannot clobber compressor weights, phases or persistent
pool state; compression writes cannot clobber low-rank qr, normalized hidden,
query/window outputs or carry coefficients. Only the common error flag is a
shared writer. Submission orders attention input/query/window before compression
on one stream, and leaves all downstream inputs available.

This source stage still relies on externally admitted shared-cache identities
for consumer layers, precomputed phases and caller-owned completion/lifetimes.
It does not yet submit index selection or connect source buffers to the prepared
block tail automatically, and does not claim full model execution.

## Source-to-index preparation operation

`pih_deepseek_v41_indexed_sources` supplies `IndexedSourcesLaunch` and
`IndexedSourcesOperation` for 2/4/8 ranks. Index execution is present exactly for
index-owner layers whose step has a nonempty compressed prefix. Ratio-zero,
shared-index consumers and empty-prefix steps omit it; those paths still record
and admit source-stage error/event completion rather than declaring enqueue
success complete.

The index query projection consumes normalized low-rank qr from attention query
preparation, head weights consume normalized hidden state, and newly produced
index keys consume the unrotated normalized compressor latent. Query/key phases,
step, rank head count and error/stream identity are bound to their respective
source descriptors. Frozen index-layer validation still governs candidate-mask
production/consumption and key-cache ownership.

Whole-graph preflight protects source inputs, weights and retained Q/KV/mHC
outputs against indexer scratch and cache writes, and protects external indexer
weights, existing key caches and candidate masks against earlier source writes.
After source enqueue, index owners run the existing NCCL index pipeline; other
layers record a source-only completion event. `Poll` is pending until zero-error
completion and deadline admission, and failed/moved operations cannot replay.
Cache-generation identity, shared-index freshness, phase contents, upstream
Engram and attachment to the prepared attention/FFN block remain outer-runtime
responsibilities, not proven by these region connections.

## Prepared block source binding

`pih_deepseek_v41_block_bindings` exposes `BindPreparedBlockSources` and
`ValidatePreparedBlockSources`. Given admitted source descriptors and a block
template containing weights, scratch and declared geometry, binding populates
query and inverse-RoPE phase inputs, the prefill KV rows or decode ring, attention
residual/post/comb, and the carried pre/residual inputs for FFN. Locally produced
selected indices and the populated prefix of a locally updated compressed cache
are bound directly to their producers.

Declared layer/start/token/rank/ratio/stream/error geometry is not silently
rewritten: mismatches are rejected after binding along with component layout
errors. Shared-index/cache consumers, including owner steps with no new cache
write, must supply externally admitted existing regions. The binder does not
invent cache ownership, freshness or phase contents from shape alone.

Binding is a CPU descriptor transformation, not execution or completion proof.
Run it before submitting the relevant work, retain all buffers, and admit source
completion before starting the prepared block. Whole source-to-block cross-stage
alias admission and automatic orchestration still need integration; the existing
component validators do not by themselves prove that the entire combined graph
is safe. No whole-model readiness is claimed from this connection helper.

`ValidateBlockCrossStageBuffers` adds source/index-to-block lifetime admission
to the binding target. It validates the bound descriptors and FFN tail, includes
all ordered local expert weights, derives the accumulator from the exact owned
workspace extent and gathers the live source/index regions. Earlier source/index
writes cannot overwrite external later-block inputs or weights; later block
writes, including the **entire** expert workspace allocation, cannot overwrite
retained source/index outputs or inputs. The shared four-byte error flag is the
only allowed write overlap across these stages.

Producer-consumer reads are allowed only through the connections already checked
by the binder. Existing shared compressed/index regions remain external reads,
so local preparation cannot silently overwrite them. This conservative contract
does not allow cross-phase scratch reuse, output-in-place-over-original-residual,
or recycling cache capacity before the block finishes. Component-local exact
RoPE aliases and sequential per-expert workspace reuse remain supported.

This is an additional static admission check, not GPU completion, cache freshness
proof or automatic scheduling. Run it before the source stage is submitted and
still run every execution stage's own preflight. Integrating it with one automatic
source-to-block operation remains the next orchestration step.

## Automatic source-to-block operation

`pih_deepseek_v41_block_operation` now supplies `BlockOperation` for a backbone
layer on 2/4/8 ranks. `Start` binds produced source regions into the block
template, runs cross-stage lifetime admission and workspace/communication/event
preflight, copies host descriptors, then reserves the workspace before source
submission. `Advance` waits for admitted source/index completion, starts the
prepared attention/FFN block with that same reservation, and reports complete
only after its final residual and workspace retirement complete.

The exclusive computation event and communicator are reused only across admitted
stage boundaries. Failed or moved operations cannot replay; failures may retain
partial GPU work and a workspace reservation. The caller must coordinate rank
abort/retirement and retain the workspace owner, weights, caches, pinned count
storage, event and capability activation until safe cleanup. No destructor in
this orchestration stack implies cancellation or safe automatic GPU deallocation.

This entry covers mHC/query/window/compression/index preparation through the
attention/FFN return, using already admitted residual, phase and shared-cache
inputs. Engram integration, RoPE table generation, shared-cache publication and
freshness, weight admission/loading, single-rank execution and the full model
engine remain unfinished. Passing syntax checks does not establish numeric,
NVCC-link or hardware qualification.

## Step-bound RoPE table generation

`RopeSequenceLaunch` generates a bounded U32 original-position sequence on the
device before invoking the existing RoPE table kernel. It admits strides 1/2,
nonempty rows and a final original position below 1,048,576; position/output/error
storage must be disjoint. This removes the need to upload a hand-built position
array for these normal sequence steps.

`pih_deepseek_v41_step_phases` binds query and optional compressed sequences to a
backbone layer/step. Query positions start at the absolute step start with stride
one; nonzero-start decode has one token. A compressed table is required exactly
when the configured compressed-KV owner emits completed rows. It starts at
`floor(start/ratio)*ratio` with stride `ratio`, never at the compressed cache slot
number. Empty-emission and consumer layers omit that table. Both tables share
the admitted stream/error flag but have disjoint writable storage.

`LaunchStepPhases` enqueues position/table generation only. Connecting generated
tables to all consumers and admitting their writes against the whole block graph
remain required before integrating it into `BlockOperation`. This step contract
does not initialize sequence caches or infer shared-cache freshness and is not
hardware/numerical qualification.

`BlockOperation::Start` now requires `StepPhasesLaunch` and generates those
tables before submitting attention sources. It binds generated query phases to
query/window/index-query consumers and generated compressed phases to new
compressed KV/index-key consumers; the block binder propagates the same query
table into inverse RoPE. Layer, step, emitted rows, stream and error identity are
validated across generators and consumers. The previous externally prepared
phase-only block signature is replaced, without a compatibility overload.

Phase position/output regions participate in cross-stage lifetime preflight.
Their writes must be disjoint from all non-phase source inputs, persistent
weights, caches, expert workspace and later writable regions. Only actual phase
consumer fields are treated as produced edges: a weight region coincidentally
equal to a phase region is still rejected. Workspace reservation occurs before
generation; partial phase/source failures retain the same failure-retirement
obligations as other block stages. Completion of the source stage covers phase
generation too because all commands use the same stream.

This closes normal backbone RoPE table generation inside the block operation;
Engram, shared-cache publication/freshness, full sequence/model loading and
single-rank orchestration remain unfinished.

## Engram integrated into block execution

`BlockOperation::Start` now takes an `EngramPlan` and arena and generates the
optional `EngramLaunch` internally from configured layer ownership.
It is required on configured backbone layers 1 and 14 and forbidden elsewhere.
The incoming sequence residual supplies the generated Engram gate input.
The operation binds the gate output into both attention residual fields before deriving
attention residual connections. Layer, token count, rank/world, stream and error
flag must match the rest of the block. This follows the reference model's
pre-block Engram placement; Engram is not applied after attention or FFN.

The block reserves its expert workspace before any Engram command. It waits for
Engram lookup reduction, projection, gate and zero-error completion before
generating phases and starting sources. The same reservation persists through
FFN retirement. Completion resources are reused only after the preceding stage
has completed. Failure retains the reservation and borrowed buffers for owner
abort/retirement; callers must not reset flags, replay the block or release
storage while work may remain in flight.

Whole-block admission rejects Engram scratch/output overwriting future inputs,
weights or workspace, and rejects later block/phase writes into Engram storage.
Only the explicit gate-output-to-residual edge and common error flag are allowed;
arbitrary address-equal weights do not become permitted producer edges.

Engram hash generation/upload and revision/weight admission remain caller-owned.
This completes the Engram stage connection, not shared-cache freshness, the
full sequence/model loader, single-rank orchestration or GPU qualification.

## Completion-owned backbone sequence publications

`BlockOperation::Start` now requires a `BlockSequence&` constructed with the
admitted configuration. It is noncopyable/nonmovable and must outlive every
operation borrowing it. Submit layers 0 through 39 in order, polling each block
to completion before submitting the next. First prefill starts at zero with
1–4096 tokens; subsequent steps have exactly one token and start at the previous
completed step's end. Config identity, stream, communicator, rank/world and error storage stay
fixed. No compatibility overload bypasses this sequence contract.

For layers after zero, the sequence supplies the previous FFN's residual and
pre-mix coefficient. On Engram layers, that residual becomes the gate input and
the gate output then becomes the attention input. A block publishes only after
its entire attention/FFN execution and workspace retirement complete with no
device error. Merely enqueuing a producer does not authorize consumers.

Compressed KV and index-key caches retain allocation identity between steps,
including ratio-two decode steps that emit no new compressed row. Window rings
and compressor tail state are likewise persistent. Later layers consume the
configured completed owner's compressed prefix; index owners reuse the correct
key cache, and index consumers reuse this step's selected rows. Candidate-mask
consumers use layer 20's completed mask. Per-step indices and masks are cleared
from the host publication registry at each 40-layer boundary, not reused based
only on matching dimensions. The underlying GPU bytes are not cleared.

Cross-stage admission also protects other layers' retained rings, compressor
state, KV/key caches, indices and candidate masks from all writes in the next
block, including Engram, phase generation and expert workspace scratch. This
initial implementation conservatively retains publications through the step;
it does not optimize storage reuse after a consumer's last read.

`StepOutput()` returns the final residual and carried pre coefficient only after
all 40 layers complete; the caller must consume them before starting the next
step. Final collapse, normalization and vocabulary projection remain downstream.
A failed advance or destruction of an unfinished block permanently invalidates
the sequence. Destruction does not synchronize, abort NCCL or release memory:
the caller still owns generation abort and device-resource retirement. A new
sequence requires separately safe/initialized resources, not reuse of possibly
in-flight buffers.

This registry is process-local sequencing, not a memory-lease security boundary.
All buffers remain borrowed and must not be externally modified or concurrently
used. Embedding/hash preparation, allocation-provider lease validation, complete
model loading, distributed agreement and GPU numerical qualification remain
separate required work. The implementation currently follows the existing
2/4/8-rank block path, not single-rank or speculative/chunked-prefill execution.

## Sequence-to-logits output head

`ModelHeadLaunch` implements final carried-pre mHC collapse, RMS normalization
with epsilon `1e-20`, and projection of the last normalized BF16 position. The
projection reads FP32 weights `[129280/world_size,5120]` (checkpoint BF16 must
be promoted by the loader) and produces FP32 local logits. Rank `r` owns the
contiguous vocabulary interval starting at `r*129280/world_size`. It uses FP32
products/accumulation without TF32 or reduced-precision output, and signals
non-finite inputs/weights and arithmetic overflow through the common error flag.
The initial CUDA kernel is a warp reduction, not an optimized GEMM capability.

`HeadOperation::Start` consumes a completed `BlockSequence`, binds its final
residual/pre into that local head, derives the last normalized row, and gathers
equal FP32 vocabulary shards into a separate `[129280]` device output on every
rank. Rank-major gathering is already vocabulary order for this one-position
head; this API does not claim full-prefill logits or DSpark multi-position heads.
Projection geometry supports 1/2/4/8 ranks, while this sequence operation follows
the current 2/4/8-rank backbone path.

The sequence is reserved before local GPU work and remains unavailable to the
next block until NCCL enqueue, zero-error CUDA completion and final NCCL status
have all been observed. The same step cannot publish its head twice. Failure,
timeout or destruction of an unfinished head invalidates the sequence and
requires caller-controlled generation abort/retirement; no destructor frees or
synchronizes GPU resources. `Logits()` exposes the borrowed output only after
completion. The owner must keep it alive and unmodified until downstream
sampling or transfer has finished.

Head scratch, local logits, full logits and error storage are admitted against
all retained sequence caches, including layer zero's ring. Full logits cannot
alias any head input, weight or local scratch. The shared FP32 all-gather adapter
requires exact output extent, disjoint input/output and matching communicator
rank/device, and uses the same pinned NCCL release and nonblocking-state checks
as existing collectives.

This closes the ordinary completed-backbone-to-logits connection. It does not
yet implement sampling, checkpoint promotion/loading, embedding preparation,
model-plugin registration or hardware/numerical qualification. No Python or
legacy inference fallback is introduced.

## Tentative GPU sampling

`SamplingLaunch` follows the repository's `pih-sampler-v1` algorithm contract,
not the reference script's Gumbel sampler. It accepts finite temperature `[0,2]`,
top-p `(0,1]`, top-k `0..129280` (zero means unrestricted), a U64 effective seed
and caller-supplied stochastic ordinal. Greedy (`temperature=0`) requires
top-p=1/top-k=0 and does not evaluate Philox. At most 17 distinct valid token IDs
may be suppressed; unused ID slots must be zero. Suppression is supplied by the
controller after applying the request's min-token/EOS policy, not inferred here.

The CUDA implementation validates every raw logit, even suppressed entries,
scales in FP32, and bitonic-sorts `(score descending, token ID ascending)` over
131072 padded entries. Full-distribution and top-k sums use a fixed pairwise
reduction tree. Top-p chooses the shortest nonempty sorted prefix reaching its
threshold; inverse CDF uses the ABI's FP64 open-interval uniform derived from
Philox4x32-10. The supplied ordinal is not mutated. Only rounding fallback may
select the last survivor. Greedy ties choose the smaller ID.

With logprobs enabled, the selected logprob is returned independently of up to
20 sorted alternatives, using the temperature-scaled allowed full vocabulary
before top-k/top-p filtering. Disabled logprobs requires zero alternatives and
returns zeroed logprob fields. NaN/Inf, scaling/arithmetic overflow and invalid
storage fail the operation rather than sampling remaining finite tokens.

All storage is borrowed and mutually disjoint: FP32 logits `[129280]`, sorted
FP32 scores/U32 IDs, FP32 weights/reduction scratch each `[131072]`, two FP32
statistics, a 180-byte `SamplingCandidate` and the common U32 error flag. The
candidate contains token, RNG word, selected logprob and fixed-capacity top-20
arrays. This is an internal device layout, not the controller's accepted-token
record or a stable external plugin ABI.

`SamplingOperation::Start` requires the current completed `HeadOperation` on
the last rank only. It binds logits/stream/error identity, rejects stale heads
and duplicate sampling for the same step, protects all retained caches and
residual/pre buffers, and reserves the sequence until zero-error completion.
Failures and abandonment invalidate the sequence. `Candidate()` exposes an
owned host observation only after readback completion and structural validation.
No random ordinal, accepted length, usage counter or
pending-input token is advanced by this operation.

The initial sort/reduction path is intentionally unoptimized and has no
deterministic-profile, cross-device bitwise or performance qualification merely
from compilation. Independent sampler fixtures and GPU qualification remain
required. Controller generation/config/ordinal validation,
stop/EOS handling, atomic accepted-token publication and next-input broadcast
are still required before this becomes an end-to-end generation service.

## Bounded sampling candidate readback

`SamplingOperation::Start` requires `SamplingIdentity` (nonzero epoch, plan
sequence, sequence generation and sampling-config ID) plus a separate pinned
180-byte host candidate slot. It freezes those values, the launch parameters,
ordinal and completed head position before enqueue. All eight sampling device
buffers must belong to the current device; pointer checks are not allocation
extent/lease authentication. The host result cannot overlap the pinned error
slot. All borrowed slots and the completion event remain exclusive until safe
completion or caller-controlled failure retirement.

The operation enqueues candidate D2H after sampling, then error-flag D2H and the
completion event on the same stream. `Poll()` never reads candidate bytes while
the event is pending or the error flag is nonzero. After completion it copies
the candidate into owned storage, validates it against frozen parameters, and
only then exposes a value `SamplingObservation`. Reusing the pinned slot after
successful observation cannot mutate that owned result. The earlier
device-region-only accessor has been replaced without a compatibility overload.

Validation rejects invalid/suppressed tokens, incorrect Philox words, non-finite
or positive logprobs, incorrect alternative counts, duplicate/out-of-order
alternatives, mismatched selected-token logprob and nonzero unused/reserved
fields. Greedy selected token must equal the first alternative when present.
Returned logprob rounding can merge distinct score values, so equal rounded
logprobs are not used to assert a token-ID tie order; original-logit ordering
still requires independent numeric fixtures. This validation does not recompute
sampling on the CPU or transfer the complete logits to the controller.

Identity fields are caller-supplied correlation metadata, not proof of a live
controller transaction. The controller must still authenticate epoch,
generation, plan/config and ordinal, perform stopping/output-credit decisions,
obtain required distributed completion acknowledgements and atomically publish
the accepted token. This operation neither advances the RNG ordinal nor grants
permission for the next decode by itself.

## Transactional token-stop preview

`TokenStopState` consumes a structurally admitted `SamplingObservation` and the
chosen token's exact tokenizer bytes. It does not decode text or authenticate
the tokenizer. Its bounded policy allows min/max completion counts, at most 17
EOS/user stop-token IDs, and at most 16 nonempty UTF-8 stop strings (256 bytes per
string, 2048 total). IDs and patterns must already be deduplicated in request
order. Raw individual token bytes may contain partial UTF-8 sequences; matching
does not normalize Unicode. A token-byte span is limited to 8192 bytes.

`ApplySuppression` replaces the sampling parameter suppression list with this
policy's EOS/stop IDs exactly when the next accepted count would still be below
the minimum. It validates the resulting sampler parameters before replacing
them. `Preview` rejects an early suppressed stop token and keeps early ordinary
bytes out of the matcher; matching starts fresh at the threshold token.

For ordinary tokens, a fixed 256-byte tail detects stop strings across token
boundaries. The first ending match wins, then the longest matching string;
request order resolves remaining ties. Matched bytes and the rest of that token
are withheld. Otherwise at most `longest_pattern_bytes-1` bytes remain held.
EOS/stop token terminates without exposing its bytes; prior unmatched held bytes
are flushed. String/token stop outranks max-token length on the same candidate.
Length termination flushes the remaining tail. The terminal token still counts
as one accepted completion; no next-input/KV record is invented.

Preview returns fixed-capacity visible bytes, finish classification, proposed
accepted count and the original sampling observation, without changing state.
Dropping a preview is an abort with no count or matcher advancement. `Commit`
checks owner instance identity and predecessor count before applying the
transition; foreign, stale and duplicate commits fail. Moving a stop state
invalidates its old previews. No dynamic allocation occurs in successful
preview/commit; policy strings are allocated only during admission.

This is a local stop-state transaction, not a distributed accepted-token commit.
The controller must coordinate it with generation validation, output credits,
RNG/usage records and worker acknowledgements. Tool/parser completion has higher
finish precedence and must be resolved by that controller before publishing this
preview's bytes; this component does not implement DSML or tool-call parsing.

## Local accepted-token ledger transaction

`TokenLedger::Create` freezes epoch, request sequence generation, sampling-config
ID, ordinary prefill length, sampling and stop policy. The initial plan ID,
ordinal and suppression list must be zero. Prompt length is 1–4096; conservative
`prompt + maximum_completion` reservation must fit 1048576 positions. It reserves
all accepted-record slots at admission, bounded by an explicit record-storage
byte budget; that budget excludes the fixed object/policy storage and downstream
output buffers. Allocation failure is reported before execution. The ledger is
noncopyable/nonmovable and owned through a unique pointer.

`Prepare(plan_seq)` requires a strictly increasing nonzero plan ID, freezes the
current ordinal and expected processed length, and applies min-token suppression.
`Stage` compares the candidate epoch/plan/generation/config, ordinal and exact
causal position, revalidates the candidate fields, then previews stopping without
changing accepted state. A mismatched/corrupt observation fails the ledger; it is
not silently resampled. One prepared plan has at most one staged candidate.

After the controller has secured output capacity and all required distributed
completion/commit authority, `CommitLocal` applies the prepared stop transition
and writes the token/logprobs, accepted count, finish reason, processed length,
pending input and stochastic ordinal using preallocated storage. The queued
fixed-capacity publication owns its visible byte span. Greedy does not increment
the ordinal; stochastic increments exactly once, including a terminal token.
Terminal candidates count in usage but leave no pending input. After `A` accepted
tokens the committed logical processed position is `prompt + A - 1`.

`Abort` drops an unlaunched prepared decision without changing matcher state,
records, pending input or RNG ordinal. After launch, `AbortRetired` additionally
requires caller-established worker retirement. A retry uses a newer plan ID with
the same ordinal. Neither method rolls back GPU caches or authorizes replay after
device failure: the controller must retain/retire worker state according to its
actual distributed transaction. Duplicate/old commit and abort requests are rejected.

This is serialized process-local atomicity, not crash durability or a distributed
commit protocol. `CommitLocal` is deliberately named to avoid implying that a
caller-supplied boolean proves rank acknowledgements or output credit. It must
not be exposed directly as a service commit endpoint. Controller output-credit,
parser/tool-finish precedence, rank acknowledgements and next-input dispatch
remain to be wired before the ordinary generation loop is complete.

## Reserved output slots connected to the ledger

`TokenOutputQueue` allocates 1–4096 fixed `TokenPublication` slots at admission
and backs them with `output_commit_burst_credit_v1`. The explicit byte budget
covers publication payloads; credit/slot metadata and the queue object are
additional bounded overhead. `pih_output_burst_credits` compiles the existing
model-agnostic credit implementation into an independent target, without linking
the legacy monolithic model/runtime archive.

`TokenLedger::Create` now requires a borrowed queue that must outlive the ledger
and every pending receipt. `Prepare` reserves one complete publication slot before
returning a sampling request. Capacity exhaustion leaves the ledger unprepared.
Call `MarkInFlight` before submitting work; `Stage` and `CommitLocal` require the
same live committed reservation. A full slot covers the selected token and all
top-20 alternatives plus the bounded visible byte payload, never assuming the
selected token appears in the alternatives.

After controller-side distributed authorization, `CommitLocal` validates the
reservation before changing the local ledger and publishes into that already
allocated slot. It now returns a `TokenOutputLease`, replacing the unreserved
publication-returning signature. `Read` returns an owned publication copy only
after publication; `Release` recycles the slot only after the consumer is done.
Queue instance identity, slot generation, plan and envelope checks reject stale
or foreign receipts. Filling all slots creates backpressure instead of dropping
accepted records or logprobs.

`Abort` is restricted to prepared, unlaunched reservations. Once marked in
flight, `AbortRetired` may discard an unaccepted result only after the caller has
established worker retirement and safe state handling. This method does not
establish retirement itself. Failed ledgers keep their pending receipt available
through `pending_output()`; the fault owner must record it before destruction and
reconcile the queue explicitly. Destruction does not guess whether workers have
stopped or silently release output capacity.

All queue and ledger calls require exclusive serialized ownership. Their local
publish transition does not replace distributed generation acknowledgements,
engine-poison rechecks or network/parser serializer capacity admission. The
in-memory publication envelope is not a proven bound for JSON/SSE/tool events.

## Token embedding and first-layer input publication

`TokenEmbeddingLaunch` reads U32 IDs and the rank's contiguous BF16 vocabulary
shard `[129280/world_size,5120]`. Each rank writes zero for valid IDs owned by
another rank; invalid IDs and non-finite selected weights set the error flag.
After BF16 TP SUM, the hidden `[tokens,5120]` is copied into four residual streams
and the initial FP32 pre-mix is `[1,0,0,0]` for every token, matching the reference.
The ID, weight, hidden, residual, pre and error regions have exact bounded sizes
and must be disjoint. A low-level single-rank launch is available, but does not
constitute the still-unfinished single-rank full-model operation.

`EmbeddingOperation` provides the 2/4/8-rank completion-controlled path. It
reserves the sequence, enqueues lookup/reduction, waits for NCCL enqueue before
expansion, then observes zero-error CUDA completion and final NCCL status before
publishing the first-layer residual/pre. Prefill has 1–4096 tokens; later steps
have one token and must fit the sequence position limit. Stream, communicator,
rank, error slot and embedding-weight allocation identity remain fixed.

`BlockSequence` now refuses layer zero until this step's embedding is complete,
and binds its residual/pre rather than accepting hand-prepared first-layer input.
The publication is consumed once on layer-zero reservation. Sequence caches are
protected from embedding writes; later block/head/sampling writes also protect
the embedding weight and retained ID region. Preparing the next embedding
invalidates access to the prior step's head through the sequence, preventing an
old head from being sampled after its input has been superseded.

For subsequent steps the preceding head must have completed, and the last rank
must also have completed tentative sampling. This is not controller acceptance:
the owner must still supply only the committed next token after ledger/output and
distributed decisions. Cross-rank token-ID agreement and authenticated buffer
leases are upstream responsibilities. The ordinary text path does not insert
vision embeddings. Failure/abandonment poisons
the sequence and retains the normal caller-owned abort/retirement obligations.

## Unified token IDs and Engram hash upload

`EmbeddingOperation::Start` now also requires the host token span, its sequence's
admitted `EngramHashState`, and `TokenInputUploadLaunch`. The earlier
device-ID-only embedding signature is replaced. Hash position must equal the
sequence's next position, and the same hash-state object must remain bound for
subsequent steps. The compressed-token map's independently admitted digest and
normalization equivalence remain prerequisites of `EngramHashState::Create`;
uploading it does not establish tokenizer/map authenticity.

The upload helper snapshots up to 4096 host U32 tokens before changing any pinned
buffer. That snapshot supplies both embedding IDs and the 24-hash rows for each
Engram layer. CPU hash output is rearranged from token-major pairs into separate
`[tokens,24]` planes for layers 1 and 14. Three pinned sources and three
current-device destinations have exact extents, U32 alignment and disjoint
storage. Copies are enqueued on the same explicit stream before embedding lookup;
stream ordering prevents lookup from consuming unfinished uploads. The embedding
completion event covers all three transfers as well as lookup/reduction/expansion.

Host staging sources must remain exclusive and alive until that completion or
safe failure retirement. They cannot overlap the completion error slot. Upload
writes are checked against retained sequence weights/caches and embedding
storage. Prior step input buffers may be reused only at the admitted next-step
boundary; future block/head/sampling writes protect the newly published token
and hash regions.

After input completion, `BlockOperation` automatically binds Engram lookup IDs
from the sequence's layer-1/layer-14 hash planes. Callers cannot substitute an
unrelated hash region through the layer descriptor. This integrated path is
ordinary text: every token participates in hashing, and image gating masks are
rejected rather than silently producing inconsistent hashes.

Hash history advances before asynchronous copies. Once advancement or upload
begins, a later failure poisons the sequence; no retry/reset is implied. The
owner must retain the hash object and buffers and retire in-flight work before
disposing of the failed sequence. This closes text token/hash staging, not map
generation, artifact admission, visual input processing or the complete model
loader/controller transport.

## Canonical text-backbone runtime weight inventory

`BackboneWeightInventory::Create(config, world, rank)` in `weight_inventory.h`
constructs the exact expected runtime tensor set for the 40-layer ordinary text
path, with TP1/2/4/8 geometry. This is metadata admission, not an executable model
plugin, source checkpoint admission, conversion, allocation or payload loading.
`Validate` admits an order-independent exact set and rejects unknown, missing,
duplicate, incorrectly shaped, incorrectly typed or incorrectly sized members.
Names and descriptors are owned; the returned spans borrow the inventory.

Each descriptor records the physical stored shape and byte count. Packed FP4
weights use `[out,in/2]` bytes, with E8M0 scales `[out,in/32]`; FP8 weights use
`[out,in]` bytes with 32-by-32 block scales. Vectors are explicitly rank one.
Canonical norms and Engram q/k are BF16, while the router and output head are
FP32. A future converter must promote checkpoint BF16 router/head weights and
pooling projections, and dequantize `wo_a` where required; this API does not
pretend the raw HF files already have runtime storage or names.

The `axis`, `first`, `valid`, and `padding` fields describe contiguous physical
row/column slices. Replicated tensors and whole local experts have `axis=-1`;
expert names retain their global IDs. Query/output/index head partitions are
contiguous. Engram tables use ceil-row partitions, with padding confined to the
last rank; padding weight bytes must be zero and E8M0 scale bytes encode one
(127). Metadata validation does not inspect those bytes. Shared KV/index weights
exist only on configured owner layers; they are not invented for consumers.

The inventory intentionally excludes MTP/DSpark, vision and visual router biases.
An authenticated source-to-runtime conversion manifest must account for those
exclusions explicitly before a complete checkpoint can be admitted. Total bytes
cover tensor payload only, not alignment, allocator metadata, cache or scratch.
The standalone `pih_deepseek_v41_weight_inventory` static target includes native
configuration parsing and Engram geometry without linking the legacy model.

### Attention transport scratch

`AttentionLocalOutputLaunch::reduction` is mandatory separate FP32
`[tokens,5120]` scratch, including in the low-level TP1 descriptor. Local output
launch enqueues BF16 projection followed by FP32 promotion. TP reduction uses
this scratch, and `AttentionTensorParallel` rounds back into `linear.output`
only after NCCL reports enqueue on the same stream, before launching mHC.
TP1 retains its original BF16 output and requires no collective or extra rounding.
The scratch is protected against attention inputs, residual coefficients, later
FFN reads, workspace, phases, Engram and retained sequence state. Non-finite
promotion/rounding sets the shared error flag; completion admission remains
mandatory. This fixes the reduction dtype but is not numerical qualification.

## Runtime safetensors catalog and device layout

`BackboneWeightCatalog::Create` consumes the admitted configuration, TP rank,
runtime shard prefixes and an explicit device payload budget. Each prefix is
exactly the eight-byte length plus the complete JSON header, with the actual
file extent supplied separately. The implementation reuses the model-agnostic
native `SafetensorsHeader` parser; it does not link the old model or invoke Python.
Its `pih_deepseek_v41_weight_inventory` CMake target is CPU-only and available
without a CUDA compiler; it is excluded from the default build until requested
or linked by a consumer.
At most 256 flat ASCII shard member names and 256 MiB of total prefix data are
accepted; each header retains the parser's 16 MiB/4096-tensor bounds. Slashes,
backslashes, traversal names, duplicate shard names and extra prefix payload
are rejected.

Every parsed tensor is joined to the sorted canonical inventory. Across the
entire shard set, all expected names must occur exactly once with exact storage,
physical rank/shape and byte length. The shared parser also requires data ranges
to cover each file exactly without overlap or gaps. Canonical FP4 uses safetensors
`U8` with physical `[out,in/2]` dimensions, low nibble first; source `I8` weights
must be explicitly converted/repackaged rather than admitted as a runtime alias.
FP8 weight and scale types are `F8_E4M3` and `F8_E8M0` respectively.

The resulting catalog owns shard names and tensor locations, so prefix buffers
can be released after creation. `Find(name)` returns a borrowed immutable entry
with shard index, absolute file offset, tensor geometry and device offset.
Device offsets follow canonical name order, start at zero, and reserve each
tensor with 256-byte alignment; `device_bytes()` includes this padding. Budget
checks occur before returning a usable catalog. This budget excludes caches,
workspace, scratch and allocator overhead.

The caller still has to authenticate and retain the exact shard objects, check
their unchanged identity while reading, validate numerical/padding payloads,
allocate through the plugin memory contract, upload and admit completion before
binding kernels. Passing a valid header does not authenticate any payload or
prove the supplied file extent came from a real file. No source-checkpoint
converter, executable model registration or hardware qualification is implied.

## Retained runtime weight files

`BackboneWeightFiles::Open(directory, config, world, rank, expected, budget)`
connects the catalog to real Linux files. `expected` contains exact member names,
byte lengths and nonzero SHA-256 digests. These must originate in an independently
trusted manifest bound to this configuration and rank; accepting hashes supplied
by the checkpoint itself is not authentication. This class does not implement
manifest signatures, publication authority or source-to-runtime conversion.

Opening walks an absolute canonical directory one component at a time without
following symbolic links, then opens each member with no-follow/nonblocking flags.
Only regular files with no write permission bits, a single hard link and the
exact expected extent are admitted. File identities must be distinct. All bytes
are hashed through retained descriptors using a reusable 1 MiB buffer. Headers
are then read from those same descriptors and joined to the exact inventory;
the total admitted file extent cannot exceed payload plus the header budget.
The independent `pih_deepseek_v41_weight_files` target is Linux/CPU-only.

Successful construction owns the descriptors and catalog. `Read(name, offset,
destination)` resolves the tensor internally and accepts only nonempty staging
spans of at most 1 MiB within that tensor. It uses positional reads, handles
interruptions/short reads, and checks descriptor and directory-entry identities
before and after each read. `Revalidate()` additionally reopens the absolute
directory to detect its replacement. Call it at loading/admission boundaries.
Do not upload a destination from a failed read; it may contain partial data.
The destructor closes retained files, not caller-owned staging/device resources.

Stat identity checks include inode/device, mode, owner/group, link count, size,
mtime and ctime. Sealed permissions and these checks detect ordinary mutation;
they do not defend against a privileged actor able to modify the underlying
filesystem while forging metadata. Deployment must enforce the immutable-store
trust boundary. These checks also do not validate finite numerical values,
Engram padding bytes, a CUDA upload, or completion of any model execution.

## Bounded native device weight upload

`BackboneWeightUpload::Start` borrows admitted `BackboneWeightFiles`, an exclusive
device arena exactly `catalog.device_bytes()` long, pinned staging, a stream and
an idle event. Arena and staging addresses must be 256-byte aligned and disjoint;
staging is a nonzero multiple of 256 bytes, at most 1 MiB. The arena must belong
to the current CUDA device. Allocation extents, stream/event ownership and all
resource lifetimes remain the caller's plugin-memory-contract responsibility.

`Advance` alternates a bounded file read/payload check/H2D copy with event polling.
No staging bytes are overwritten while a copy is pending. Tensor addresses are
derived from the catalog; callers cannot inject offsets. BF16/F32 infinities and
NaNs, FP8 E4M3 NaNs and E8M0 NaNs are rejected before upload. Packed E2M1 allows
every nibble encoding. Engram padded rows must have zero weight bytes and scale
bytes 127. Alignment gaps between tensors are unused, not initialized weights.

Every call rechecks the CUDA device and deadline. Once the final event completes,
file identities are revalidated before the operation becomes complete. Only
then does `Find(name)` expose a borrowed device tensor region for binding. The
files/catalog must outlive the upload object, including later `Find` calls.
The owner must not schedule readers against a partially uploaded arena.

An error permanently fails the operation. Copies may still be in flight even if
recording their completion event failed; there is no retry, implicit synchronize,
destructor cleanup or inferred permission to free memory. The owner must retire
the stream and retain all borrowed resources safely on failure or abandonment.
The Linux CUDA target `pih_deepseek_v41_weight_upload` connects file reads and
copy completion; it is not yet a registered model engine, automatic allocator,
complete kernel descriptor binder or evidence of hardware qualification.

## Binding uploaded weights to native descriptors

`uploaded_bindings.h` supplies `BindUploadedEmbedding`, `BindUploadedHead`,
`UploadedRoutedExperts` and `BindUploadedSharedExpert`. Each requires completed
upload and resolves canonical tensor names rather than trusting supplied weight
pointers. Binding returns a copy only after the corresponding descriptor has
passed geometry/alias validation. It does not enqueue GPU work or modify the
input descriptor on error. Scratch, token state and stream connections must
already be supplied by the caller.

The catalog retains its admitted configuration digest, world size and rank.
Embedding/head binding rejects a different TP partition; routed expert binding
also checks the configuration digest and backbone layer. The routed result is
ordered by increasing global expert ID within the rank's contiguous partition,
exactly `384/world` entries for `BuildExpertWorkspaceBatch`. Shared experts use
FP8 weights; routed experts use packed FP4. Final normalization uses canonical
BF16 and the output head uses the promoted FP32 shard.

`BackboneWeightUpload::ValidateScratch` protects the full arena, including
alignment gaps and weights belonging to later layers. These binding helpers
check their descriptor writes against it. A complete model owner must also call
this check on other allocations and operation writes, including expert workspace,
input uploads, caches, collectives and sampling scratch; these helpers cannot
protect buffers omitted from their arguments. These helpers do not assemble
the complete model engine.

`BindUploadedMhc` selects either the attention or FFN branch using an enum and
binds the three FP32 mixing parameters plus canonical BF16 normalization. It
preserves the caller's carried pre-mix connection; it never substitutes the
current sublayer's freshly generated coefficients. `BindUploadedRouter` binds
the FP32 gate matrix and text bias and rejects any image mask/bias descriptor.
`BindUploadedEngram` binds all six parameters only for layers 1/14, checks the
rank partition, requires the text path without an image gate mask, and sets the
canonical BF16 gate storage before validating the complete chain.

`BindUploadedAttentionPrepare` adds the query low-rank/expansion and window KV
projections, their scales and Q/KV norms to the mHC input. Its configuration
digest and TP size must match the uploaded catalog. The validated copy protects
projection/normalization/RoPE outputs and the window ring against the complete
weight arena. `BindUploadedAttentionOutput` binds the local attention sinks,
BF16 grouped output projection and FP8 final projection; configuration, local
group count and layer compression ratio are checked. Its full writable set
includes the FP32 collective scratch introduced for accurate output reduction.

`BindUploadedCompressed` binds only configured compressed-KV owners. Ratio-two
pooling uses FP32 value/gate projections, while ratio-one uses the BF16 value
projection without a gate. Both use BF16 normalization; incomplete steps retain
their existing absent normalization/cache outputs. Complete descriptor validation
checks the selected branch and emitted-row geometry before returning the copy.
Protection includes projected values/scores, pooling state, normalized output,
rotated KV and the persistent compressed cache.

`BindUploadedIndexer` checks the configured index owner and TP-local head count,
then binds FP8 query projection/scales and BF16 scoring projection. Key projection
and key normalization can be bound only by the KV owner when the step emits new
rows. Consumers cannot silently load a second copy of another layer's key weights.
The existing layer validator enforces key/candidate presence and compression-step
relationships. Query, score, selection, candidate-mask and key-cache writes are
all checked against the full immutable weight arena.

State/cache provenance, phase positions, collective rank agreement and
cross-stage lifetime checks remain responsibilities of the full block/model
owner; a successfully bound descriptor does not establish those properties.

## Uploaded weights in the execution entries

`EmbeddingOperation::Start`, `BlockOperation::Start` and `HeadOperation::Start`
now require a completed `BackboneWeightUpload` object. The former signatures
are removed; there is no compatibility overload accepting manually selected
weights. The block entry no longer takes a caller-supplied expert array: it
constructs that array in rank-local order from the uploaded catalog and binds
all mHC, attention, compressor, indexer, Engram, router and shared weights.

Data-edge wiring precedes weight binding, and graph admission follows both.
`WireSourcePhases` and `WirePreparedBlockSources` only construct connections;
they are not evidence that a descriptor is safe to execute. The block entry
still runs phase/source, block-connection, workspace, cross-stage lifetime and
transport admission before reservation and GPU submission. Scratch shapes and
intra-stage connections remain explicit caller inputs, not an automatic planner.

Embedding binds its vocabulary shard internally and protects the full uploaded
arena against token/hash staging and embedding writes before the first token.
The sequence retains that arena for its lifetime; blocks require matching
configuration, partition and embedding identity, and cannot switch arenas.
The head binds normalization and output weights from the same arena. Retained
region admission consequently protects every layer's weights during later
input, block, head and sampling operations, including otherwise unused gaps.

The upload object and its files/catalog must remain alive through each binding
call; device weight allocation must remain alive and immutable through all
operations and sequence retirement. These entries do not allocate a complete
model, establish distributed controller authority, or register a serving plugin.
TP1 full-model scheduling and MTP/vision execution remain unfinished.

## Persistent sequence cache planning

`SequenceCachePlan::Create(config, maximum_positions, byte_budget)` builds a
single-sequence cache arena with 256-byte-aligned segments and checks the budget
before allocation. Capacity is 1..1048576 tokens. Each of the 40 backbone layers
receives a BF16 `[128,512]` window ring. Only KV owners 2/8/14/20 receive compressed
BF16 `[floor(capacity/ratio),512]` and index-key BF16
`[floor(capacity/ratio),128]` storage. Ratio-two owners also receive two FP32
`[2,512]` pooling-state arrays. Nonowners do not get duplicate shared caches.
A ratio-two capacity of one correctly reserves no complete compressed rows.

`Allocate` requests the exact planned arena from the plugin's native memory
capability. The output ledger must be empty and ABI-initialized. Device, extent,
alignment, generation and provider status are checked; malformed/ambiguous
results remain in the ledger for quarantine, never silently discarded or freed.
`ValidateAllocation` checks a supplied allocation record. Neither method owns
retirement or automatically deallocates. The host-only
`pih_deepseek_v41_sequence_cache` target requires no CUDA compiler.

`BlockOperation::Start` now also requires the cache plan and its arena region.
It binds the layer's window, compressed/key cache and pooling-state addresses
before sequence preparation. Context capacity, configuration and arena identity
must remain unchanged across blocks. The arena cannot overlap uploaded weights.
Shared consumers still obtain populated cache prefixes from `BlockSequence`,
not from an uninitialized plan view. Planning/allocation does not initialize
GPU bytes or authorize reading unpopulated cache rows.

This is persistent cache planning only: transient activations, logits, routing,
selection masks, completion resources and weight storage remain separate.
The caller must own a valid device allocation, establish its lifetime and ensure
cache capacity is checked at request admission before input upload. A complete
model memory owner and automatic transient descriptor builder remain unfinished.

## Input/head/sampling boundary layout

`BoundaryPlan::Create(config, token_capacity, world, rank, device_budget,
host_budget)` plans two arenas for a token capacity of 1..4096 and TP1/2/4/8.
All segments start on 256-byte boundaries and remain distinct; no lifetime-based
aliasing is assumed. The device arena includes token IDs, both Engram hash planes,
embedding hidden/residual/pre, head collapse/norm, local and full logits, sampling
sort/weight/reduction scratch, statistics, candidate and shared error flag.
The pinned-host arena includes token/hash staging, completion error readback,
backbone expert-count readback and sampling candidate readback.

`Bind(device_arena, host_arena, tokens, stream)` checks exact arena sizes,
alignment, disjoint address ranges and the bounded token prefix. It returns
`BoundaryViews` ready to pass to the existing execution entries. When changing
from prefill to one-token decode, each view keeps its original base address and
uses the smaller logical extent; the error flag and fixed vocabulary/sampling
buffers are unchanged. The configured rank fixes local logits and expert-count
extents. Both memory budgets include alignment padding.

Embedding/head weights remain empty until their execution entries bind admitted
uploaded weights. The head's residual/pre placeholders are replaced by the
completed sequence output. Set the sampling parameters from the prepared token
ledger request before starting sampling; default parameters are not request
admission. Completion events, communicator and host-file/hash state remain
separate resources.

This planner does not allocate memory, check CUDA pointer attributes or submit
initialization work. The model owner must allocate the two arenas through the
plugin memory capability, keep them disjoint from weights/cache/other scratch,
initialize the device error flag before a new sequence, and retire operations
before reuse or release. Binding alone does not make any tensor contents ready.
Per-layer activations and full-model memory ownership remain unfinished.

## FFN transient arena and generated descriptors

`FfnPlan::Create(config, token_capacity, world, rank, byte_budget)` plans 22
distinct aligned segments for the backbone FFN. These hold mHC pre/post/comb,
collapse/norm, router logits/indices/weights, per-rank dispatch counts/slots,
shared-expert FP8 activation scratch and BF16 outputs, merged output and the
final four-stream residual. Routed-expert execution/accumulator storage stays
in the separately owned `ExpertWorkspaceOwner` allocation.

`Bind` accepts the current layer/token prefix, external incoming residual and
carried pre-mix, routed accumulator, shared error flag and stream. It checks
configuration/rank/arena identity, rejects external-state overlap, binds the
uploaded mHC/router/shared weights and runs the route/tail validators before
returning `FfnViews`. No weights or scratch pointers need to be manually placed
inside the FFN descriptors. Budget includes padding, and view addresses remain
stable when binding a smaller token prefix.

`BlockOperation::Start` now requires an `FfnPlan` and its arena. It generates the
FFN route/tail internally after connecting attention outputs, deriving the
accumulator from the actual expert workspace allocation. The caller no longer
supplies FFN members as an alternative path.
The complete block still performs cross-stage lifetime/alias admission before
submitting work. The accumulator must be produced and globally reduced by the
existing FFN operation; descriptor generation does not mark its contents ready.

The owner must keep incoming residual/pre and retained outputs disjoint from a
new FFN arena. A full layer scheduler must arrange separate/ping-pong storage
where those values remain live; it must not blindly reuse an arena at the next
layer. Allocation, retirement, attention-side transient planning and complete
model scheduling remain separate unfinished work.

## Attention preparation transient plan

`AttentionPreparePlan::Create` plans 18 distinct 256-byte-aligned segments for
the attention input path: mHC coefficients/collapse/norm, FP8 Q low-rank scratch,
normalized query rank, FP8 query expansion, separate rotated query, and window
KV projection/norm/rotation. Capacity is 1..4096 tokens and TP1/2/4/8; local query
storage scales with `64/world` heads. The budget includes alignment gaps.

`Bind` accepts the actual layer/start/token prefix, incoming residual and carried
pre, generated query phase table, persistent window ring, error flag and stream.
It rejects arena overlap with those external regions, checks configuration and
rank against uploaded weights, generates all stage connections, binds weights
and runs attention preparation validation. It does not allocate, initialize,
enqueue or authorize cache reads. Smaller prefixes retain fixed base addresses.

`BlockOperation::Start` now takes this plan and its arena, replacing the scratch
and projection descriptors in the attention preparation template. The remaining
template fields identify the step, incoming stream/error and optional source
stages. Generated normalized hidden is connected to compressor/scoring inputs,
normalized low-rank query to the index query projection, and emitted normalized
compressed latent to index-key projection. Missing local/emitted latent for a
key-producing stage is rejected before GPU work.

Attention output, compressor and Engram scratch have automatic plans described
below, together with indexer scratch. The full block retains cross-stage alias/lifetime
admission, and the owner must keep generated query, hidden and mHC coefficients
alive through their consumers. This is not yet a complete model memory owner or
serving plugin.

## Attention output transient plan

`AttentionOutputPlan::Create` reserves ten distinct aligned segments covering
KV/indices assembly, sparse attention, inverse RoPE, grouped BF16 output, FP8
final-projection activation scratch/output, FP32 reduction scratch and the
four-stream residual. The maximum token/context capacities and rank determine
the budget. KV capacity covers both full prefill concatenation and maximum
decode prefix; index capacity covers at most 640 choices per token. Binding uses
the exact active step extents without changing segment base addresses.

`Bind` takes the admitted attention preparation descriptor and this step's
compressed/selected prefixes. It validates the producer, derives ratio/step
geometry, rejects overlap with external inputs, binds uploaded sinks/output
weights and validates the complete attention residual chain. The FP32 reduction
segment is separate from the BF16 local output; the existing TP operation still
performs promotion, SUM and final rounding before mHC.

`BlockOperation::Start` now requires this plan/arena and no longer accepts a
`PreparedBlockLaunch` template. Step geometry is derived from the source and
phase declarations, shared cache/selection prefixes from the sequence, and the
attention output plus FFN descriptors from their plans. No old template overload
is retained. Full source/phase, weight, lifetime and transport validation still
precedes GPU submission.

This plan does not allocate or complete GPU work. The owner must retain source
queries, phases, cache prefixes and mHC coefficients through all consumers and
keep the arena separate from live inputs and other arenas. The full model
scheduler remains unfinished.

## Engram transient plan

`EngramPlan::Create(config, token_capacity, world, rank, budget)` plans five
distinct 256-byte-aligned segments: BF16 lookup output `[tokens,6144]`, FP8
projection activation and E8M0 scale scratch, BF16 projected `[tokens,25600]`,
and BF16 residual output `[tokens,4,5120]`. Capacity is 1..4096 tokens; smaller
bound prefixes retain fixed base addresses. The budget includes alignment gaps.

`Bind` admits only layer 1 or 14 with the matching uploaded configuration/rank.
It checks arena extent and disjointness from external hashes, incoming residual
and error flag, protects the uploaded weight arena, generates the stage
connections and binds all six Engram weights before chain validation. The
generated path is text-only and contains no image gating mask.

The block entry requires an Engram plan/arena instead of a manually constructed
optional launch. It activates the plan only on configured Engram layers and
uses the exact sequence-owned hash plane and incoming residual. Other layers
do not submit Engram work. Lookup TP reduction and GPU completion still occur
through the existing Engram operation; planning does not declare buffers ready.

The caller remains responsible for allocation and lifetime. The arena must stay
alive through gate-output consumers and must not be reused before block
retirement. Full model memory ownership remains unfinished; this does not
register an executable serving plugin.

## Compressor transient plan

`CompressorPlan::Create(config, token_capacity, budget)` reserves five distinct
256-byte-aligned segments for projection values/scores, pooled BF16 rows,
normalized latent and rotated latent. Capacity is 1..4096 tokens. Projection
values use FP32 for ratio two and BF16 for ratio one; bound extents shrink to
the actual token/emitted-row count without changing segment base addresses.

`Bind` accepts only configured compressed-KV owners, the normalized attention
hidden state, generated compressed phases, and persistent views from
`SequenceCachePlan`. It validates contiguous prefill/decode geometry, capacity,
configuration and disjoint storage, binds uploaded weights and validates the
complete compressor/cache chain. Ratio-two incomplete steps have no pooled
output, normalization or cache stage and require absent compressed phases;
they still update their two-row FP32 persistent state. Emitting steps preserve
unrotated normalized latent for the index-key projection.

The block entry now requires this plan/arena and generates compressor topology
from the configured role and step before sequence cache admission. It replaces
the caller's compressor descriptor with the planned launch after attention
preparation is bound. Nonowners have no compressor stage. No legacy overload or
Python inference fallback is provided. Full source/phase and cross-stage
lifetime validation remains required before any GPU submission.

The plan neither allocates nor initializes memory and does not prove GPU
completion. Its owner must retain the arena through index-key and cache
consumers. Complete model resource ownership and serving-plugin registration
remain unfinished. Syntax checks are not model
numerical, CUDA runtime or B300 qualification.

## Indexer transient plan and template-free block entry

`IndexerPlan::Create(config, token_capacity, maximum_positions, world, rank,
budget)` reserves eleven distinct 256-byte-aligned segments: FP8 query
activation/scales, projected/rotated query, projected/normalized/rotated key,
head weights, BF16 scores, selected indices and U8 candidate mask. Score/mask
capacity is `max(token_capacity^2, maximum_positions)`, not their product:
prefill starts at zero and all later steps contain one token. Local query/head
storage scales with `32/world`; supported descriptor ranks are TP1/2/4/8.

Binding requires an index-owning layer with a nonempty populated key prefix.
It derives emitted key rows, the causal selection offset, shared candidate-mask
consumption and layer-20 candidate production. Newly emitted local keys consume
unrotated normalized compressor latent and compressed phases. Other steps
require absent local latent/phases. The plan protects external inputs and the
uploaded weight arena, binds canonical weights and validates the indexer chain.
It neither performs TP reduction nor declares any device buffer complete.

`BlockOperation::Start` accepts plans/arenas, with no `IndexedSourcesLaunch`
argument. Sequence state identifies layer, position, token count, stream and
error flag; the FFN plan identifies rank. The entry generates
compressor/indexer ownership topology before sequence admission, obtains
shared key/selection/candidate prefixes from the sequence, then constructs all
source descriptors using their plans. No indexer is launched for an empty
compressed prefix. The old source-template signature is removed.

The owner must preserve selected indices and layer-20 candidate masks until
all sharing layers finish: a single indexer arena cannot be blindly reused for
every index owner. Existing cross-stage retention validation rejects overwrites
of published outputs. Whole-model allocation/lifetime scheduling and executable
plugin registration remain unfinished. Static syntax
checks do not establish model correctness or hardware qualification.

## Sequence-derived phase arena

`StepPhasesPlan::Create(config, token_capacity, budget)` reserves four disjoint
256-byte-aligned regions for query U32 positions/FP32 phase tables and compressed
U32 positions/FP32 phase tables. Each table uses 256 bytes per row. Binding keeps
fixed segment addresses and selects exact active extents. It rejects mismatched
configuration, oversized or noncontiguous steps, invalid stream/error storage,
and overlap between the full arena and the error flag.

Query positions start at the current sequence position with stride one.
Compressed positions exist only at configured KV owners with emitted rows;
their first position is `floor(start/ratio)*ratio`, with stride `ratio`, retaining
the original position of each group's first token. Incomplete ratio-two steps
have no compressed phase descriptor. Binding validates the complete phase chain
but does not initialize device positions, enqueue kernels or allocate memory.

The block entry requires the phase plan/arena instead of `StepPhasesLaunch`.
It derives layer, start, token count, stream and error flag from an idle sequence
whose embedding input or preceding layer has completed. Full weight-arena and
cross-stage liveness checks still precede phase generation on the stream. The
owner must retain phase storage through all consumers and retire in-flight work
before reuse. This completes descriptor planning, not whole-model allocation,
scheduling, serving-plugin registration or GPU qualification.

## Forty-layer backbone operation

`BackboneOperation::Start(config, sequence, resources, deadline)` starts one
ordinary-text prefill/decode backbone step after completed embedding at layer
zero. It owns the operation state, not the device allocations. `Advance` polls
the current block, waits for its device/error completion and expert workspace
retirement, then starts the next layer. All forty layers share one absolute
deadline. Output is available only after layer 39 publishes the completed step.

`BackboneResources` borrows plans, uploaded weights, expert workspace,
communicator and completion resources. It requires disjoint exact-size aligned
arenas: shared phase/Engram/cache/attention/output/compressor storage, two FFN
arenas selected by layer parity, and eight indexer arenas in layer order
2/8/14/20/24/28/32/36. Alternating FFN arenas preserve the preceding residual and
carried pre. Separate index-owner arenas preserve selections and layer-20
candidate masks until downstream sharing layers finish. Initial admission
checks extents, pairwise overlap, expert-workspace overlap and uploaded weights;
each block still performs full configuration, phase, cache and liveness checks.

Use the sequence and all resources exclusively until this operation completes
or fault retirement has reconciled outstanding work. Destruction of a running
operation poisons the sequence, without synchronizing or freeing buffers.
An error between layers also poisons the sequence and cannot be retried. An
output request after a later sequence step supersedes it is rejected.

This operation uses the current TP2/4/8 block implementation. It does not yet
allocate these resources, orchestrate embedding/head/sampling and controller
commit, provide TP1 whole-model execution, or register a serving plugin. No
model inference or hardware qualification is implied by host syntax checks.

## Ordinary inference-step operation

`InferenceOperation::Start` connects the existing embedding, forty-layer
backbone, head and sampling operations. Inputs are admitted configuration,
sequence/hash state, host token span, boundary plan/device/pinned-host arenas,
stream, borrowed backbone resources, controller sampling parameters/identity,
and one absolute deadline. Boundary configuration/rank, sampling parameters,
weight protection and overlap with backbone/expert arenas are checked before
input submission. The boundary plan supplies pinned expert-count/error storage;
the borrowed completion event is used sequentially by each stage.

`Advance` performs one bounded state transition or poll: embedding completion,
backbone completion, head completion, then last-rank sampling completion. Other
ranks complete after head logits have been gathered. `Logits` requires a
completed step and uses the head's stale-step checks; `Candidate` additionally
requires last-rank sampling. Tokens are snapshotted by input upload during
`Start`; device allocations, hash/sequence state, plans, weights and provider
resources must outlive the operation and any fault retirement.

The caller must initialize the device error flag to zero before the first
step and use these resources exclusively. A failed or interrupted active step
poisons its sequence without retrying, resetting hashes, synchronizing or
freeing in-flight buffers. Host allocation failures in continuation are also
converted to a failed sequence.

Sampling produces an observation only. The controller must still establish
cross-rank success and commit the token ledger before publishing output or
dispatching the next token. This implementation does not supply that transport,
an automatic generation loop, resource allocation, executable plugin loading,
TP1 whole-model execution, or hardware/model qualification.

## Combined inference memory layout

`InferenceMemoryPlan::Create` takes configuration, token/context capacities,
rank/world and separate device/pinned-host byte budgets. It composes all nine
component plans into seventeen disjoint aligned device regions: boundary,
phases, Engram, persistent cache, two FFN regions, attention preparation/output,
compressor and eight index-owner regions. The total device budget includes all
regions and alignment gaps, not merely each component's individual limit.
Pinned-host capacity comes from the boundary plan.

`Bind` validates exact full-arena extents, alignment, uploaded rank/configuration,
and disjointness from weights and the separately allocated expert workspace.
It returns the boundary subarena and borrowed `BackboneResources` for
`InferenceOperation::Start`; that entry fills host error/count addresses from
the boundary plan. Keep the memory plan at a stable address after binding,
because returned resources refer to its component plans. None of these views
own memory or establish GPU completion.

The device budget excludes uploaded weights, routed-expert workspace, CUDA/NCCL
handles and driver allocations. The owner must budget those separately and
allocate the planned device/pinned-host arenas before binding. Automatic
allocation and initialization are provided by the owner below; fault-retirement
reconciliation remains separate. Layout does not establish hardware support.

## Explicit inference memory owner

`InferenceMemoryOwner` borrows the memory plan, public CUDA memory/async
capabilities, retained context, stream and exclusive retirement event. `Allocate`
checks the capability tables and idle event, allocates the exact planned device
arena and pinned-host arena (host device ordinal `-1`), validates both allocation
identities, and enqueues zero initialization of the device arena followed by an
event. `Advance` must confirm initialization completion before the first step.
The entire arena is initialized once, never reset between decode steps.

`StartStep` binds the owned arenas and starts `InferenceOperation`; its
computation event must differ from the owner's retirement event. `Advance`
drives the step, records a final retirement fence only after normal completion,
and returns ready only after that fence completes. Logits/candidate access and
the next step are gated on readiness. A new step supersedes previous output
views. The caller still controls ledger commit and next-token dispatch.

The supplied expert workspace must be ready on the same stream and sized for
the exact current token count; this is checked before embedding submission.
Prefill and single-token decode therefore need appropriately sized workspace
owners even when they reuse the same inference arena.

`Release` is explicit and only allowed when ready. It releases pinned host then
device memory, retaining original records and per-allocation successful-release
flags. Any malformed response, allocation failure, initialization/step failure,
or ambiguous release quarantines the owner. In particular, a second allocation
failure preserves the first allocation; an unsuccessful device release after
host release never attempts to free the host a second time. Quarantined owners
cannot resume, reset or release through the normal API. Fault retirement must
reconcile the retained ledger after proving outstanding work is quiescent.

Destruction does not synchronize or free allocations. Destroying an active
step poisons the borrowed sequence. Keep plans, capabilities, sequence/hash
state, uploaded weights, expert workspace and handles alive through completion
or fault reconciliation. This owner does not allocate expert/weight arenas or
CUDA/NCCL handles, implement failure recovery, or establish device qualification.

## Rank completion and local token commit gate

`InferenceMemoryOwner::Receipt` is available only after a completed step and
confirmed memory retirement. Its in-process `RankStepReceipt` contains the
sampling identity, ordinal, processed length, rank/world and, on the last rank
only, the sampled observation. Construction is private to the inference path;
this is a local API invariant, not a cryptographic or network credential.

`RankCommitGate::Create` freezes a `TokenSamplingRequest` from ledger preparation,
TP2/4/8 world and absolute deadline. Before workers are dispatched the controller
must call `TokenLedger::MarkInFlight` for its reserved output credit. `Submit`
checks source rank, exact step identity/ordinal/length, candidate ownership and
candidate structure. Duplicate or inconsistent receipts close the gate. Receipt
arrival order is otherwise arbitrary. Missing ranks never imply completion.

After every rank has completed, `Commit` stages decoded token bytes in the ledger
and performs its local commit/publication. Expired deadlines and staging/commit
errors close the gate and fail the ledger. A successful gate cannot publish a
second time. No next-token dispatch is performed by this helper.

The actual controller still needs authenticated worker transport, serialized
fault/epoch handling and a final fault recheck immediately before commit. A
copied receipt cannot prove that a later fault has not occurred. Failed gates
must enter controller fault retirement; output credit must not be released
until outstanding workers are retired. Transport/retirement integrations and
the generation loop remain unfinished.

### Fixed rank receipt frame

`RankStepReceipt::Encode` emits exactly 256 bytes, little-endian, with no C++
padding: 8-byte `PIRANKR1` magic, U32 version/length, U32 rank/world/processed
length/candidate-present, four U64 sampling-identity fields, U64 ordinal,
180-byte candidate field sequence and four zero reserved bytes. Candidate
floating-point fields preserve their IEEE-754 binary32 bits. Non-last ranks
have a zero candidate payload; unused top-logprob entries are canonical zeros.

`RankCommitGate::SubmitWire` requires a complete frame and a source rank obtained
from the authenticated connection. It rejects truncation, trailing bytes,
unknown versions, nonzero reserved/unused fields and malformed candidate
presence before applying normal rank, request and candidate validation. Any
malformed frame closes the gate; it does not advance the completion mask.
Transport must assemble exactly one bounded frame before calling it and must
enforce its own disconnect/deadline handling. This codec neither opens a socket
nor authenticates peers, and does not replace the final controller fault check.

### Linux receipt transport

`RankReceiptChannel::Attach` duplicates an already connected descriptor with
`F_DUPFD_CLOEXEC`. It requires `O_NONBLOCK`, `AF_UNIX`, `SOCK_STREAM`, and exact
`SO_PEERCRED` PID/UID matching the supervisor's expected peer. The supervisor
also supplies immutable rank/world and send/receive direction. It must obtain
these identities from the admitted worker lifecycle, not from incoming bytes.
Use a connection whose kernel peer credentials match the actual process;
socketpairs created before fork may report their creator instead of the child.

The sender queues an encoded completed receipt and repeatedly calls `PollSend`.
The receiver calls `PollReceive` with the current commit gate. Each poll makes
at most one nonblocking syscall, retaining partial-frame progress. EINTR or
EAGAIN returns pending; timeout, EOF, syscall failure or invalid complete frame
permanently fails the channel. Send uses `MSG_NOSIGNAL`; successful send means
bytes were accepted by the socket, not that the controller committed the token.
Stop polling a rank for that gate after its receipt is accepted.

Immediately before local commit, the serialized controller checks every
receiver with `CheckQuiet`, verifies supervisor liveness/fault state, and then
commits the ready gate. Unexpected queued data, EOF or an unverifiable peek
closes the gate. A quiet socket is only a point-in-time observation: it does
not prove the process is alive, prevent a subsequent fault, or replace the
supervisor's ordered fault/epoch protocol.

No other code may read/write these endpoints. The channel closes only its owned
duplicate; the caller must manage the original descriptor and ensure inherited
duplicates cannot keep dead-worker connections artificially alive. Listener,
worker bootstrap wiring, generation requests, process-liveness integration and
fault retirement remain unfinished. No socket/model execution test was run.

### Controller receipt collector

`RankReceiptCollector::Begin(ledger, plan_seq, channels, deadline)` requires
ready receiver channels in exact rank order. It prepares the ledger request
and output reservation, creates the rank gate and marks output credit in-flight
before returning. Dispatch only the returned `request()` to workers. A failure
after that boundary retains the reservation for explicit fault retirement.

`Poll` visits each rank not yet received, once per call, and never blocks waiting
for I/O. Completed ranks are not polled again for the same gate. `Candidate`
requires all receipts and allows the controller to decode the sampled token.
`Commit` checks every receiver boundary before calling the gate's local ledger
commit; successful publication is one-shot. The controller may then obtain the
ledger's pending input for its next dispatch. No automatic generation loop is
started by this helper.

Keep ledger and channels alive and exclusively borrowed by the collector. A
failure or destruction before successful commit fails the ledger but does not
release in-flight output credit. The supervisor must serialize epoch/fault
events with this collector and call `Fail` before commit on observed worker
failure. PID liveness, worker request transport, decoding and post-failure
retirement remain separate integrations; quiet sockets alone do not prove
global success.

### Worker inference request frame

`InferenceRequest` contains a ledger sampling request and up to 4096 token IDs.
Its 16,640-byte frame is explicitly little-endian: a 256-byte header followed
by 4096 U32 token slots. The header carries `PHIFREQ1`, version/length, the four
identity fields, processed length/count, binary32 temperature/top-p, top-k,
logprob flag, seed/ordinal, top-logprob count and seventeen suppression slots.
Reserved header bytes and unused token/suppression slots must be zero.

Encoding and decoding both validate identity, sampling constraints, vocabulary
bounds, context length and prefill/decode shape. More than one token is permitted
only for a prefill starting at zero; later requests contain one token. Exact
frame length is required, with no native ABI/padding dependence. The bounded
frame deliberately avoids allocations based on untrusted counts.

After authenticated reception, `InferenceMemoryOwner::StartRequest` verifies
that local sequence position plus token count equals the declared processed
length, then enters the existing step pipeline. It does not authenticate a
controller, establish epoch authority or prove the input token was committed;
those remain worker-controller protocol obligations. Request socket framing,
bootstrap integration and the automatic generation loop are separate from this
codec; the request channel below provides socket framing.

### Nonblocking worker request channel

`RankRequestChannel` uses the same connected Unix socket admission as the
receipt channel: nonblocking stream, supervisor-supplied rank/world and exact
kernel peer PID/UID. It owns a CLOEXEC duplicate through its private endpoint.
Use dedicated request sockets; request and receipt consumers must never read
from the same underlying stream. No endpoint may be concurrently shared with
another reader/writer or have its blocking flags changed externally.

The controller queues a validated `InferenceRequest` and calls `PollSend` until
the frame is sent. The worker calls `PollReceive`, which returns an empty
optional while pending and a validated request only after all 16,640 bytes
arrive. Each poll performs at most one syscall and preserves partial progress.
EAGAIN/EINTR is pending; timeout, EOF, I/O failure or decoding failure makes the
channel permanently unusable. Successful send is not worker admission or a
token commit. Failed/partial requests must enter supervisor fault handling,
not automatic resend.

The worker must still validate admitted epoch/sequence identity, select the
proper expert workspace, call `StartRequest`, drive inference to retirement and
send its completion receipt. Listener/bootstrap wiring and that integrated
worker loop are separate; the loop below now connects these stages. These changes were syntax checked, not exercised
with sockets, GPUs or a model checkpoint.

### Rank worker execution loop

`RankWorkerLoop::Create` admits one fresh sequence with frozen epoch, sequence
generation, sampling-config identity, base sampling parameters, prompt length
and a single absolute sequence deadline. It borrows initialized inference
memory, admitted weights/hash state, prefill/decode expert workspaces, dedicated
request/receipt channels, communicator and computation event. Channel rank/world
must match each other and the uploaded rank. The caller drives `Poll`; no thread
or blocking wait is started.

The loop receives a complete request, rejects nonincreasing plan IDs, identity
or base-parameter drift, incorrect sampling ordinal and invalid first/decode
token counts, then calls `StartRequest`. It chooses the prefill workspace at
position zero and the single-token workspace later. After inference and memory
retirement complete, it queues the rank receipt and finishes sending it before
receiving the next request. Stochastic ordinal advances once per sent result;
greedy ordinal stays unchanged. Per-step suppression comes from the admitted
controller request and remains subject to sampling validation.

Any failed poll terminates the loop, poisons the sequence and quarantines memory
without replay or freeing active allocations. Destroying an executing/sending
loop has the same effect; destruction while idle in receive state does not
free the borrowed memory. A subsequent request is accepted only through the
authenticated controller endpoint, but receipt send alone is not global token
commit. The controller must dispatch the next input only after its commit.

This loop does not spawn workers, load their artifacts, create CUDA/NCCL handles,
monitor supervisor pidfds, send a terminal-control message or implement the
controller's automatic generation loop. Those integrations and executable
plugin registration remain unfinished; no model execution is claimed.

### Controller generation loop

`GenerationLoop::Create` admits a fresh ledger, matching prompt tokens, a borrowed
129280-entry token-byte table, request/receipt channels in rank order, first plan
sequence and one absolute deadline. The tokenizer adapter must provide admitted
immutable raw token bytes; each entry is bounded to 8192 bytes. The prompt is
copied at admission; channels, ledger and byte-table storage remain borrowed.

`Poll` prepares output credit, queues the same ledger-derived request to every
rank, incrementally sends each frame, collects all receipts, obtains the sampled
token's bytes and commits through the collector. It never sends the next token
before local commit. Each call makes bounded progress without sleeping. If no
output credit is available, preparation remains pending without dispatching
workers or consuming a new plan ID. Other errors fail the sequence ledger.

After commit, `TakeOutput` transfers the publication lease to the consumer,
which reads/releases it through the originating output queue. Until that lease
is taken, the loop remains output-ready. After transfer, stop/length completion
starts terminal acknowledgement; otherwise the ledger's committed pending input forms a new
single-token request. Publication credits can still apply backpressure until
the consumer releases earlier leases. Exhausted plan IDs fail future generation
without discarding an already accepted output lease.

The supervisor must serialize fault/epoch events with `Poll` and call `Fail`
before any subsequent commit when a worker fault is observed. This loop is not
the process supervisor, tokenizer loader, HTTP/SSE serializer or terminal worker
shutdown protocol. Those integrations, executable plugin registration and GPU
qualification remain unfinished. The loop was syntax checked only.

### Normal terminal handshake

Request frames now require version **2** (the `PHIFREQ1` family magic is
unchanged). U32 at offset 164 is the terminal flag, restricted to 0/1; reserved
bytes begin at 168. Version 1 is rejected rather than accepted through a legacy
decoder. Ordinary requests retain their existing shape checks. Terminal
requests have zero token count and an entirely zero token payload, but keep
the final processed length, sampling identity and post-commit ordinal.

After the consumer takes the final stop/length output lease, the generation
loop queues terminal requests to all ranks and waits for their acknowledgements.
The terminal plan ID references the last completed inference plan; no new
inference plan or output credit is consumed. Workers accept it only after that
step's receipt has been sent, with matching frozen identity/parameters, ordinal,
processed length and ready inference memory. `StartRequest` explicitly rejects
terminal requests so they cannot enqueue inference.

Terminal acknowledgements use the 256-byte receipt format with flag value 2,
zero candidate payload and matching identity/rank/world/processed length/ordinal.
The ordinary token gate rejects them. `PollTerminal` compares the exact expected
frame; mismatch, disconnect or deadline expiry fails termination. A worker enters
complete only after sending its acknowledgement; the controller enters complete
only after every acknowledgement arrives. No extra stochastic ordinal increment
occurs for termination.

Final token publication and worker termination are separate events. The service
adapter must keep polling after taking that publication and defer its successful
terminal response until generation is complete. Afterwards it may explicitly
release ready memory and retire handles under supervisor ownership. This normal
handshake does not implement cancellation, crashed-worker cleanup or peer-process
reaping, and does not automatically free resources or exit a worker process.

### Rank pidfd liveness gate

`RankProcessWatch::Attach` takes rank-ordered pidfds retained by the supervisor
from worker creation and expected positive PIDs. It owns CLOEXEC duplicates,
rejects duplicate process identities, and reads at most 8192 bytes of each
duplicate's `/proc/self/fdinfo` to verify its `Pid:` identity. The watch never
opens a replacement pidfd using a potentially recycled numeric PID.

`CheckLive` performs a zero-timeout poll of every retained pidfd. Only a clean
no-event observation succeeds; process exit, invalid descriptors, interrupted
or failed polling permanently fails the watch. It does not signal processes,
consume exit status, reap children or free GPU allocations. Destruction closes
only the owned duplicates; original pidfds remain the supervisor's responsibility.

`GenerationLoop::Create` now requires this watch and verifies that every request
and receipt endpoint's authenticated peer PID matches the rank binding. Active
generation polls check liveness, and collector commit rechecks it immediately
after quiet socket checks and before local ledger acceptance. The watch must
stay at a stable address and outlive the generation loop. A worker must remain
alive after its terminal acknowledgement until supervisor-directed teardown.

This adds concrete process-exit detection, not an atomic guarantee against a
future crash. Epoch/fault events and commit still require a serialized controller
policy. Normal process teardown, failed-rank termination/reaping, NCCL abort and
GPU resource reconciliation remain unfinished integrations. No process-lifecycle
or model execution test was run.

### Fault process retirement

`RankProcessRetirement::Create(processes, ledger, kill_after, deadline)` retains
its own pidfd duplicates, freezes the ledger's pending plan for reconciliation,
and closes both liveness admission and ledger execution. It does not signal
processes until polled. The supervisor must bind the correct process set to
that ledger and retain exclusive parent/subreaper ownership of rank exit status.

Each `Poll` nonblockingly reaps already exited ranks using `waitid(P_PIDFD)` and
checks exact PID/exit classification. Remaining ranks receive SIGTERM via
`pidfd_send_signal`; after `kill_after`, still-live ranks receive SIGKILL. There
is no numeric-PID fallback, blocking wait or automatic retry after a terminal
error. EINTR remains pending; ECHILD is an ownership failure, not proof of
retirement. `reaped_mask()` retains partial progress on failure. Destruction
closes only owned pidfds and never implicitly kills or reaps processes.

Only after every rank is reaped may `DiscardOutput` reconcile the captured
unaccepted output reservation. Already committed publications are untouched,
and failed ledger state remains failed. `AbortRetired` now allows that cleanup
on failed ledgers; it still requires the exact pending plan and caller proof of
retirement. Do not release active credit merely because a termination signal
was delivered.

Process reaping establishes that these ranks cannot produce another receipt;
it does not independently prove GPU/NCCL cleanup, cover unsupervised descendant
processes, or permit freeing another process's CUDA allocation handles. Cgroup
containment, device cleanup reconciliation and supervisor wiring remain separate
work. No process was signalled or reaped during development checks.

### Supervised generation session

`GenerationSession::Create` takes ownership of a fresh generation loop, borrows
rank-ordered lifecycle channels and cgroup owners and accepts a startup deadline, termination
grace interval and total retirement timeout (at most five minutes).
The supervisor must be the ranks' parent/subreaper. The session derives its
process watch and ledger from that loop, so cleanup cannot accidentally be
supplied a different ledger at cancellation time.

`Poll` drives generation and captures committed publication leases. `TakeOutput`
transfers a captured lease to the consumer without releasing queue credit. While
output is waiting for consumption, polling still checks rank liveness and the
sequence deadline. The caller must continue polling through the final worker
handshake before reporting successful completion.

Generation errors or explicit `Cancel(non_ok_reason)` close dispatch/commit and
enter pidfd retirement with a fresh cleanup deadline. Polling then drives signal
escalation/reaping and discards only the captured unaccepted output reservation.
`kFailedRetired` means those rank processes were reaped, pending output credit
was reconciled and owned cgroups were emptied/removed. `kFailedUnreconciled` means that proof is missing; original failure,
retirement failure and partial reap mask remain available. Neither state proves
CUDA/NCCL resource cleanup. Already committed leases remain retrievable even if
cancellation occurs before the consumer takes them.

No background cleanup is started. Keep the session and its borrowed ledger,
channels/process watch alive while polling retirement, and transfer unresolved
records to the supervisor if continuation is impossible. Destroying the session
does not implicitly signal/reap processes or release publication credit. Plugin
service integration, process bootstrap and device-level reconciliation remain
unfinished; only syntax/documentation checks were performed.

### Worker CUDA handle owner

`WorkerHandles` borrows the public device, resource and async capability tables
from admitted plugin activations. `Create(expected_sm_major, expected_sm_minor)`
asks the backend to prepare the selected ordinal, retains/binds its primary
context, creates a nonblocking stream and four distinct disable-timing events.
The events are assigned to computation/upload, inference retirement, prefill
expert retirement and decode expert retirement respectively. Upload must finish
before computation reuses its event. The expected SM comes from the selected
deployment profile, not a hardware-support guess made by this helper.

Creation failures preserve every returned handle in `WorkerHandleLedger` and
quarantine the owner. It never guesses whether a failing provider call allocated
a resource. The ledger remains available for supervisor reconciliation.

For normal teardown, stop all submitters and ensure NCCL has no pending enqueue
before `BeginRetirement`. `PollRetirement` nonblockingly observes a final stream
event. Release all communicator/memory owners before `Release`, which destroys
events in reverse order, destroys the stream, then releases the primary context.
Each successful release is recorded separately; a later failure does not retry
earlier destroys. No destructor frees or implicit stream synchronization occur.

This helper does not establish quiescence after a stuck NCCL enqueue, create a
communicator, authenticate plugin tables or complete worker bootstrap. Those
are still required integration work. Syntax checks passed after fixing a
missing standard header; no CUDA resources were created during development.

### Worker NCCL communicator owner

`WorkerCommunicator` is a model-owned, exclusive worker-thread adapter to
`pih.transport.nccl`; NCCL 2.31.2 state lives only in that provider.
The supervisor obtains a 128-byte ID from `NcclBootstrapBroker` and distributes
the same ID through authenticated bootstrap channels to exactly 2, 4 or 8
ranks. Distribution, epoch binding and secret-buffer erasure remain caller
responsibilities; generating an ID does not authenticate the participating ranks.

After `WorkerHandles::Create` binds the selected device, call `Start` with the
ID, world, rank, device ordinal and absolute steady-clock deadline. Poll until
true before `Borrow` exposes an opaque provider handle to the worker execution
loop. The provider checks the compiled/runtime NCCL release, world, rank and
CUDA device and selects nonblocking mode. Never submit collectives while
initialization is pending.

For normal teardown, stop all submitters, wait for their enqueue state to finish,
call `BeginFinalize` on all ranks and poll each owner to completion. Only then
call `Release`. Keep the CUDA context alive throughout. Finalization is not
proof that unrelated CUDA work or borrowed memory can be released; memory and
stream owners still enforce their own completion fences.

On faults, stop submitters before an explicit `Abort`. The owner retains handle
values for reconciliation and makes at most one abort/destroy attempt. A failed
destroy cannot be followed by abort because handle validity is ambiguous. NCCL
host calls themselves are not preempted by this helper's polling deadline;
the supervisor must retain its independent process retirement timeout. No
destructor invokes NCCL, CUDA synchronization or resource cleanup.

This component is included in the worker-loop static target and syntax checker.
It has not yet been wired into an executable worker bootstrap or validated on
multiple GPUs; it does not register a deployable V4.1/B300 capability.

### Weight allocation and upload owner

`WeightMemoryOwner` joins the authenticated `BackboneWeightFiles` catalog,
public CUDA memory/async contracts and `BackboneWeightUpload`. Construct it
with borrowed files, admitted capability tables, retained context, worker stream
and exclusive upload event. Keep all those dependencies alive until release.
`Start(device_budget, staging_bytes, deadline)` checks the catalog-sized device
allocation against the budget; staging must be a nonzero multiple of 256 bytes
and at most 1 MiB. Pinned-host allocations use device ordinal -1, not the worker
GPU ordinal. Returned allocation generation, kind, extent, alignment and
non-overlap are checked before upload begins.

Call `Advance` until true; each upload continuation performs at most the bounded
chunk work implemented by `BackboneWeightUpload`. Only then does `Upload` return
the borrowed immutable weight view for inference. Keep that view's owner alive
through every operation that uses it. The weight budget excludes expert scratch,
inference caches and allocator overhead; deployment must budget their sum.

Before normal teardown, stop every consumer and finish pending NCCL enqueues.
All weight reads must occur on the configured stream, or have been explicitly
joined into it. `BeginRetirement(deadline)` records a final event, and `Advance`
must complete before `Release` frees staging and device weights. The upload
event may be reused for computation only after upload completion and must be
exclusively returned to this owner before retirement. Upload completion alone
does not authorize freeing weights after subsequent inference use.

Allocation/upload/retirement failures quarantine the owner and preserve its
allocation ledger. There is no destructor cleanup, rollback of partially
allocated resources or retry of ambiguous frees. The supervisor must reconcile
faults or retire the worker process. This owner is compiled with the upload
target; executable bootstrap integration and GPU validation remain outstanding.

### Worker memory startup assembly

`WorkerMemory` owns the weight, inference and two expert-workspace owners at
stable addresses. It borrows authenticated files, the inference memory plan,
public capability tables and ready `WorkerHandles`. These dependencies must
outlive the assembly, including fault reconciliation. The selected device must
match the handle owner, and plan/catalog config digest, world and rank must
match before any allocation.

`Start` checks one device budget covering weights, inference storage, an exact
prompt-sized expert workspace and a separate one-token decode workspace. It
also checks the sum of inference pinned-host storage and weight-upload staging.
The checks use subtraction to avoid overflow. Both expert allocations exist
even for a one-token prompt. CUDA/NCCL implementation overhead is outside these
payload budgets and must be reserved separately by the deployment profile.

`Advance` first drives weight upload to completion, then allocates both expert
workspaces and initializes the inference arena. It checks all six returned
device/host allocation ranges for overlap across owners before declaring the
assembly ready. `Views` then supplies the four references required by
`RankWorkerResources`; no child owner or borrowed plan may move or disappear
while the worker loop uses them.

After the worker stops, inference memory and both workspaces must have retired
to their ready states, and no future consumer or pending NCCL enqueue may exist.
`BeginRetirement` records the weight consumer fence. Subsequent `Advance` polls
it and releases inference storage, decode workspace, prefill workspace, then
weights. Only after successful memory teardown may the caller retire/release
CUDA handles. The upload/computation event must be exclusively available again.

Any partial failure quarantines the assembly without dropping child allocation
records. Its read-only ledger accessors support supervisor reconciliation;
destruction does not free GPU resources or retry failed releases. This connects
the memory startup components but does not yet provide worker process launch,
authenticated bootstrap dispatch or a runnable inference service.

### Single-rank runtime assembly

`WorkerRuntime` connects the handle, communicator, memory and request-loop
components for one rank of a TP 2/4/8 sequence. Construct it with admitted files,
a stable inference plan, public CUDA capability tables, a fresh Engram hash
state and authenticated request/receipt plus lifecycle command/notice channels. Channel world/rank and peer
PID must match each other and the plan. Hash position must initially be zero;
the plan token capacity is the exact initial prompt length, not a batch maximum.

`Start` validates sampling/bootstrap metadata, creates CUDA handles and starts
nonblocking NCCL initialization. `Poll` drives connection completion, memory
startup, then `RankWorkerLoop` execution. Startup has an absolute deadline;
the separately supplied whole-sequence deadline must be later. A positive
retirement timeout of at most five minutes bounds subsequent cleanup. The runtime
does not generate/distribute IDs, accept arbitrary clients, spawn processes,
authenticate capability activations or supply missing tokenizer/map admission.
All borrowed dependencies and the runtime must remain at stable addresses.

Once the terminal receipt has been sent, the runtime remains in
`kAwaitingRetirement`; it does not exit or free resources on its own. The
supervisor must observe all ranks' terminal receipts before sending a `kRetire`
lifecycle command to every rank. `Poll` receives that authorization within the
whole-sequence deadline, then finalizes/destroys NCCL, fences and releases memory,
and finally fences/releases CUDA handles. It sends `kReleased` on the notice
channel before entering the runtime's `kReleased` state. Failure to send that
notice is a failed runtime even if local resource teardown already succeeded;
the retained child ledgers distinguish those cases.

Errors stop the local loop and leave the runtime failed with its resource
ledgers intact. Explicit `AbortCommunicator` is allowed only in that state;
successful abort does not prove device memory is safe to free. The supervisor
still owns epoch failure and process retirement. There is no automatic retry,
destructor GPU cleanup or Python fallback. This runtime is a native static
component, not yet a launched worker executable or registered model plugin.

### Lifecycle wire protocol

`RankLifecycleChannel` uses two dedicated one-way endpoints per worker: a
supervisor-to-worker command channel and worker-to-supervisor notice channel.
It reuses the same connected, nonblocking Unix stream and `SO_PEERCRED`
admission as token channels, owning a CLOEXEC duplicate of the admitted socket.
Never attach multiple protocol readers or writers to the same stream. Every
endpoint for a worker must agree on rank/world and authenticated peer PID.

Frames are exactly 64 bytes, little-endian: magic `PIHLIFE1` at 0, version 1
at 8, length at 12, kind at 16, rank at 20, world at 24, zero padding at 28,
epoch at 32, sequence generation at 40, sampling config ID at 48 and zero
padding at 56. Kinds are ready=1, retire=2, released=3. The identity is the base
sequence identity with plan sequence zero. Receivers compare the entire frame
against their expected kind and identity, rejecting noncanonical padding and
wrong order. Changing expectations during a partial frame poisons the channel.
Each poll makes at most one bounded nonblocking socket call; EOF, malformed
frames and deadlines permanently fail the endpoint.

The runtime publishes ready only after NCCL initialization, weight loading and
memory initialization finish. The supervisor must gather ready notices from
all ranks before sending inference requests. It may send retire only after
all terminal token receipts have been admitted, then gather released notices
before normal process reaping. `GenerationSession` now enforces these all-rank
barriers; an individual lifecycle frame is not proof of global completion.

### Supervisor lifecycle integration

`GenerationSession::Create` requires rank-ordered lifecycle command senders and
notice receivers in addition to its fresh generation loop and fault-retirement
settings. Every lifecycle endpoint must match the rank/world and exact PID from
the loop's process watch. The startup deadline must be future and earlier than
the whole-generation deadline. The lifecycle identity comes directly from the
ledger's immutable base identity, not from a separate caller assertion.

The session starts in `kWaitingReady`, polls each missing ready notice once per
continuation and checks process liveness before entering `kRunning`. It never
polls the generation loop or reserves token output while startup is incomplete.
Normal output backpressure and publication lease ownership remain unchanged.

After generation has received all terminal token receipts, the session verifies
all ranks still live, queues retire commands and enters `kSendingRetire`, then
`kWaitingReleased`. All command sends and release notices share one bounded
retirement deadline. A rank that already sent its release notice may exit while
other ranks are cleaning up; therefore teardown does not apply the all-ranks-live
check used during generation. Missing/malformed notices or EOF still fail the
barrier and invoke pidfd fault retirement. No external reaper may race this path.

`kComplete` requires every release notice followed by successful OS process
reaping and empty owned-cgroup removal, not merely final token receipts. Cancellation during startup or normal
teardown uses the existing fault retirement and retains committed output leases.

### Normal worker process exit

After collecting every release notice, `GenerationSession` enters `kReaping`
and calls `RankProcessWatch::PollNormalExit` under the same absolute teardown
deadline used for retire commands and release notices. The worker executable
must return exit code zero after `WorkerRuntime` reaches `kReleased`. The parent
must not have another reaper, SIGCHLD auto-reap policy or concurrent wait operation
that consumes these child statuses.

Each continuation performs at most one `waitid(P_PIDFD, WEXITED | WNOHANG)` per
unreaped rank. No signal is sent during normal reaping. The exact returned PID,
exit classification and zero status are checked. Starting reaping permanently
closes the watch to new inference admission. EOF or a release message alone is
not proof of successful process exit.

Consumed exit records are retained in the watch's rank mask, including abnormal
exits. If normal reaping fails or times out, the session starts fault retirement
with that mask; already reaped ranks are neither reaped nor signalled again.
Remaining ranks use the existing pidfd-only terminate/kill escalation. A failure
to obtain exclusive parent/subreaper ownership remains unreconciled, not assumed
success. The session's `reaped_mask` reports progress for both paths. This
completes the supervision-side normal exit flow; spawning the executable and
wiring its admitted resources/channels are still required.

### Post-exec rank connections

`RankSocket` creates Linux abstract Unix stream endpoints without filesystem
socket entries. The supervisor calls `Listen` before spawning a worker. Use
one distinct unpredictable name per rank and protocol stream; names must be
32–96 lowercase letters/digits/dots/hyphens. Include at least a fresh 128-bit
random nonce; accepted syntax alone is not an entropy guarantee. Abstract
addresses are scoped to the network namespace and are not filesystem-protected.

The exec'd worker calls `Connect` with the admitted supervisor PID/UID and
deadline, then `PollConnected` until true. The supervisor calls `PollAccept`
with the exact worker PID from its spawn/pidfd ledger and expected UID. Both
sides validate `SO_PEERCRED` and connected Unix peer identity before
`ConnectedDescriptor` exposes a borrowed FD. Feed that FD to the corresponding
request, receipt or lifecycle channel's `Attach`; it independently validates
identity and retains a CLOEXEC duplicate. Close the temporary socket owner after
attaching, retaining only one protocol reader/writer for that stream.

Listeners and connections are nonblocking and CLOEXEC. Each accept continuation
performs at most one accept. A listener admits only one peer; unauthorized
connections consume the admission and fail the epoch instead of silently
accepting another worker. Pending connects are checked for socket errors and
connected peer identity; AF_UNIX backlog-full `EAGAIN` is not treated as an
in-progress success. All admission paths enforce absolute deadlines and never
automatically reconnect. Destructors close owned descriptors only, with no
unlink, signal or process cleanup.

Do not use a socketpair created before fork as a substitute: its recorded peer
credentials can identify the creator rather than the exec'd child. Socket
names, namespace placement and PID identity distribution still belong to the
supervisor bootstrap. This helper does not spawn processes; no socket/process
execution tests were run for this change.

### Four-channel rank bundle

`RankChannels` owns the request, receipt, lifecycle-command and lifecycle-notice
channels, in that order. Supply four distinct abstract endpoint names for one
rank; names must also be unique across ranks and epochs. The supervisor calls
`Listen(names, rank, world)` before spawn, then `BeginAccept(child_pid, uid,
deadline)` after obtaining the exact child identity from its spawn/pidfd ledger.
The exec'd worker calls `Connect(names, rank, world, supervisor_pid, uid,
deadline)`. The name views are consumed during these calls and need not outlive
them; the admitted identity and absolute deadline cannot be replaced mid-flight.

Both sides call `Poll` until true. Each continuation performs at most one
accept/connect observation per unfinished stream. Attach independently validates
each peer. Supervisor directions are send/receive/send/receive; worker directions
are the inverse. A failed or partially admitted bundle cannot expose `Views` and
cannot reconnect. Invalid/duplicate names are rejected before opening sockets.

On complete admission, `Views` supplies references directly usable by
`WorkerRuntime`, or by the rank-ordered channel arrays passed to `GenerationLoop`
and `GenerationSession`. Keep the nonmovable bundle alive while any reference is
in use. Temporary accepted sockets, connectors and listeners close after the
corresponding protocol retains its own descriptor. Bundle destruction closes
owned descriptors only; process retirement remains the supervisor's obligation.

This assembles the real transport endpoints but does not launch a worker or
distribute its configuration, authenticated artifact descriptors or NCCL ID.

### Explicit native worker spawn

`WorkerSpawn` is a Linux `clone3`/`execveat` primitive for a dedicated,
single-threaded supervisor before CUDA initialization. Supply an authenticated,
sealed `WorkerExecutable` owner (described below); arbitrary raw executable FDs
are no longer accepted by the launch API. Supply a sealed bootstrap
FD (WRITE/GROW/SHRINK/SEAL), an already configured delegated cgroup-v2 directory
FD, explicit argv including argv[0], and a `WorkerEnvironment` owner. No ambient
environment is copied and arbitrary key/value environment arrays are no longer
accepted. Library and system configuration admission remains separate, as noted
below. Child stdin/stderr are inherited; child stdout is redirected to stderr
before exec so diagnostics do not corrupt supervisor token output.

`Start` uses `CLONE_PIDFD | CLONE_INTO_CGROUP`, so the child is tracked by pidfd
and placed in its cgroup from birth. Unsupported/denied clone3 or cgroup placement
fails without a fork fallback. The child sets parent-death SIGKILL and no-new-
privileges, clears its signal mask and restores common termination dispositions.
Bootstrap is mapped to FD 3. All other descriptors from 4 upward are marked
CLOEXEC before executing the retained binary FD. The executable must be ELF,
not a script requiring an interpreter to reopen a CLOEXEC descriptor.

The nonblocking exec-status pipe reports child setup/exec errno. `PollExec`
retains the PID/pidfd even on error. Clean pipe EOF with a live pidfd advances
only to `kAwaitingChannels`: it is not proof of model readiness or a substitute
for authenticated connections and the all-rank ready barrier. The launcher
does not parse the bootstrap payload or claim executable service integration.

After clone, `binding()` can populate the rank process watch even if exec fails.
Before a full rank watch exists, partial-startup cleanup may explicitly `Kill`
and `PollKilled(deadline)` each launched child. Once custody passes to the watch,
only its supervisor session may signal/reap; do not also call those launcher
methods. Auto-reaping SIGCHLD policy is rejected; competing reapers are forbidden.
Destructor cleanup closes descriptors only, never signals or waits. Keep the
launch ledger until custody transfer or proven reap. Cgroup descendant cleanup,
bootstrap resource admission and worker executable wiring remain outstanding.

### Sealed worker bootstrap envelope

`WorkerBootstrap` version 1 is a fixed 2048-byte little-endian envelope. It
contains supervisor PID/UID, TP world/rank, device and expected SM, exact prompt
length, context limit, retirement timeout, base sampling identity/parameters,
aggregate device/host/staging budgets, absolute CLOCK_MONOTONIC startup and
sequence deadlines, 128-byte NCCL ID, four endpoint names, artifact directory
and plugin-lock path. Workers and supervisor must share a Linux time namespace.
Path syntax is absolute and excludes dot components; path validation is not
symlink protection or authorization to load arbitrary plugins.

Four mandatory digests bind the admitted HF configuration, compressed-token
map, rank weight manifest and plugin lock. The worker must hash those actual
inputs against these digests before parsing/loading them. Nonzero digest bytes
do not prove provenance; the supervisor supplies identities from trusted
admission, never from adjacent untrusted files. The weight manifest must itself
bind rank/config and list exact shard names, extents and digests. The loader
below enforces this; bootstrap-to-resource orchestration remains integration work.

The binary header is magic `PIHBOOT1`, version and extent. Topology/scalars start
at 16; 64-bit sequence/budget/deadline values at 56; sampling scalar fields at
128; four SHA-256 digests at 160; NCCL ID at 288; four 100-byte endpoint slots
(U32 length plus 96 bytes) at 416; 512-byte NUL-padded artifact and plugin-lock
paths at 816 and 1328. All remaining bytes are zero. Decoding validates fields
then re-encodes for an exact canonical comparison, rejecting nonzero unused
slots/padding, embedded NULs, oversized extents and unsupported versions. There
is no compatibility decoder for older or alternate startup formats.

`SealedWorkerBootstrap::Create` writes the canonical bytes into a CLOEXEC memfd
and applies WRITE/GROW/SHRINK/SEAL. `Read` duplicates the descriptor, verifies
all seals and exact regular-file size, then reads/parses the bounded envelope.
`WorkerSpawn` performs this read and checks its own PID/effective UID before
creating a child; the child receives the same sealed file as FD 3. Short or
interrupted reads/writes fail closed rather than admitting partial metadata.
The NCCL ID is bootstrap material: do not log or expose the envelope. Callers
must close inherited/temporary descriptors and erase sensitive copies after
bootstrap use. No memfd/process/model runtime tests were executed here.

### Authenticated rank weight manifest

`BackboneWeightFiles::OpenManifest(directory, digest, config, world, rank,
device_budget)` reads the fixed member `weights.manifest.json`. The digest must
come from the sealed bootstrap's `weight_manifest_sha256`, not a sibling digest
file. Directory traversal rejects symlinks; the manifest must be a read-only,
single-hard-link regular file no larger than 128 KiB. File identity/timestamps
are checked across the read. Raw bytes are hashed before any JSON parsing.

The version-1 root has exactly five keys: `schema` equals
`pih.deepseek-v41.weights.v1`, `config_sha256` binds the admitted HF config,
`world_size` and `rank` bind partition topology, and `shards` lists 1–256 objects.
Each shard object has exactly `name`, `bytes` and `sha256`. Names are flat
`.safetensors` members under the existing shard-name policy, unique across the
list; bytes is an integer at least eight, and digests are mandatory SHA-256
identities. Unknown/missing keys, invalid types, duplicates and mismatched
topology/configuration fail admission. Parser depth, nodes and string lengths
are bounded independently of the file-size limit.

After manifest admission, the existing `Open` pipeline verifies every full
shard digest, immutable file identity and exact canonical tensor inventory.
The manifest is not accepted as a replacement for payload authentication.
It also does not perform source-checkpoint conversion or verify tokenizer
normalization. The loader is included in the CPU weight-file target and syntax
checker; no checkpoint read/upload or model execution was run here.

### Worker artifact admission assembly

`WorkerArtifacts::Open(bootstrap)` validates the envelope, then reads
`hf_config.json` and `compressed-token-map.bin` from its artifact directory.
The bounded reader walks parent directories without symlinks and requires a
read-only, single-link regular file. It compares identity/timestamps across the
read and checks the independently admitted SHA-256 before returning a byte
snapshot. Configuration is limited to 1 MiB; the map must satisfy the existing
exact 129280-entry format and dense compressed-ID validation.

The assembly constructs the frozen `FlashConfig`, a fresh `EngramHashState`,
an authenticated plugin-lock snapshot (at most 1 MiB), and the complete
`BackboneWeightFiles` owner through rank-manifest admission. It exposes those
owned objects for memory planning and `WorkerRuntime`; keep the assembly alive
while their references are in use. It does not allocate GPU memory.

Use `pih::worker::ParseDevelopmentLock(snapshot, original_absolute_path)` on
`plugin_lock_json` after copying it to the parser's string input. The path gives
the deployment-root context; this entry point does not reopen the lock file.
Existing lock schema, artifact path and binary verification still apply.
Authenticated bytes alone do not activate plugins or prove their capability
tables; the activation stack below performs that next stage.

Artifact admission checks the sealed absolute CLOCK_MONOTONIC startup deadline
between stages and after full shard authentication. It never returns ready
after expiration, but a blocking filesystem read or a large hash operation is
not preempted; the supervisor must retain an independent process deadline.
Kernel namespace/time-namespace setup must match the bootstrap assumptions.
No configuration/map/weight artifact execution test or GPU upload was run for
this change; syntax/documentation checks are not deployment qualification.

### Backend-only worker plugin activation

`pih::worker::WorkerPluginStack` accepts the authenticated lock snapshot, original
absolute lock path, independently admitted lock digest and nonzero activation
epoch. It rechecks the snapshot hash, parses without reopening the lock, requires
every plugin/kernel-pack binary digest and rejects locks containing an engine
factory request. Rank workers activate backend providers; they must not recursively
start another engine while constructing their own runtime.

The stack uses the existing microkernel plugin/kernel-pack loaders and registration
barrier. Loaded identity/version/architecture and the complete registered
capability graph must match the lock. Registration closes before use. `Resolve`
selects exactly one locked capability for the requested contract and verifies its
scope/cardinality through the existing binding authority. It neither chooses an
arbitrary duplicate nor substitutes a fallback provider. Consumers must still
validate each returned public API table's ABI/size and required callbacks.

Call `Shutdown` only after the runtime has released all CUDA/NCCL/memory users.
It closes capability resolution, performs checked plugin shutdown and revokes
bindings. Failed startup/shutdown is terminal, with no in-process retry.
Destruction of a non-shutdown stack intentionally retains its implementation
metadata until process exit: plugins may still hold host callback pointers, so
freeing those contexts would be unsafe. This is quarantine, not proof of GPU
cleanup; the supervisor must retire the worker process. The executable entry
below connects this activation stack to the admitted worker runtime.

### Native rank worker executable

`pih-v41-rank-worker` is built only for `103-real` with the
`pih.kernels.deepseek-v41.sm103` plugin selected, alongside the native Linux
CUDA/NCCL rank runtime and worker/microkernel build. Build the target explicitly with
`cmake --build <configured-linux-build> --target pih-v41-rank-worker`. It requires
the same pinned CUDA/NCCL development dependencies as the native runtime;
this target has received syntax checks, not a successful NVCC/Linux link or
multi-GPU qualification in the current Windows development environment.

This is an internal supervisor-launched executable, not a standalone inference
CLI. It accepts no command-line arguments and reads sealed bootstrap FD 3.
The entry checks parent PID, effective UID, parent-death SIGKILL and
no-new-privileges, then closes its inherited bootstrap descriptor. Invoking it
without the prepared supervisor launch will fail; there is no path/argv/Python
fallback. The supervisor must give `WorkerSpawn` only argv[0] for this binary.

The entry connects the four channels, authenticates CPU artifacts, subtracts
weights and both expert workspaces from the aggregate device budget, and
constructs the bounded inference plan. It activates the backend-only plugin
lock and resolves the unique process-scoped CUDA runtime/memory/resources/async
contracts through the existing microkernel. Then `WorkerRuntime` performs CUDA
and NCCL startup, weight upload, request execution and coordinated retirement.
Polling yields for at most one millisecond between continuations. The in-memory
NCCL ID is erased after communicator startup consumes it.

Only after the runtime sends its release notice does the entry perform checked
plugin shutdown and return zero. Shutdown failure therefore yields nonzero exit
even if a release notice was already sent; the supervisor's normal reap check
rejects that outcome. Errors emit a fixed stage name without bootstrap material,
paths, prompts or provider-controlled strings and exit the process. Supervisor
epoch failure, peer cancellation and cgroup reconciliation remain mandatory.

The rank entry is now wired, but a user-facing supervisor command, deployment
manifest/profile registration, full source conversion and service integration
remain unfinished. This is not a claim of supported V4.1/B300 deployment.

### Supervised rank group launch

`WorkerGroup::Start` takes the admitted worker executable FD, rank-ordered
ready `WorkerCgroup` owners, 2/4/8 sealed-envelope values and an explicit environment.
It validates all ranks before spawning: parent identity, rank order, unique
device ordinals/cgroup inodes and globally distinct endpoint names. Canonical
bootstrap comparison requires the same model/map, sampling identity and
parameters, deadlines, context limit, expected SM, NCCL ID and backend lock.
Only rank/device, payload budgets, artifact directory, shard-manifest digest
and endpoint names may vary. Physical device placement remains the deployment
profile's responsibility; ordinal uniqueness is not a topology proof.

All four-stream listeners and immutable bootstrap files are prepared before
the first child is launched. Each successful clone is retained immediately in
its `WorkerSpawn` slot, and its PID is then bound to the corresponding listeners.
`Poll` observes exec status and authenticates each channel bundle, then creates
the pidfd-bound process watch. Startup and handoff both enforce the common
deadline; handoff rechecks process liveness. Group readiness means transport
admission, not model readiness—the generation session still collects all ready
notices before dispatching tokens.

`StartGeneration` constructs `GenerationLoop` and `GenerationSession` directly
from the admitted channels, process watch and cgroup owners. It transfers custody
only after both constructions succeed. Keep the group alive until the session
and all borrowed references are finished. After transfer, the session is the sole
process retirement authority; startup kill/reap is rejected. If construction
fails, the group retains custody and its startup retirement path remains usable.

Before handoff, failures or cancellation use `BeginStartupRetirement(deadline)`
and `PollStartupRetirement`. Every spawned child is explicitly pidfd-killed and
nonblockingly reaped; failure for one rank does not prevent attempting the other
ranks. Spawn/reap masks remain available. Group destruction closes descriptors
only, never performs hidden process termination or blocking waits. This group
does not create cgroups, generate trusted bootstrap identities or supply the
user-facing supervisor/service command. The dedicated single-threaded launch
precondition from `WorkerSpawn` still applies.

### Worker cgroup-v2 preparation

`WorkerCgroup::Create` creates one fresh direct child under an explicitly
delegated cgroup-v2 directory FD. The parent must have the required memory,
pids and CPU controllers enabled and must be exclusively managed during worker
startup/teardown. This helper does not modify ancestor controller settings or
adopt an existing group. The leaf is a bounded lowercase alphanumeric/hyphen
name; callers must choose a fresh generation-specific name.

Creation verifies cgroup-v2 filesystem identity and an initially empty domain,
then sets and reads back `memory.max`, `memory.swap.max=0`,
`memory.oom.group=1`, `pids.max` and `cpu.max`. Memory is page-aligned; PID and
CPU quota/period inputs are bounded. These are host resource limits, not GPU
VRAM limits. Budget the supervisor-admitted resident/pinned memory, file-cache
charges and backend overhead separately from the runtime's device payload sum.
The worker group now requires these ready owners and verifies empty groups
before cloning, rather than accepting arbitrary unconfigured directory FDs.

After launch, keep each owner alive and do not externally rename, populate or
change the limits of its delegated subtree. `Kill` writes `cgroup.kill=1` to
terminate the complete subtree. `Empty` parses bounded `cgroup.events` and
requires an unambiguous populated field; populated zero is distinct from zombie
reaping and both proofs are needed at shutdown. `Remove` verifies the named
member still matches the owned directory, requires empty state and performs
nonrecursive rmdir. Nested groups are never recursively deleted.

Partial configuration failures retain ownership; inspect `created`/`removed`
and explicitly reconcile. A failed open after mkdir may require supervisor
reconciliation rather than guessing directory ownership. Destructors only close
FDs. The provided methods do not automatically kill or remove any host cgroup
during development. The orchestration below joins cgroup and process-reap paths;
no cgroup or process execution tests were run here.

### Combined process and cgroup retirement

The worker group retains borrowed pointers to its admitted cgroup owners and
passes those exact owners to `GenerationSession::Create` inside
`StartGeneration`; keep the owners alive and exclusively controlled
through session completion. Do not substitute unrelated cgroups. Session
admission rejects absent/repeated/not-ready owners before it takes responsibility.

Before handoff, `PollStartupRetirement` kills each owned cgroup subtree as well
as explicitly killing/reaping launched rank leaders. Once leaders are reaped,
it waits for populated zero and removes each empty direct child. Unlaunched
rank groups are cleaned too. Failures on one group do not prevent attempting
the other groups. The group reports retired only after all these steps finish.

After handoff, normal session reaping advances to `kCleaningCgroups`. Each
owned group must be empty and successfully removed before `kComplete`. Remaining
descendants turn normal completion into failure rather than being ignored.
The fault path closes the generation, performs the existing terminate grace,
then writes cgroup.kill alongside pidfd escalation. After rank reaping/output
credit reconciliation, it also waits for empty groups and removes them before
`kFailedRetired`. Cleanup shares the bounded retirement deadline.

`removed_cgroups_mask` and `reaped_mask` expose distinct progress. Already
removed groups are skipped during subsequent cleanup or fault transition.
Failed observations, nested groups that prevent nonrecursive removal, expired
deadlines or denied kill/removal operations remain `kFailedUnreconciled`.
Process leader reaping and populated zero do not prove GPU numerical correctness
or reap arbitrary adopted descendant zombies; supervisor subreaper policy and
full service integration still require completion.

### Group-to-generation admission

After `WorkerGroup::Poll` reports transport admission, call `StartGeneration`
with the fresh token ledger, exact prompt token span, stable vocabulary byte
table, first plan sequence and terminate grace. The group derives startup,
whole-sequence and retirement deadlines from the sealed worker bootstraps. The
grace must be nonnegative and shorter than that retirement timeout.

Ledger epoch/sequence generation/sampling config, prompt length and initial
sampling parameters must match the launched workers. Temperature/top-p compare
their exact floating-point bit patterns; seed, top-k, logprobs settings and
initial RNG/suppression state also match. Existing generation admission validates
token IDs and vocabulary extents before any dispatch. The group rechecks process
liveness before building the loop and session, and transfers retirement custody
only on success. Invalid/mismatched requests cannot silently start a different
sequence on workers already admitted for another configuration.

There is no raw `HandOff` compatibility API. Construction failures preserve the
group's startup cleanup responsibility; after successful transfer, consume and
release published output through the original queue and keep polling the session
through all-rank release, process reap and cgroup removal. Session construction
and these checks do not authenticate tokenizer vocabulary semantics; tokenization
and source-artifact admission remain required in the supervisor entry.

### Raw token bytes for the supervisor

The shared native byte-level tokenizer now exposes `RawTokenBytes(id)` and
`RawVocabulary(payload_budget)`. Use the raw vocabulary for the generation
ledger's token-byte table, not `Decode` called once per token: byte-level tokens
may split a UTF-8 code point, and per-token replacement decoding would corrupt
cross-token stop matching. Text-facing decode still applies its existing
incremental/final UTF-8 policy after raw bytes are assembled.

The vocabulary export owns ID-ordered strings, rejects absent mappings, limits
each token to 8192 bytes and enforces the aggregate payload budget. Container
metadata/allocation overhead is not part of that payload count. Keep all strings
and any string-view array stable while a generation session borrows them.
Existing family-specific special-token handling is preserved.

The independent `DeepSeekV41Flash` family now requires the exact official
[frozen tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/517ef625df97ec57aadc91b67506a57c20fdc5bb/tokenizer.json)
with SHA-256 `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b`.
The downloaded snapshot has 128000 base vocabulary entries and 1283 added
tokens; its byte-level BPE and pre-tokenization structure use the shared parser,
but selecting V4.1 does not accept an arbitrary older tokenizer file.

`V41Tokenizer::Open(artifact_directory, raw_vocabulary_budget)` authenticates
the read-only, non-symlink `tokenizer.json` before parsing, owns the vocabulary
strings and stable views, and requires the model's 129280-entry dimension.
Keep this owner alive while a generation session borrows `token_bytes()`.
`EncodeRendered` accepts an already rendered prompt and requires 1–4096 tokens;
it does not add BOS or implement chat/tool/DSML rendering. The CPU static target
is `pih_deepseek_v41_tokenizer` when native weight-file support is available.
Tokenizer execution/equivalence tests remain deferred. This adapter does not
authenticate the compressed Engram map or complete the supervisor/service entry.

For offline encoding/decoding only, build `pih-tokenize` and run
`pih-tokenize deepseek-v41-flash /absolute/tokenizer.json /absolute/request.json`.
The request is either `{"text":"already rendered prompt"}` or `{"tokens":[0,1,2]}`;
the output is the opposite representation. This command checks the pinned
tokenizer byte digest but does not apply supervisor filesystem admission,
the 4096-token prefill limit, chat rendering or model execution. Use the
`V41Tokenizer` adapter, not this offline command, inside the supervisor.

### CPU supervisor request admission

`SupervisorRequest::Create` owns the authenticated tokenizer, rendered prompt
IDs, output queue and token ledger in dependency-safe destruction order. Supply
separate raw-vocabulary, publication and record budgets plus an output slot
count; these payload budgets do not include allocator/container overhead or the
temporary tokenizer parser. The full prompt plus maximum completion reservation
must fit the configured worker context, which cannot exceed 1048576 positions.
Prompt encoding remains limited to 4096 tokens. Sampling/stopping validation is
shared with the existing ledger, with no Python fallback or chat-template guess.

For each admitted rank placement, call `BindBootstrap` to copy placement,
artifact hashes, endpoints and deadlines while setting the request's sequence
identity, sampling, prompt length and maximum positions. The returned bootstrap
passes its full structural validation; this does not authenticate the supplied
placement, executable, NCCL ID or artifact hashes. Pass these envelopes to
`WorkerGroup::Start`, poll readiness, then use the owner's `ledger()`, `prompt()`
and `token_bytes()` with `StartGeneration`. The group independently checks the
completion reservation against the launched context before custody transfer.
Read/release output leases through `output()` and keep the request owner alive
through session retirement and all outstanding leases. Destruction does not
cancel or reap a running session.

`pih_deepseek_v41_sampling_host`, token stopping/output/ledger and Linux rank
channels now build outside the CUDA block. Their dependency graph contains no
CUDA/NCCL library; `sampling.cu` alone belongs to the GPU sampling target, which
links the shared host validator. With `PIH_BUILD_TOKENIZER_TOOLS=ON`, Linux also
exposes `pih_deepseek_v41_supervisor_request`. This is request assembly, not yet
a runnable supervisor/service command. The current Windows cross-build lacks
Linux OpenSSL libraries, so complete CMake configuration/link validation remains
pending; source syntax checks are not a substitute for that validation.

### Sealed worker executable admission

Before starting a group, call `WorkerExecutable::Open(absolute_path,
trusted_deployment_sha256, byte_budget, deadline)`. Every parent component and
the final member is opened without following symbolic links. The source must
be a read-only regular file with a single hard link and an executable bit, with
no setuid/setgid bits. The explicit budget is 64 bytes to 512 MiB. Admission
checks the x86-64, little-endian ELF executable/shared-object header, copies the
source in 1 MiB chunks, verifies source identity and the full expected digest,
then seals the executable memfd against writes, growth and shrinkage. The
expected digest is trusted deployment input, never derived from the same file
as a substitute for authentication. Structural ELF checking does not establish
that an arbitrary trusted ELF implements the rank-worker protocol.

Both `WorkerGroup::Start` and `WorkerSpawn::Start` now require this typed owner;
the raw executable-FD overload is removed. Keep the owner alive during group
startup. Launch duplicates the sealed descriptor and uses `execveat`; it never
reopens the source file. A changed source path after admission cannot change
the admitted bytes. Source mutation observed during copying is rejected.
Destruction closes descriptors only and never affects worker process custody.

The host must allow executable memfds (`MFD_EXEC`); denied/unsupported memfds
fail admission with no disk/path fallback. The snapshot consumes up to the
admitted binary size in kernel-backed storage plus the bounded copy buffer.
Deadline checks occur between bounded I/O operations, not by forcibly
interrupting a blocked filesystem syscall. ELF interpreter and shared libraries
are **not** included in the snapshot. In particular, `$ORIGIN`-relative library
resolution must not assume the original on-disk executable directory after
memfd execution. A trusted deployment must supply a compatible admitted loader
and library search environment; that closure is still pending. No memfd exec,
dynamic linking, worker launch or model execution test has been run here.

### Explicit worker environment

`WorkerEnvironment::Create(library_directories)` constructs the only environment
accepted by group/child launch: `LANG=C`, `LC_ALL=C`,
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `NCCL_CONF_FILE=/dev/null`, and, if supplied,
one `LD_LIBRARY_PATH`. It never copies the supervisor's environment. There is
no generic key/value override, inherited `LD_PRELOAD`/`LD_AUDIT`, or
`CUDA_VISIBLE_DEVICES` remapping. Bootstrap device ordinals must be assigned
in ascending PCI bus order among devices exposed to the worker's container,
matching [NVIDIA's CUDA ordering contract](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).

At most 16 unique library directories are allowed, each at most 1024 bytes and
the combined search string at most 4096 bytes. Each must be absolute with no
dot components, repeated/trailing separators, symbolic-link components, loader
list separators (`:`/`;`) or `$` substitution. Every opened component must be
owned by root or the effective supervisor user and not writable by group/other.
Use canonical installed locations rather than symlink aliases. An empty list
omits `LD_LIBRARY_PATH`; it does not disable the loader's system default search.

These are path checks at construction time, not file hashing, a mount namespace,
or proof that a directory cannot subsequently change. The trusted deployment
must retain control of these directories and authenticate their shared objects,
ELF interpreter and default loader configuration before launch. NCCL's explicit
config path avoids its default per-user config file, but does not suppress
`/etc/nccl.conf`; system-wide configuration remains part of host admission.
See [NCCL 2.31.2 configuration rules](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
System loader/NCCL configuration, topology files and dynamic-library closure
are still pending deployment work. Do not describe this environment owner as
complete runtime dependency authentication or B300 qualification.

Immediately before `clone3`, the launcher now verifies that `/proc/self/task`
contains exactly the calling process leader and rechecks the startup deadline.
Absent procfs or any additional thread rejects launch before a child is created.
The supervisor must remain dedicated/single-threaded, including signal handlers;
this check is not a general lock-safety solution for multithreaded applications.
In particular, obtaining an NCCL bootstrap ID in a library that starts background
threads cannot be followed by this launcher in the same process. Isolated
bootstrap-ID production/distribution remains to be connected to the service.

### Isolated NCCL bootstrap helper

The pinned [NCCL bootstrap implementation](https://github.com/NVIDIA/nccl/blob/v2.31.2-1/src/bootstrap.cc)
creates a detached listener thread when making a normal unique ID. Therefore
the ID producer must remain alive during communicator initialization, and it
must not run inside the single-threaded process that launches rank children.
The old `WorkerCommunicator::NewBootstrapId` entry has been removed.

`pih-v41-nccl-bootstrap` is an internal helper executable, not a user-facing
command. It requires inherited FD 3 to be a private, connected, nonblocking
UNIX packet socket; checks parent credentials, parent-death SIGKILL and
no-new-privileges; and verifies NCCL runtime version 2.31.2. Its sole argument
is an absolute monotonic lifetime deadline. It generates exactly one ID,
sends one versioned 144-byte packet (16-byte header plus 128-byte opaque ID),
clears its local copies, and remains alive until parent retirement or failure.
It never prints the ID. The lifetime deadline cannot interrupt a blocking NCCL
call; the independent supervisor must enforce its process timeout.

The CPU supervisor uses `NcclBootstrapBroker` as follows:

1. Admit the helper ELF with `WorkerExecutable`, construct `WorkerEnvironment`,
   and create an empty, dedicated `WorkerCgroup` separate from rank cgroups.
2. Call `Start` with startup and later lifetime deadlines. Startup allocates a
   private socketpair and launches through the same single-threaded
   `clone3`/sealed-ELF/explicit-environment path. It does not fabricate a rank
   startup envelope containing a dummy NCCL ID.
3. Poll until ID admission succeeds, then use `BorrowId` to copy the ID into
   every rank's sealed bootstrap. Continue polling broker liveness throughout
   rank initialization. Packet shape, nonzero ID, duplicate packets, process
   death and deadlines are checked; ID creation alone does not prove rank
   readiness or authorize generation.
4. Keep the helper alive until every rank has completed communicator
   initialization. Once the all-rank ready boundary is established, or after
   cancelling startup, call `BeginRetirement` and poll retirement through
   cgroup kill, exact helper pidfd reaping, empty-state observation and removal
   of its owned cgroup. Explicit SIGKILL is the helper's retirement mechanism;
   it has no application shutdown protocol or fake successful exit receipt.

Retire even a partially started broker after an error. Destruction only closes
descriptors and clears its stored ID: it does not kill or reap a process. Keep
the cgroup owner alive, preserve the broker on cleanup failure, and erase
caller-owned ID/bootstrap copies separately. Its protocol/parser and launch
code have been compiled, not executed; end-to-end service coordination,
real NCCL initialization and failure/reaping tests remain pending.

### Broker custody follows the generation session

`WorkerGroup::Start` now requires the live `NcclBootstrapBroker` in addition to
the worker executable, cgroups, bootstraps and environment. It compares the
rank bootstrap ID to `BorrowId()`, requires broker lifetime beyond the rank
startup deadline, and rejects using the broker's cgroup as a rank cgroup. Keep
the broker and its cgroup owner alive through group/session retirement. If
validation fails before group startup takes custody, the caller still retires
the broker; after startup enters a failed/connecting state, the group's explicit
startup retirement includes it. Pre-handoff channel polling checks broker
liveness; generation-session creation checks it again.

The group alone constructs a `GenerationSession`; its constructor factory is
no longer a public bypass around group binding. Successful handoff transfers
retirement responsibility for ranks and helper together. While waiting for all
rank ready notices, the session monitors broker liveness. Once every authenticated
ready notice has arrived, it enters `kReapingBootstrap`: it explicitly kills,
reaps and removes the helper's dedicated cgroup before entering `kRunning`.
No inference request is submitted during this bootstrap retirement stage.
The worker process liveness and overall generation deadline remain enforced.

Cancellation or startup failure begins broker retirement alongside rank fault
retirement. A helper cleanup error is recorded without skipping rank kill/reap;
final successful or failed-but-retired session states require the broker to be
retired too. Unresolved cleanup remains an error with retained owners, not a
successful terminal result. The service must keep polling; no destructor or
background thread performs cleanup. These connections have source compilation
coverage only; actual process lifecycle and model tests remain deferred.

### Unified supervisor operation

`SupervisorOperation` provides one CPU-side `Start`/`Poll`/`Cancel` entry across
request, broker, rank-group and session lifetimes. It borrows admitted worker
and helper executables, a `WorkerEnvironment`, `SupervisorRequest`, 2/4/8 empty
rank cgroups and a separate empty helper cgroup. Keep these owners alive until
retirement is proven; output leases may require retaining the request longer.
The operation owns the broker, rank group and session in dependency-safe order.
It does not create/configure cgroups or authenticate artifact hashes supplied by
the caller, and it never initializes CUDA in the supervisor.

Supply ordered rank placement templates with empty NCCL IDs, common absolute
startup/sequence deadlines and retirement timeout, rank-local artifacts/devices,
and unique endpoints. Also supply the initial plan sequence and termination
grace. After the isolated helper returns an ID, the operation binds the request's
sequence fields into every template, seals/launches the rank group, polls channel
admission and transfers custody into generation. Full bootstrap/group admission
still occurs; the templates are not a way to bypass it.

Poll `kStartingBroker` → `kConnectingRanks` → `kSession`. On `kOutputReady`,
`TakeOutput` transfers the existing lease; read and release it through the
original request's output queue after consumption. Polling without consuming
output retains backpressure and deadline/liveness checks. Already committed
session output remains retrievable during failure cleanup. `kComplete` means
normal generation and process/cgroup retirement finished, not that the caller
has released every transferred publication lease.

`Cancel` takes a non-success reason and drives the same retirement path as a
startup/generation error. If `Start` reports an error after entering a nonempty
state, keep polling for cleanup. An error before resource custody leaves the
operation empty and the caller responsible for its pre-created empty cgroups.
Before rank-group custody, cleanup reaps the helper and removes untouched empty
rank cgroups without killing unexpected occupants. After group custody, its
partial-startup retirement is used; after session handoff, session retirement
is used. `kFailedRetired` proves cleanup, while `kFailedUnreconciled` retains
unresolved resource obligations. Inspect `failure()` and `cleanup_failure()`
separately. Destruction performs no implicit process kill/reap.

This entry is linked into `pih_deepseek_v41_supervisor_request` on Linux with
tokenizer tools enabled. It is not yet a CLI/HTTP service or an assertion of
end-to-end inference correctness. Compilation/static checks are the current
evidence; runtime process/model tests and deployment-command wiring remain pending.

### Authenticated supervisor configuration

`SupervisorConfig` now provides bounded, digest-authenticated parsing/loading of
the native deployment/request schema. It emits executable identities, library
paths, cgroup limits, request/sampling/stop budgets, relative timeout durations
and ordered rank placement templates without starting processes. See
[the exact supervisor schema](SUPERVISOR_CONFIG.md) for required fields,
limits and the remaining runtime assembly responsibilities.

### Native single-request supervisor command

`pih-v41-supervisor` now connects the authenticated configuration to a complete
single-request orchestration loop. This is a development command, not an HTTP
server, release qualification, or an automatic checkpoint converter. It has
been source-compiled on a Linux target but has not been fully linked or run
against model weights/GPU hardware in this environment.

On an appropriately configured Linux build with tokenizer tools and native CUDA
worker support, build the three targets explicitly:

```sh
cmake --build /absolute/build --target pih-v41-supervisor pih-v41-rank-worker pih-v41-nccl-bootstrap
/absolute/build/pih-v41-supervisor /absolute/supervisor.json --sha256 TRUSTED_CONFIG_SHA256
```

The JSON schema is documented in [SUPERVISOR_CONFIG.md](SUPERVISOR_CONFIG.md).
Use a digest from your trusted deployment record; the command does not accept
an unsigned/implicit configuration fallback. Prepare pinned read-only tokenizer,
model artifacts, backend plugin lock and executable files at the configured
locations. The helper and worker ELF hashes/budgets are separate from the
configuration hash. Existing source checkpoints are not directly accepted as
converted rank artifacts. The host must already provide the required cgroup-v2
delegation/controllers, clone3/pidfd/execveat/executable-memfd support, compatible
NCCL/CUDA libraries and admitted GPUs. Library closure/host qualification remain
the deployer's responsibility; the CLI does not grant those prerequisites.

The command authenticates configuration, prepares tokenizer/request and executable
snapshots, obtains a 128-bit endpoint namespace from kernel randomness, computes
bounded absolute monotonic deadlines, and creates fresh direct cgroup children
under the configured delegated parent. Names are logged as `pih-v41-<nonce>`
on stderr for troubleshooting. It does not alter ancestor controllers or adopt
existing groups. Child startup follows the broker/group/session ownership path
described above. The supervisor becomes a dedicated child subreaper and retains
all owners through final cleanup, including adopted descendant exit statuses.

Stdout carries committed raw token bytes, without an added newline, JSON wrapper,
chat/tool parsing, or per-token UTF-8 replacement. Consumers needing structured
responses must use an appropriate serializer, not assume this is an OpenAI API.
Stdout is temporarily nonblocking and restored on exit. A stalled/closed output
sink remains subject to the sequence deadline; signals or write failures cancel
generation while retirement continues. Blocking filesystem syscalls are not
forcibly preempted. Keep stderr separate from stdout: all supervisor and child
diagnostics go to stderr. Pending output may be discarded after cancellation or
output failure, without rolling back already accepted model tokens.

Exit 0 means the operation completed and owned-process/cgroup cleanup was
observed; exit 2 indicates admission, generation, cancellation or output failure;
exit 3 indicates unresolved cleanup. On exit 3, inspect the logged namespace and
do not treat the request as successfully retired. Final cleanup only addresses
the fresh groups owned by this invocation. No command or model execution test
has been performed here, and this command does not complete the V4.1/B300 release
requirements or the old-runtime removal work.
