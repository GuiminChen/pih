# Native plugin inference: build, serve and observe

English | [中文](native-inference.zh.md)

Current code/build status is in [integration status](integration-status.md). Production links and device images are verified; hardware execution is unverified. Build/release and system prerequisites are in [deployment](deployment.md). Historical qualification commands below are optional operator actions, not delivery acceptance tests.

Qwen `serve`/`seal` and DeepSeek `serve-deepseek` accept `--generation-timeout-ms` (default `600000`, range
`1..86400000`). This required model configuration field is recorded in the new
Lock; regenerate old text-service Locks rather than using a compatibility default. The
deadline starts before request parsing/tokenization and is checked before opening
the output stream and during native generation. It does not preempt a running
CPU function or CUDA call. Cooperative retirement has its own bounded interval;
the HTTP receive/write timeouts are separate. No timed model execution has been
performed to validate this path.

Model-only Qwen builds no longer create offline converter targets to inspect
their source lists. The model excludes the `plugins/offline-qwen` source owner
directly. Select `PIH_BUILD_QWEN_ARTIFACT_TOOLS=ON` when conversion commands are
needed; the documented bundle builder already does this explicitly. This build
boundary change is not CUDA/link or model-execution qualification.

Deployment configure commands explicitly reset optional tooling, native contract
tests, historical tests and transports before applying the selected profile.
Qwen bundles explicitly enable their CPU artifact tools because those executable
targets are part of the bundle build. V4.1 shell builds also disable both test
families. This controls CMake cache selections, not old files already installed:
continue to use separate build/install directories per profile and architecture.

This path loads native C ABI plugins only. Python orchestrates the build, seals
a deployment Lock, then execs `pih-worker`. It does not import `_pih`, execute a
Python model process, link the monolith or silently fall back to a reference.
The first-release code and Linux/CUDA build delivery are complete; this is not
evidence of Linux execution, hardware inference or numerical correctness.

## Environment matrix

KV scrub/recycling also uses the bound async capability for zeroing and event
completion, with address-range checks and bounded waiting. A pending event is
not successful retirement. The production CUDA targets and native contract
targets have been compiled and finally linked; the resulting Linux executables
and the new regression assertions have not been run.

Per-step error clearing is now a separate capability-bound operation, validating
the exact four-byte target and rank rather than sharing the concrete matrix
driver. Matrix execution is unchanged and still has direct backend dependencies.

| Environment | Current native path |
| --- | --- |
| Linux, exact RTX 4090 D, Qwen3-0.6B | New model + SM89 pack + HTTP; BF16/PIH INT4; unqualified |
| Linux, exact H100 PCIe, Qwen3-0.6B | Same model + SM90 pack; single GPU; unqualified |
| Linux, RTX 4090 D, DeepSeek V4-0731 | Native token/text/chat and SSE wired; verified weights plus pinned semantic snapshot required; unqualified |
| Linux, B300, DeepSeek V4.1 Flash | Native model/SM103 pack, rank worker, supervisor and HTTP source path; Linux/CUDA production links verified; hardware execution unverified; unqualified |
| Windows/macOS/CPU-only | Inspect commands and send HTTP; no local GPU inference through this path |

### DeepSeek V4.1 / B300 development bundle

On a Linux B300 host with a CUDA toolkit that can compile `sm_103`, ICU,
OpenSSL, NCCL and CMake/Ninja, create a new bundle and seal its rank Lock:

```bash
bash deploy/run-v41-native.sh build-service /absolute/build-dir /absolute/new-bundle 4
bash deploy/run-v41-native.sh seal-rank /absolute/new-bundle
```

Prepare an authenticated `pih.deepseek-v41.supervisor.v1` configuration using
the exact paths and SHA-256 values of that bundle's worker, helper and
`v41-rank.lock`, the B300 rank placements, delegated cgroup parent and external
verified model generation. The complete field and admission rules are in
[`SUPERVISOR_CONFIG.md`](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md).
Obtain the configuration digest from an independently trusted deployment record;
do not substitute a digest merely computed from unreviewed input. Then:

```bash
bash deploy/run-v41-native.sh seal-service /absolute/new-bundle /absolute/config.json TRUSTED_SHA256
bash deploy/run-v41-native.sh serve /absolute/new-bundle 8000
# From a second shell, if explicitly testing on that host:
bash deploy/run-v41-native.sh probe /absolute/new-observation.json 8000 32
```

`build-service` compiles and installs; `seal-rank` and `seal-service` write
new Locks but do not execute the model. `serve` and `probe` do execute it and
are not part of the current static-only development gate. No checkpoint is
downloaded or converted automatically, and the Python probe only sends HTTP.
The rank Lock requires `pih.transport.nccl` alongside the CUDA backend; its
provider owns NCCL communicators and collective submission. Rebuild and reseal
older V4.1 bundles, since no raw NCCL-handle compatibility path remains in the
rank worker.
For a code-only check, `python3 tools/check_native_syntax.py --v41-model`
requires the pinned local syntax-header/sysroot preparations; it does not
compile `.cu`, link Linux ELF objects, or validate B300 inference.

## DeepSeek V4-0731 token generation

The new `inference.tokens.v1` capability (`pih.inference.tokens.v1`) runs chunked
prefill, sampling, iterative decode, output acknowledgement and request retirement
inside the native model plugin. The C ABI exposes temperature, top-p, seed,
minimum/maximum completion lengths, up to 16 stop tokens and a monotonic deadline.
The Worker CLI currently exposes greedy generation, seed 7, a 30-minute generation
deadline and a five-minute close deadline. The diagnostic smoke reuses this loop.

Input is **token IDs, not text or chat messages**. Use the matching V4 Flash-0731
tokenizer and template; V4.1/Qwen IDs are not interchangeable. Output contains
`token_ids`, `finish_reason` and prompt/completion counts, never fabricated text.

Prepare an external, previously published PP1 verified generation containing
`pih.manifest.json`, `pih.runtime-records.json` and its weights. Use its trusted
published artifact root, not a hash of the directory name. A raw HF weight folder
is not a verified generation. On Linux with an exact RTX 4090 D:

```bash
export ARTIFACT_ROOT='replace-with-verified-64-character-lowercase-root'
export PROMPT_TOKEN_IDS='replace-with-comma-separated-0731-token-IDs'
python3 deploy/native.py generate-deepseek-tokens --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" \
  --prompt-tokens "$PROMPT_TOKEN_IDS" --max-tokens 128 --jobs 4
```

This builds/seals the native bundle, validates/stages the external generation,
then execs the Worker for the explicitly requested generation. No test suite or
Python model is run. Omit `--build` and both artifact arguments on subsequent runs.
The default build directory is `out/build/deepseek-4090d-pp1`; its
`bundle/deepseek-4090d-pp1` subdirectory is recreated during bundle builds. Never
store weights or evidence there. Input plus requested output must fit the Lock's
attention capacity (4096 in the template). Empty entries/spaces are not accepted.

Direct invocation uses `pih-worker --lock PATH --generate-tokens ID,ID,...
--max-tokens COUNT`. Results are published only after successful generation and
cleanup. Rebuild the whole bundle/Lock to include the newly registered capability;
old Locks are not repaired or accepted silently. This implementation has not yet
been compiled or executed on Linux/CUDA and makes no numerical correctness claim.

V4.1 has a separate frozen-HF-config parser, native model plugin, SM103 Kernel
Pack, rank worker and supervisor in `plugins/model-deepseek-v41` and
`plugins/kernels/deepseek-v41-sm103`. It rejects V4-0731 configuration and
registers `inference.text.v2` only in a sealed V4.1 bundle. The code path still
requires Linux/CUDA linking, runtime integration and model qualification;
source compilation alone is not proof that B300 inference runs correctly.

## DeepSeek V4-0731 native text service

The model plugin now exposes `inference.text.v2`, consuming authenticated
semantic bytes and the existing native PP1 engine. The deployment command loads
only native providers and `pih.surface.text-http`, not the diagnostic HTTP
surface or a Python model backend. On Linux RTX 4090 D with CUDA, ICU and
OpenSSL development libraries:

```bash
python3 deploy/native.py serve-deepseek --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" \
  --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731 \
  --max-context 4096 --port 8000 --jobs 4
```

`ARTIFACT_ROOT` must be the exact verified weight-generation root described
above, not a raw checkpoint hash. The semantic snapshot must meet the pinned
15-file admission rules below. Both directories must be outside source/build/
bundle trees. Default build/install directories are `out/build/deepseek-text`
and `out/install/deepseek-text`; use a fresh build directory, and stop any old
worker before rebuilding. The command stages the external weight-generation
link, hashes built plugin binaries and writes a content-addressed native Lock.
Omit `--build` only after building that bundle. It does not convert raw weights.

Alternatively, start from a committed and activated generation store:

```bash
python3 deploy/native.py serve-deepseek --build \
  --store-dir /srv/models/deepseek-store \
  --pointer-root "$POINTER_ROOT" --catalog-root "$CATALOG_ROOT" \
  --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731 \
  --max-context 4096 --port 8000 --jobs 4
```

This mode first builds/runs the CPU-native store resolver, verifies the pinned
pointer, receipt and every target byte, then seals the selected artifact root
into the text Worker Lock. `--build` also builds the service as above; the resolver
has a separate build tree, default `out/build/deepseek-artifact` (override with
`--artifact-tools-build-dir`). Without `--build`, both native tool and service
bundle must already exist. Store, semantic snapshot and build/bundle trees must
not overlap. Do not also pass `--artifact-dir` or `--artifact-root` in store mode.
Missing/stale roots or failed verification terminate startup with no fallback.
The original weight checkpoint is not needed for this read-only target check.

Selection is pinned at resolution time, not a subscription to future pointer
changes. Python only dispatches and validates the small native observation; the
Worker must still perform its own artifact/storage admission. The catalog root
is matched to the pointer, not authenticated against catalog bytes here. Later
activation does not hot-reload this process. This startup path is implemented
but has not been executed on Linux/CUDA locally.

After `state: serving`, a request can be sent explicitly:

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"你好"}],"thinking_mode":"chat","max_tokens":128,"temperature":0,"top_p":1,"stream":true}'
```

`stream:false` returns a complete JSON response. Chat accepts text histories
ending in user/developer/tool input, `thinking_mode` chat/thinking (default
chat), `reasoning_effort` low/high/max, `drop_thinking`, `tools` and prompt-level
`response_format`. A response schema is an instruction, **not constrained
decoding or guaranteed schema validation**. Tools are returned as parsed data
with IDs and are never executed. Tool-call chunks are emitted only after full
DSML validation; text/reasoning remains incremental. At length exhaustion,
unfinished tool blocks are withheld. `tool_choice`, images, arbitrary stop
strings, logprobs, penalties and other unknown fields are rejected.

Both chat and `/v1/completions` support `max_tokens`, temperature 0–2, top_p
(0,1], nonnegative integer seed, boolean stream, and n=1. Greedy temperature=0
requires top_p=1. Raw completions take a string `prompt` without chat framing;
EOS is omitted from raw returned text. Prompt plus output must fit the sealed
context (this development launcher caps it at 4096, not a qualified capacity).
Requests are serialized on loopback, without TLS/auth. The configured deadline defaults to 10 minutes;
interruption drains the current committed device plan for up to another five
minutes before retirement. This is cooperative, not CUDA preemption. Cleaned-up
timeouts return 504; malformed generated structure or failed cleanup terminates
the worker. Production linking has passed; hardware execution remains unverified.

## Prerequisites and artifacts

Install a C++20 compiler, CMake 3.26+, Ninja, OpenSSL 3 development files, ICU
development files, CUDA Toolkit 13.2 (the current CMake requirement), a matching
driver, and Python 3.11+ for orchestration. Scripts do not install system packages
or drivers or download weights. Set `CUDACXX` if nvcc is not discoverable.
The worker rejects loader injection variables such as `LD_LIBRARY_PATH` and
`LD_PRELOAD`; use correct system libraries/RPATH rather than injecting libraries.

Keep weights outside the source/build trees. A BF16 directory needs
`config.json`, `tokenizer.json`, `model.safetensors`. An INT4 directory needs the
first two plus PIH's `model.xing-int4`, not GGUF/AWQ/GPTQ. A new native offline
converter reuses the verified conversion implementation without `_pih`:

```bash
python3 deploy/native.py convert-int4 --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json \
  --output /srv/models/Qwen3-0.6B-int4/model.xing-int4
```

The existing expectation schema binds `schema`, `file_bytes`, `tensor_count`,
`data_bytes`, `file_sha256` to a trusted source. The output parent must exist;
the output and `.staging` must not exist. Copy config/tokenizer from the same
revision separately and retain the stdout receipt. A self-computed hash alone
does not establish official provenance.

To copy the same-revision config/tokenizer into the existing INT4 directory:

```bash
python3 deploy/native.py prepare-qwen-metadata \
  --source-dir /srv/models/Qwen3-0.6B \
  --model-dir /srv/models/Qwen3-0.6B-int4 \
  --config-sha256 "$TRUSTED_CONFIG_SHA256"
```

Obtain the config digest from an independently trusted same-revision record;
the tokenizer digest is pinned to the native service's tokenizer. Both bounded
snapshots are authenticated before either destination is created. Copies are
exclusive, read-only, readback-hashed and synced. This does not convert weights,
validate config semantics or qualify a model. The output directory must be owned
by the current user and not writable by group/others. Exclude concurrent writers
to it and its parent paths during preparation. Existing files, including symlinks,
are never overwritten. The pair and weights are not published atomically:
failure can leave one or both files. Do not serve partial results; inspect the
failure and use a fresh output directory instead of automatic deletion or blind
retry. This command has not been executed against model assets here.

Alternatively, if metadata has not yet been copied, prepare it and convert in
one invocation:

```bash
python3 deploy/native.py convert-int4 --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json \
  --output /srv/models/Qwen3-0.6B-int4/model.xing-int4 \
  --prepare-metadata --config-sha256 "$TRUSTED_CONFIG_SHA256"
```

Metadata is taken from the source weight's parent directory. The destination
filename must be `model.xing-int4`. The command builds/selects native tools,
authenticates and copies metadata, then invokes the native converter. Metadata
failure prevents conversion; conversion failure can still leave metadata and
weight staging files. This is not an atomic transaction and performs no automatic
cleanup. Do not copy metadata separately and then pass this option for the same
directory: existing files are rejected. Perform the independent INT4 verification
below before proceeding to service startup.

Before conversion, or for BF16 deployment without conversion, validate the source
against that same trusted expectation file independently:

```bash
python3 deploy/native.py verify-qwen-source --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json
```

This CPU command validates the official-layout manifest, whole-file digest,
tensor/data lengths and tied embedding/LM-head byte equality using the native
source verifier. It neither converts weights nor loads an engine. It emits a
source observation receipt, not a deployment Lock or numerical qualification.
The expectation parser is shared with conversion: exactly five schema fields,
positive bounded counts and a nonzero lowercase digest are required. Trust the
expectation independently and keep the source immutable throughout verification.
No source verification was executed during this implementation.

After conversion, verify the INT4 artifact:

```bash
python3 deploy/native.py verify-int4 --build \
  --artifact /srv/models/Qwen3-0.6B-int4/model.xing-int4 \
  --file-sha256 "$FILE_SHA256" --source-root "$SOURCE_ROOT" \
  --binding-root "$BINDING_ROOT" --disposition-root "$DISPOSITION_ROOT"
```

This entry reuses the CPU build path, rejects malformed expected digests before
building, and invokes only the native verifier. It does not modify the artifact,
generate its own expected digests, or start a service. Omit `--build` to use the
installed tool selected by `--bundle`. To invoke the built executable directly:

```bash
out/build/qwen3-artifact/plugins/offline-qwen/pih-qwen-int4-verify /srv/models/Qwen3-0.6B-int4/model.xing-int4 "$FILE_SHA256" "$SOURCE_ROOT" "$BINDING_ROOT" "$DISPOSITION_ROOT"
```

Set the four nonzero lowercase digests from your independently trusted expected
record (`file_sha256`, `source_artifact_root`, `source_binding_root`,
`disposition_root`). Checking against a saved converter receipt detects drift
relative to that receipt, not independent official provenance. The verifier
checks layout, payload hashes and the four expected bindings, emits a receipt,
and exits 0 on success or 2 on error. It does not assess quantization quality or
run a model. Keep the artifact immutable during verification; this command does
not create a trusted mount or protect against another writer truncating mappings.

The offline quantizer disables fast-math/contraction and requires round-to-nearest
mode; unsupported floating-point modes fail explicitly. Conversion receipt output
failure also returns nonzero even if the artifact was already published. A failure
may leave output or `.staging`; inspect and verify before retrying, never assume
failure means no file was published. Neither conversion nor verification has been
executed as part of this implementation.

Before allocating the Qwen GPU engine, the service now verifies the tokenizer
byte snapshot (maximum 12 MiB) against the repository-pinned SHA-256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4`.
It parses those same bytes, without reopening the path. A differently formatted
or different-revision tokenizer is rejected even if it looks structurally valid.
The CPU `pih-qwen-format --tokens` path uses the same admission helper; raw
`pih-tokenize qwen3` remains an unpinned format tool. This byte pin does not
authenticate weights/configuration, prove tokenizer equivalence, or establish a
trusted filesystem/dependency closure. Runtime qualification is still pending.

The new C++ tokenizer reads ByteLevel/BPE vocabulary, merges and regex from the
local tokenizer JSON. The restricted non-thinking template follows the
[official Qwen tokenizer configuration](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/tokenizer_config.json).
Numerical, template and Unicode equivalence still need independent validation.

## One-command build and launch

From the repository root:

For a build without launching inference, specify the target explicitly:

```bash
python3 deploy/native.py build --target rtx4090d --jobs 4
```

The deployment builder selects only SM89 for `rtx4090d` or SM90 for
`h100-pcie`, for both cubin generation and Kernel Pack installation. Direct CMake
users can set `PIH_QWEN_KERNEL_ARCHITECTURES=89`, `90`, or `89;90`; empty,
duplicate and unknown architectures are rejected. This selection does not grant
hardware qualification. If an installation already contains the other Qwen
architecture, select fresh `--bundle` and `--build-dir` paths; the script does
not delete existing files or silently mix packs. A build does not inspect or
execute a GPU; `serve` still performs its separate host check.

For later, separate CUDA kernel qualification on an RTX 4090 D, opt in after
the production build (this command is not installed in the service bundle):

```bash
cmake -S . -B out/build/qwen3-native -DPIH_BUILD_QWEN_QUALIFICATION_TOOLS=ON
cmake --build out/build/qwen3-native --target pih-qwen-cuda-qualify --parallel 4
export PIH_QWEN_QUALIFIER="$PWD/out/build/qwen3-native/plugins/model-qwen3/pih-qwen-cuda-qualify"
export PIH_QWEN_CUBIN_PATH="$PWD/out/install/qwen3-native/lib/qwen3-sm89/sm_89/qwen_bf16_primitives.cubin"
"$PIH_QWEN_QUALIFIER" shape
```

`shape` emits raw five-case device-output JSON; `gemm` and `metric` run the
other retained CUDA fixtures. The optional hardware collector in
`tests/hardware/qwen_m2_shape_corpus_smoke.py` uses both absolute paths and
adds GPU/artifact identity checks. No fixture has been run in this code-only
phase, and raw CLI success alone is not a release qualification.

```bash
unset CUDA_VISIBLE_DEVICES
python3 deploy/native.py serve --build --target rtx4090d --precision bf16 \
  --model-dir /srv/models/Qwen3-0.6B \
  --artifact-sha256 "$TRUSTED_BF16_ARTIFACT_SHA256" \
  --config-sha256 "$TRUSTED_CONFIG_SHA256" \
  --max-context 4096 --port 8000 --jobs 4
```

`seal` and `serve` require independent, nonzero SHA-256 values for the selected
`model.safetensors` or `model.xing-int4` and `config.json`. The launcher checks
both before sealing; the native loader rechecks the bytes it consumes. Computing
a hash from an untrusted local file does not itself establish official provenance.
Sealing also verifies `tokenizer.json` against the Qwen plugin's pinned digest,
so an incompatible tokenizer fails before a service lock is published. The
launcher rejects files above the loader's snapshot bounds (2 GiB weights,
1 MiB config, 16 MiB tokenizer) before hashing them.
Tokenizer, config and weights are read through the storage plugin's
authenticated-snapshot capability. It copies each source through a bounded
descriptor, checks the trusted SHA-256, and makes the copied pages read-only
before exposing them. Weight loading consumes that immutable snapshot, not a
reopened pathname. This is code-level admission, not hardware qualification.

Use `--target h100-pcie` for that exact GPU; use `--precision int4` and its
separate artifact directory for PIH INT4. The target always owns physical GPU 0;
the launcher now requires `nvidia-smi` index 0 to also be first in PCI-address
order, rejects incomplete/duplicate device rows, and sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID` before the native worker starts. NVIDIA documents
that CUDA otherwise defaults to heuristic fastest-first enumeration, which is
not a promise of matching management-tool indexes. See the
[CUDA environment-variable reference](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html).
This check does not establish MIG/MPS compatibility, device exclusivity, or
protection against topology changes between preflight and worker startup.
do not mask CUDA devices. Containers must expose the physical device namespace.

Defaults: `out/build/qwen3-native` and `out/install/qwen3-native`; override with
`--build-dir` and `--bundle`. Subsequent launches omit `--build`; rebuild after
source changes. The build disables monolith, Python extension, NCCL and tests.
Binary hashes are sealed into a new content-named Lock without overwriting one.
It is a development Lock, not a signed production admission document.

Wait for `"state":"serving"`. The server binds loopback only, serializes
requests, and provides no TLS or authentication: do not expose it publicly.
Ctrl+C/SIGTERM stop acceptance and clean up after in-flight generation returns;
GPU cancellation is not immediate. SIGKILL cannot perform application cleanup.

```bash
curl -f http://127.0.0.1:8000/readyz
curl -f http://127.0.0.1:8000/v1/models
curl -f http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Explain tensor parallelism in one sentence."}],"max_tokens":128,"temperature":0,"top_p":1,"stream":false}'
```

Discovery requests (`GET /health`, `GET /readyz`, `GET /v1/models`) require
a nonempty Host. They reject payloads, transfer encoding, duplicate headers and
already-received pipelined bytes. Content-Length may be absent or exactly `0`.
They use the same header validation as inference requests, not a GPU self-test.
The current serialized service cannot answer health checks independently while
generation is running; a 200 response does not qualify model output.

Framing failures return specific HTTP statuses: 404 for unknown endpoints, 405
with `Allow` for wrong methods, 411 for missing inference Content-Length, 413
for oversized bodies, 415 for unsupported inference media types, 431 for header
limits and 505 for unsupported HTTP versions. Other malformed framing returns
400 with a fixed parser error identifier (no request contents echoed). The text
service rejects declared payloads above 1 MiB before waiting for their bodies.
These responses precede inference and do not change post-start SSE error handling.

The text service does not implement interim `100 Continue` responses. Any
`Expect` header is rejected with 417 as soon as the complete headers arrive,
without waiting for the body. Clients must send the bounded request directly
without `Expect`; this also applies to discovery routes. This avoids a client
waiting for permission to send while the service waits for its payload.

Each accepted connection has a 30-second absolute request-read deadline (not
extended by trickled bytes). A partial request reaching that deadline receives
a best-effort 408 before closure; an entirely idle connection is simply closed.
Response writes have their own bounded 30-second deadline. Signal cancellation
interrupts polling; listener poll failures terminate serving with an error rather
than repeatedly retrying a broken descriptor. These paths are implemented but
have not been exercised in socket/fault-injection tests here.

Qwen shutdown distinguishes lock contention from a completed close failure.
Contention returns retryable unavailable without blocking on the plugin mutex.
The native BF16/INT4 engines cache their close result; a failed result is terminal,
retains the engine owner, and is not retried as if cleanup were still progressing.
The worker must exit rather than unload potentially live providers. This does
not bound the duration of a single CUDA/engine close call; external supervision
and shutdown fault-injection qualification remain required.

Qwen GPU initialization failures poison the current plugin activation, including
exceptions translated by the C ABI guard. They require worker restart, not an
in-process reload. Existing engine owners block a new load and plugin disposal.
The selected kernel capability ID must agree with its declared SM89/SM90 target;
a mismatched declaration is rejected before model loading. These admission
checks do not verify the GPU binary's numerical behavior.

For native text/token engines, terminal cleanup errors, malformed close status,
or expiry of the cleanup retry deadline cause a fail-stop process exit. The worker
emits `pih_worker_fail_stop` when diagnostics are writable, then uses `_Exit(1)`
without stack unwinding or global/DSO destructors. This is intentionally not a
graceful shutdown: do not expect final buffered output or completion receipts.
It prevents a normal-exit destructor from retrying cleanup of potentially live
device resources. Supervisors must treat the nonzero exit as failure, not successful
retirement. This path has only been compiled here, not fault-injection tested.

`/v1/completions` accepts a string `prompt` instead. Native request sampling now
accepts `temperature` in 0–2, `top_p` in (0,1], integer `n=1`, and an optional
integer `seed` in 0..INT64_MAX. Defaults remain greedy temperature=0, top_p=1,
n=1 and seed=0. Greedy mode rejects top_p other than 1. Positive temperature
selects the existing native stochastic packed sampler for BF16 and INT4;
both streaming modes use the same descriptor. Values are converted to the
sampler's FP32 representation, with underflow-to-zero rejected. An omitted
seed is deliberately zero, not implicit host randomness; this does not promise
cross-hardware or cross-build bitwise reproducibility. Sampling behavior and
quality remain unqualified until model tests run. `stream` is
a boolean, defaulting to false; true selects native incremental SSE output.
`stop` accepts null, one nonempty UTF-8 string, or 1–4 distinct strings of at
most 256 bytes each. Both modes match incrementally after UTF-8 decoding, retain
possible prefixes and omit the matched marker plus later text. Overlaps use the
earliest completed byte match (then earliest start), independent of chunking.
A match requests cooperative native retirement and reports `finish_reason=stop`
only after cleanup; disconnects/timeouts are not converted into successful stops.
Usage counts committed model tokens, including hidden stop tokens and any tokens
already committed while retirement is pending. Stop/fault regression test code
has been added but not executed; end-to-end behavior remains unqualified.
Chat permits system/user/assistant text and must end with a user. No images,
tools, reasoning histories or structured outputs. Unsupported fields are rejected.
The prompt plus output budget must fit `--max-context`. 40960 is a format ceiling,
not measured capacity, and lowering it does not resize all fixed native arenas.
Requests are bounded to approximately 1 MiB and 128 messages with network timeouts.

## Explicit later observation

The Qwen generation loop checks cancellation/deadline before submission, after
native steps and before each token callback. Output/cancellation callback
exceptions are recorded without abandoning the submitted request: later output
is suppressed and normal cancel/drain continues. A callback failure is reported
only after retirement (or without submission); an engine/cleanup failure still
poisons the engine and requires restart. These are cooperative boundaries, not
preemption of a blocked callback or CUDA operation. Fault-injection and real
disconnect/timeout verification remain pending.

Create an external evidence directory first, then execute when ready to test:

```bash
python3 deploy/native.py probe --prompt 'What is 1+1? Answer only the number.' \
  --max-tokens 32 --output /srv/evidence/qwen-first.json
```

The file must not exist. The script records the request, raw response, HTTP status
and end-to-end duration. It returns zero only for a matching model, nonempty answer
and positive completion-token usage. Errors return nonzero and are recorded.
This does not assess semantic correctness, time to first token or qualification.

Check Chinese/English decoding, arithmetic, repeated-request isolation, output
limits, 400 responses for invalid inputs, and post-shutdown GPU memory. Compare
against a pinned official revision with the same precision and input. Keep the
Git commit, Lock, driver/toolkit versions, GPU UUID, model revision/hashes, raw
answers and failures. HTTP 200 alone is not a successful model qualification.

## CPU-native tokenizer tool

`pih-tokenize` shares native BPE code with the Qwen plugin and adds the
DeepSeek-0731 tokenizer format. It does not start a GPU, execute a model, or use
a Python tokenizer. Linux builds require C++20, CMake 3.26+, OpenSSL 3, and ICU
development libraries (`libicu-dev` on Ubuntu).

```bash
cmake -S . -B out/build/tokenizer -G Ninja \
  -DPIH_DEPLOYMENT_PROFILE=custom -DPIH_BUILD_WORKER=OFF \
  -DPIH_BUILD_PLUGINS=OFF -DPIH_BUILD_MONOLITH=OFF \
  -DPIH_BUILD_PYTHON=OFF -DPIH_BUILD_TESTS=OFF \
  -DPIH_ENABLE_CUDA=OFF -DPIH_BUILD_TOKENIZER_TOOLS=ON
cmake --build out/build/tokenizer --target pih-tokenize pih-deepseek-format pih-deepseek-semantic-verify --parallel 2
out/build/tokenizer/plugins/common/pih-tokenize \
  deepseek-v4-flash-0731 /srv/models/DeepSeek-V4-Flash-0731/tokenizer.json request.json
```

Create UTF-8 `request.json` with `{"text":"complete raw prompt"}` to encode or
`{"tokens":[0,1,2]}` to decode. Standard output is one JSON object; errors go to
standard error with exit code 2. Use family `qwen3` for Qwen. No BOS insertion or
chat template is applied. DeepSeek decoding preserves structural tokens; Qwen
decoding skips special tokens and stops at its end tokens. Request files are
limited to 1 MiB, output to 8 MiB, and token arrays to 65536 entries.

This tool only reads local files; it does not download, authenticate, or publish
model artifacts. Family-structure validation is not pinned-revision hash
verification. The native text plugin uses the authenticated path described below;
raw prompt encoding is not chat encoding. Equivalence between ICU and
the official tokenizer implementation has not been tested.

### DeepSeek conversation formatting and output parsing

The same CPU build also produces `pih-deepseek-format`, a native port of the
pinned DeepSeek-0731 encoding rules. It does not import or execute Python:

```bash
out/build/tokenizer/plugins/common/pih-deepseek-format conversation.json
```

Example `conversation.json` (UTF-8):

```json
{"operation":"encode","thinking_mode":"chat","messages":[{"role":"system","content":"Be concise."},{"role":"user","content":"你好"}]}
```

The result is `{"text":"..."}`, suitable as the input format of `pih-tokenize`.
Optional encode fields: `drop_thinking` (default true), `add_bos` (default true),
and `reasoning_effort` (`low`, `high`, `max`; default low). `thinking_mode` is
`chat` or `thinking`. Message-level `tools`/`response_format`, assistant
`tool_calls`, tool-result merging/sorting, reminders and task markers are
rendered in the DeepSeek format. This is a complete-conversation API; it does
not accept a separately pre-encoded context or prefix-cache state. Put all
conversation messages in `messages`. At most 128 messages, 128 tools/calls per
list, and 1 MiB serialized input/prompt are admitted.

To parse a completed model turn, use a request with `operation: "parse"`,
`thinking_mode`, and `text` containing the raw decoded output **including EOS**.
The output contains `role`, `content`, `reasoning_content`, and `tool_calls`.
For example, chat-mode `text` may be `"Hello<｜end▁of▁sentence｜>"`; thinking-mode
text must also contain `</think>` before the final content. DSML calls are
parsed as data only, never executed. Truncated output, duplicate parameters,
invalid non-string JSON arguments, trailing data, and malformed delimiters are
rejected with exit code 2. Zero-parameter calls allow the blank arguments line
emitted by the official renderer. Output is bounded to 8 MiB.

The format implementation and its production command have been compiled, finally
linked and included in the PP1 release component. Renderer equivalence tests,
Linux execution and end-to-end HTTP qualification have not been performed. This
command is not a replacement inference backend.

For later incremental-parser inspection, the format tool also accepts:

```json
{"operation":"decode","thinking_mode":"thinking","chunks":["Reasoning","</thi","nk>Answer<｜end▁of▁sentence｜>"],"stopped":true}
```

It returns incremental `deltas` and the final `message`. Each chunk must already
be valid UTF-8; the tokenizer is responsible for retaining split byte sequences.
Use `stopped:false` for length exhaustion. Reasoning/content is emitted
incrementally, whereas tool calls are withheld until full DSML validation.
An unfinished tool block at the length limit is not emitted as content or calls.
No data may follow EOS. The offline operation is bounded to 65536 chunks and
8 MiB request/output; it is a future verification aid, not a live HTTP endpoint.

### Verify the pinned semantic snapshot

For authenticated DeepSeek tokenization, provide a materialized snapshot of
`deepseek-ai/DeepSeek-V4-Flash-0731` revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. It must contain the original bytes of
`config.json`, `generation_config.json`, `tokenizer.json`, `tokenizer_config.json`,
`encoding/README.md`, `encoding/encoding_dsv4.py`, `encoding/test_encoding_dsv4.py`,
and all four `encoding/tests/test_input_N.json` / `test_output_N.txt` pairs.
Model weights are not required for this operation. All paths must be real
directories/regular files, including the snapshot's parent components; symbolic
links (including typical Hub cache file links) are deliberately rejected. Copy
the exact file contents into a separate materialized snapshot without editing
JSON whitespace, line endings, or source text.

```bash
python3 deploy/native.py verify-deepseek-semantics --build \
  --snapshot-dir /srv/models/DeepSeek-V4-Flash-0731 --jobs 4
out/build/tokenizer/plugins/common/pih-tokenize \
  deepseek-v4-flash-0731-verified /srv/models/DeepSeek-V4-Flash-0731 request.json
```

The build requires Linux and ICU/OpenSSL development dependencies but no CUDA
or GPU. The Python command only orchestrates native builds and admission. It
does not download files, execute encoding Python, or run models. Omit `--build`
when the native tools already exist. For machine-readable admission metadata
without orchestration logs, invoke
`out/build/tokenizer/plugins/common/pih-deepseek-semantic-verify SNAPSHOT_DIR`
directly. Success emits revision, closure root, object count and token geometry;
failure returns 2. It reads bounded copies through no-follow descriptors and
hashes the complete semantic closure before constructing the tokenizer from
the same bytes. This receipt is not signed, not weight verification, not an
equivalence test, and not model/hardware qualification. The native API is ready
used by the native text plugin's load operation; target qualification is pending.

## Offline native PP1 weight verification

For an already converted generation, check the full native artifact before
service startup (Linux CPU, C++20/CMake/Ninja/OpenSSL development prerequisites;
no GPU, CUDA or ICU required):

```bash
python3 deploy/native.py verify-deepseek-artifact --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" --jobs 4
```

Use a nonzero root from an independently trusted publication. The verifier
reads every listed weight byte and checks hashes, manifest geometry, index,
runtime records, ownership and actual safetensors headers. All directory/file
symlinks are rejected. Full hashing can take considerable time and disk I/O;
it does not modify files or run inference. Omit `--build` after building.
For JSON-only stdout, directly run
`out/build/deepseek-artifact/plugins/offline-deepseek/pih-deepseek-artifact-verify GENERATION_ROOT ARTIFACT_ROOT`.
Exit 0 emits an observation; exit 2 indicates failure. The observation is not
immutable storage admission, source-conversion equivalence or hardware evidence.
Worker must still validate storage independently. Native weight conversion and
publication are not implemented by this command.

### Source-bound verification of existing weights

To verify source-to-target equivalence of an existing generation without
converting or writing it again, use the stronger read-only mode:

```bash
python3 deploy/native.py verify-deepseek-artifact --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --jobs 4
```

All five source/converter authority roots are required together with
`--source-dir`; partial authority is rejected, never downgraded to target-only
verification. Source, target and build directories must be disjoint. This mode
reads the full original checkpoint and target, including MTP source tensors
excluded from PP1, but never writes either directory. Successful JSON has
`source_payload_equivalence=true`; target-only verification reports `false`.
Neither result is a publication receipt, immutable lease or model qualification.
The roots must be independently trusted; no automatic trust provisioning exists.

For JSON-only stdout, run the native executable with positional arguments:
`pih-deepseek-artifact-verify --source-bound SOURCE_DIRECTORY GENERATION_DIRECTORY MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT`.
Exit 0 reports success; exit 2 reports failure on stderr without a success JSON.

For the existing-format structured verification projection, replace the native
flag with `--source-bound-projection` (the same eight positional arguments).
It emits canonical ASCII JSON with **no trailing newline**, so hashing its stdout
bytes gives the projection SHA-256. The result includes the converted shard
table, index/records/manifest hashes and sizes, conversion/layout/disposition
roots and `verification_scope=converted_bytes_and_source_payload_non_authorizing`.
It is returned only after all source-bound checks succeed. The C++ observation
also carries these exact JSON bytes and their digest; target-only observations
leave both absent/zero. No file is created or published by this mode.

The wrapper accepts `--verification-projection` together with `--source-dir`
and all five roots, but also prints orchestration logs: invoke the native binary
directly when capturing canonical bytes. This output is not yet a publication
receipt or catalog activation. The production command has been compiled and
finally linked, but this preparation flow has not been executed on Linux.

### Native PP1 weight preparation

The Linux CPU-only preparation command uses the same build prerequisites as
verification. Supply independently admitted model, semantic, inventory and
payload roots plus a converter identity root tied to the reviewed converter
build. Do not invent these hashes or copy them from an untrusted source just to
pass validation. Automated trust-root provisioning is not yet supplied.

```bash
mkdir -m 700 /srv/models/deepseek-pp1-staging
python3 deploy/native.py prepare-deepseek-artifact --build \
  --source-dir /srv/models/deepseek-original \
  --staging-dir /srv/models/deepseek-pp1-staging \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --jobs 4
```

Source must contain config, index and all 48 original shards, with no symlink
components. Staging must already exist, be empty, owned by the current user and
not group/other writable. Keep it outside the repository and disjoint from the
source and build directories. The native command checks staging before hashing
the source and checks again before writes; this does not lock out concurrent
writers, which the operator must exclude. Reserve over 156 GB for output plus filesystem
overhead; source bytes remain intact. Several full read passes are performed.
No GPU or inference is used. The command derives source semantics/inventory,
compares expected roots, copies 44 PP1 shards, writes metadata and fully verifies
the output. Final verification freshly rebuilds source semantics, inventory,
payload hashes, disposition and target metadata; it checks the exact target
member set, canonical headers and every target tensor against its source payload
hash. This uses the shared native planning implementation, not an independently
implemented numerical oracle. It adds full source/target read passes and reports
`source_payload_equivalence=true` only after success. It does not publish,
activate or start a service.

For JSON-only stdout invoke `pih-deepseek-artifact-prepare` from the build's
`plugins/offline-deepseek` directory with positional arguments in this order:
source, staging, model root, semantic root, inventory root, payload root,
converter identity root. Exit 0 emits an observation with `published=false`;
exit 2 retains partial staging files. Inspect failures and use a fresh empty
staging directory; never rerun over partial output or delete the source. Keep
the successful artifact root for later independent verification. Local checks
cover syntax only: Linux link/runtime and real conversion remain unqualified.

### Generation store initialization and structural verification

The Linux CPU-only store tool creates the frozen `store.json`, `.staging`,
`generations`, `receipts` and `pointers` layout. Its parent must already exist,
be owned by the caller and not group/other writable. The new store path must
not exist, even as an empty directory or symlink. Keep the store outside the
repository and disjoint from the build directory.

```bash
python3 deploy/native.py deepseek-generation-store init --build \
  --store-dir /srv/models/deepseek-store --jobs 4
python3 deploy/native.py deepseek-generation-store verify \
  --store-dir /srv/models/deepseek-store
```

Native JSON-only invocation is `pih-deepseek-generation-store init|verify ABSOLUTE_STORE_DIRECTORY`.
Exit 0 means the requested structural operation succeeded; exit 2 means failure.
Initialization creates directories with mode 0700 and the manifest with 0600,
syncs them and their parent, then reopens and validates. Existing stores are
never overwritten. Failed initialization retains any partial store: inspect it,
do not blindly retry, and choose a fresh path instead of deleting model data.

Verification is read-only: canonical manifest, exact root members, no symlink
components, ownership/modes, distinct same-filesystem directories and child
name/type rules. It allows at most 65,536 entries per directory. It does **not**
verify generation payloads, receipt contents or the current pointer's contents;
the output explicitly reports `member_contents_verified=false`. Callers must
exclude concurrent writers; retained descriptors and mutation checks are not a
lock or immutable lease. This tool does not publish weights or activate a model.
The generation-store command and the Worker store-resolution path have been
compiled and finally linked. They have not been executed against a real Linux
repository, and runtime qualification remains outstanding. The commit/activation
commands below are separate from this structure-only operation.

### Commit a prepared generation and receipt (without activation)

Prepare the generation directly in an empty private directory such as
`/srv/models/deepseek-store/.staging/run-001`, using the preparation command above
with that `--staging-dir`. The store must already be initialized. Do not move an
unverified directory into `generations` manually. After preparation succeeds:

```bash
python3 deploy/native.py deepseek-generation-store commit --build \
  --store-dir /srv/models/deepseek-store --staging-name run-001 \
  --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" --jobs 4
```

This command performs fresh source-bound verification, atomically moves staging
with no replacement to `generations/sha256-<artifact-root>`, seals generation
members 0444 and the directory 0555, then repeats source-bound verification.
Only matching pre/post verification projections permit exclusive creation of
`receipts/<artifact-root>.json`; the receipt is reread, sealed 0444 and synced.
Existing generation or receipt destinations are rejected. Expect several full
source/target read passes and retain exclusive control of source/store writers.

The native positional interface is
`pih-deepseek-generation-store commit SOURCE_DIRECTORY STORE_DIRECTORY STAGING_NAME MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT`.
Exit 0 requires `success=true` and `receipt_committed=true`; it does **not**
activate anything. On failure, exit 2 may include an outcome JSON with
`success=false`: inspect `renamed`, `directories_synced`, `read_only_sealed` and
`receipt_may_exist`. These describe completed/possibly-started phases, not overall
success. Nonzero receipt-root metadata alone proves no publication. A moved
generation or partial receipt is retained; there is no automatic retry, rollback,
overwrite or cleanup. Source-admission failures can occur before an outcome exists.
No `pointers/current.json` update, model execution or immutable lease is provided.
Actual commit/crash-recovery operation has not been qualified locally.

### Activate a committed generation

Activation re-verifies source/target equivalence and the sealed stored receipt.
Supply the receipt root reported by successful commit and an independently
admitted catalog root. This command binds the catalog digest but does not parse,
authenticate or install catalog bytes. The first activation explicitly expects
no previous pointer:

```bash
python3 deploy/native.py deepseek-generation-store activate --build \
  --store-dir /srv/models/deepseek-store --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --receipt-root "$RECEIPT_ROOT" --catalog-root "$CATALOG_ROOT" \
  --activation-ordinal 1 --previous-pointer-root none --jobs 4
```

For later activations, supply the expected current pointer root and exactly the
next ordinal (bounded to INT64_MAX); stale predecessors, skipped ordinals and
overflow fail before pointer mutation. Commit and activation use the same
nonblocking advisory lock on the store root. Busy stores fail rather than wait;
external writers must still be excluded. Verification can read the entire source
and target checkpoint again.

The native interface is `pih-deepseek-generation-store activate SOURCE_DIRECTORY STORE_DIRECTORY MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT RECEIPT_ROOT CATALOG_ROOT ORDINAL PREVIOUS_POINTER_ROOT_OR_none`.
The tool exclusively writes and verifies `.current-<ordinal>-<root>.tmp`, then
atomically installs `current.json` (no-replace for the first pointer), syncs the
pointer/store directories and rereads the installed pointer. Success requires
exit 0, `success=true` and `activated=true`. Failure can still have
`pointer_replaced=true`; inspect `temporary_may_exist` and `directories_synced`
before deciding recovery. Retained temporary files intentionally cause structural
validation to fail until inspected/resolved; nothing is automatically rolled back
or removed. A root digest alone is not evidence of completed activation.

This switches only the store pointer. It does not reload a running Worker,
validate catalog contents, establish immutable storage admission or execute a
model. Hardware/runtime and crash-recovery qualification remain outstanding. The
production command and Worker integration are compiled and finally linked, but
no actual Linux activation was executed.

### Resolve and verify the active generation

To check an activated store without retaining the original checkpoint:

```bash
python3 deploy/native.py deepseek-generation-store resolve --build \
  --store-dir /srv/models/deepseek-store \
  --pointer-root "$POINTER_ROOT" --catalog-root "$CATALOG_ROOT" --jobs 4
```

Both roots must come from an independently trusted activation/catalog decision.
The native tool reads the current pointer under a shared advisory lock, validates
the sealed receipt against its expected root, joins the receipt's complete object
table to the artifact manifest, and hashes every target file. Exact membership,
sealed modes, ownership, canonical metadata, pointer and path stability are
checked. It does not re-scan source payloads or admit the catalog's contents.
Allow full target read time; a concurrent native commit/activation causes a busy
failure instead of a wait. No file is modified.

For JSON-only output run `pih-deepseek-generation-store resolve STORE_DIRECTORY EXPECTED_POINTER_ROOT EXPECTED_CATALOG_ROOT`.
Exit 0 returns `generation_name`, artifact/receipt/pointer/catalog roots and
activation ordinal, with `receipt_binding_verified=true` and
`source_payload_equivalence=false`. Exit 2 reports failure without a success
observation. The result identifies `STORE_DIRECTORY/generations/<generation_name>`;
it is not an open descriptor lease and must not bypass Worker artifact/storage
admission. `serve-deepseek --store-dir` connects this result to native Worker
startup as described above; it does not remove the Worker admission step. This
path is compiled and finally linked, but has not been executed with a real store.

## Troubleshooting and unfinished migration

The compressor now consumes the two raw BF16 checkpoint matrices directly.
Its development Kernel Pack request layout changed: rebuild the model and pack
together and regenerate the Lock; old fused-weight requests are rejected by
exact structure-size validation. Indexer query projection now binds raw E4M3
weights `[8192,1024]` and UE8M0 scales `[64,8]`, quantizes the BF16 query input,
and invokes the existing FP8 GEMM before RoPE. Its request layout also changed:
rebuild model/Pack/Lock together. Two separate scratch buffers add 1,032 bytes
per maximum query (before allocator alignment). CUDA compilation and numerical
execution remain unverified. Shared-expert runtime qualification and other native
weight-consumer gaps remain; a valid artifact does not prove the service can yet
execute every checkpoint tensor.

Qwen non-streaming generation now consumes native packed scheduler events inside
the plugin. The Lock's generation budget (ten minutes by default) triggers cooperative cancellation and
up to five further minutes of drain. Successfully cleaned-up timeout requests
return HTTP 504. Deadline checks occur between native steps, not by preempting
in-flight CUDA kernels. Disconnects (including an input half-close) and worker
stop signals trigger cooperative cancellation. This development service supports
neither HTTP pipelining nor client write-half shutdown.

Set `stream:true` and use `curl -N` for SSE. Each `data:` JSON chunk is decoded from
committed native token events, retaining incomplete UTF-8 across token boundaries;
the answer is not generated in full before splitting it. The terminal chunk
contains finish reason and usage, followed by `[DONE]`. Errors after headers use
an SSE `error` JSON and close without `[DONE]`; pre-stream invalid requests get
HTTP 400. A 30-second socket write timeout cancels slow consumers and drains the
request rather than retaining unbounded output. The native text contract is v2;
rebuild Worker, model, Surface, SDK and Lock together. No v1 compatibility bridge.

Output callback violations are terminal: duplicate starts, invalid stream modes,
writes before streaming starts, invalid buffer bounds, a wrong callback context,
or embedded CR/LF or invalid JSON in an SSE data value latch cancellation.
A successful non-stream response
must also be a nonempty JSON object. A provider cannot undo that failure by retrying
or returning success; the surface rejects the contract and emits no `[DONE]`.
The Linux native-contract test target contains callback regression cases; these
new cases have not been executed as part of the code-only cutover.

Missing ICU/OpenSSL/CUDA requires fixing development dependencies. A link error
requires preserving the complete log, not enabling the monolith to bypass it.
Build and bundle directories must be dedicated, disjoint trees: neither may be
the repository root or its ancestor. Sibling directories under `out/` are valid.
Qwen model directories must be disjoint from repository, build and bundle trees;
`serve` validates this before any requested build/install, not after installation.
Resolved paths are checked so an existing symlink cannot hide an overlap. This
preflight is not protection against another process changing paths concurrently.
Wrong GPU identity or format should not be bypassed by renaming a cubin/artifact.
Regenerate Locks after rebuilding. An inference 500 terminates the failed native
service; save logs before restart. This code has not yet been built on Linux/CUDA.

Qwen compute/scheduling sources are now reused inside a private native plugin
archive, not invoked through an old Python engine. Repository-wide migration is
still unfinished: V4-0731 weight preparation/qualification, V4.1/SM103 Linux
link and runtime validation, multi-GPU transport pluginization, and remaining
development tools. Compatibility bridges are not substitutes.
