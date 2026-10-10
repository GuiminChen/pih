# Native multi-sequence batching

English | [中文](batching.zh.md)

The text service uses `pih.inference.text.v3` and `pih.surface.text-http.v3`.
Regenerate deployment Locks and rebuild providers and consumers together. The
v2 headers remain installed for existing SDK users; v3 is a distinct capability,
not a silent change to the serialized ABI. `load`/`close` remain exclusive.
Concurrent `complete` calls keep callbacks on each calling thread and borrow
their response buffers only until that call returns.

| Model | Native shared execution | Configured sequence ceiling | Hardware path |
| --- | --- | ---: | --- |
| Qwen3 BF16 | Existing packed controller; shared prefill/decode plans | 32 | SM89 / SM90, single GPU |
| Qwen3 PIH INT4 | Same packed scheduler; INT4 model computations | 32 | SM89 / SM90, single GPU |
| DeepSeek V4 Flash | Shared dense/mHC/router plan; private sparse attention/KV/compressor lanes | 4 in the sealed PP1 profile | SM89, sealed native profile |
| DeepSeek V4.1 | Shared embedding, packed FFN/MoE and head; private attention/KV/Engram lanes | 32 | SM103, TP 2 / 4 / 8 |

These ceilings describe source admission limits, not tested hardware capacities.
Multiple attention lanes can execute separately within a shared model plan;
request batch and expert dispatch batch are distinct. No per-request engine
replicas or Python inference bridge are used. V4's old `inference.tokens.v1`
interface remains serialized; native continuous batching is exposed by the text
interface. V4.1's scalar supervisor command remains available for its v1 config.

## Capacity and configuration

All text paths admit at most `maximum_batch_sequences` live sequences and
`maximum_pending_requests` additional waiting requests. `maximum_batch_tokens`
bounds a shared execution plan, not the sum of a request's lifetime tokens.
Prompt and output must also fit the model's context and KV reservations.
Limits of 1 preserve single-sequence execution using the same batching path.
Capacity failures are explicit; requests never fall back to serial execution.
Queue exhaustion returns HTTP 429, and connection-worker exhaustion returns 503.

For Qwen's existing `deploy/native.py serve` or `seal` command and V4's
`serve-deepseek`, append:

```bash
--maximum-batch-sequences 4 --maximum-batch-tokens 4096 --maximum-pending-requests 64
```

Qwen reserves its existing conservative physical pool before admitting work.
V4 reserves sealed per-sequence KV and independent endpoint resources before
publishing readiness. V4 prefill advances one token per sequence per plan to
respect recent-ring overwrite dependencies; increasing the prompt count alone
would be unsafe. V4.1 plans disjoint private cache arenas plus shared packed
boundaries and expert workspaces against each rank's real device/host budget.
The rank count is independent of the request batch size. V4.1 admits each
whole prefill prompt within `maximum_batch_tokens`; a larger prompt is rejected
explicitly. It does not silently split into unsupported continuation-prefill
chunks.

V4.1 text service uses supervisor config v2. Prepare a new candidate from an
existing authenticated config, review its paths and budgets, and seal using the
independently trusted digest of the reviewed output:

```bash
bash deploy/run-v41-native.sh batch-config /absolute/old.json /absolute/new.json 4 4096 64
bash deploy/run-v41-native.sh seal-service /absolute/bundle /absolute/new.json TRUSTED_SHA256
bash deploy/run-v41-native.sh serve /absolute/bundle 8000
```

The v2 `batching` object contains the three snake_case fields above. Configuration
and rank bootstrap versions explicitly carry these bounds. See the complete
[supervisor configuration](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md).

## Scheduling and ownership

A single executor per engine submits GPU work, consumes completion events and
acknowledges each plan once. Events carry request/generation identity. Requests
join and exit dynamically, with independent sampling seed, stopping policy,
deadline and output accounting. A cancellation drains only that sequence's
submitted work before releasing its state. Device failure stops the engine and
retains resources until all submitted work has retired.

Bounded private output buffers separate caller callbacks from scheduling. A slow
consumer exhausts its own output credit; other sequences keep progressing.
Service shutdown stops admission and drains requests within the retirement
policy. V4.1 uses one persistent supervisor child and one rank group; the child
is created before HTTP threads and owns cgroup/process custody. All ranks see
the same canonical packed sequence order and global plan identity.

Ordinary JSON and SSE are both supported. For concurrent callers:

```python
from concurrent.futures import ThreadPoolExecutor
import json, urllib.request

def complete(index):
    body = json.dumps({"model": "Qwen/Qwen3-0.6B", "prompt": "Hello " * (index + 1),
                       "max_tokens": 8 + index, "temperature": 0, "top_p": 1,
                       "seed": 7 + index, "stream": False}).encode()
    request = urllib.request.Request("http://127.0.0.1:8000/v1/completions", body,
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(complete, range(4)))
```

Set `stream: true` and consume the SSE body on each caller separately for streaming.

## Verification scope

Windows CPU tests exercise real packed controller plans and native scheduling
interfaces for batch 1/2/4, event routing, variable prompts, cancellation,
deadlines, slow consumers and queue rejection. V4 endpoint allocation tests
check nonaliasing and release-to-zero. V4.1 tests exercise production packed
request/dispatch ledgers, canonical multi-rank layout, cache arena separation
and actual Engram state reset. Production session tests exercise real collectors,
commit gates and output credits with simulated OS/tokenizer boundaries, including
batch 1/2/4 over simulated 2/4/8 ranks, dynamic joining, cancellation,
stop, deadlines, slow output credit and queue exhaustion. Portable tests do not establish CUDA arithmetic.

The 2026-10-09 acceptance rebuilt the current sources with CMake 4.4.4,
Clang 22.1.8 and genuine CUDA 13.2.86 tools. All four deployment profiles
(`qwen-4090d`, `qwen-h100`, `deepseek-pp1`, `deepseek-b300`) compiled their CUDA
machine code, linked production executables/plugins with strict undefined-symbol
checks, and installed fresh bundles. Inspection passed for 62 ELF/device products:
16 / 16 / 16 / 14 respectively. Qwen cubins contain 18 entrypoints per architecture;
V4 and V4.1 packs contain 17 and 20 embedded device images. Real Linux CRT,
OpenSSL, ICU, CUDA, cuBLAS and NCCL libraries were used, without stubs or unresolved
symbol exemptions. This is cross-compilation and linking evidence, not Linux execution.
Clang's CUDA-version support warning remains; a native NVCC build is still pending.

The current Release CPU acceptance passed 10/10 executable targets with real
GoogleTest 1.17 and ICU 74.2 libraries. Qwen's cancellation/deadline/reuse regression
also passed 10 repeated runs (100 test cases). Tool tests passed 125 tests and 14
subtests. The installed SDK passed its 26-header audit and an external C11/C++
consumer build and test. Release bundles now include both batching configuration
and HTTP parity tools; their isolated installed CLI checks passed.

The host has Intel graphics and no NVIDIA GPU or Linux execution environment.
Real service startup, authenticated-model inference, batch 1/2/4 numerical parity,
actual GPU plan membership, multi-rank communication and performance remain
**unverified** for all four models. No hardware capacity is inferred from these builds.

Rebuild on the target Linux host with its supported CUDA toolkit and development
libraries, using new bundle destinations:

```bash
for profile in qwen-4090d qwen-h100 deepseek-pp1 deepseek-b300; do
  python3 deploy/build-release.py "$profile" \
    --build-dir "$PWD/build/acceptance-$profile" \
    --bundle "$PWD/bundles/acceptance-$profile" --jobs 4
done
```

The formal top-level `pih-native-contracts` clean build also linked 22 Linux
x86-64 test executables (13 GoogleTest programs and 9 direct tests). These Linux
binaries were not executed locally. Acceptance fixed ICU dependency scope/linkage,
test include paths and two unreachable V4.1 CTest registrations.

On Linux with GoogleTest, ICU and OpenSSL development packages installed:

```bash
cmake -S . -B build/batch-contracts -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPIH_ENABLE_CUDA=OFF -DPIH_BUILD_NATIVE_CONTRACT_TESTS=ON \
  -DPIH_BUILD_TOKENIZER_TOOLS=ON -DBUILD_TESTING=ON
cmake --build build/batch-contracts --target pih-native-contracts --parallel 4
ctest --test-dir build/batch-contracts --output-on-failure
```

Start each rebuilt bundle with authenticated artifacts and freshly sealed Locks,
using the configuration commands above. To compare deterministic single calls
against concurrent calls for each loaded model:

```bash
for batch in 1 2 4; do
  python3 tools/verify_batch_service.py --url http://127.0.0.1:8000 \
    --model Qwen/Qwen3-0.6B --batch-size "$batch" --max-tokens 16 \
    --output "/absolute/new-parity-${batch}.json"
done
```

Use `deepseek-ai/DeepSeek-V4-Flash-0731` or `deepseek-ai/DeepSeek-V4.1-Flash` for their
services, and run Qwen separately for BF16 and INT4. The parity tool compares
text, finish reason and token counts. Its report explicitly marks execution-plan
verification false: HTTP concurrency alone cannot prove packing. CPU plan tests
provide that structural evidence; CUDA numerical and throughput evidence must
come from actual hardware runs. No GPU parity or performance pass is claimed.
