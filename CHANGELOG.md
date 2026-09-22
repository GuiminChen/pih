# Changelog

## Implement bounded V4.1 canonical tensor partition copying

- Add a CPU-only copy plan for replicated, physical-row and physical-column
  partitions, with explicit tail padding and checked shape/byte arithmetic.
- Stream through at most 1 MiB caller-owned workspace and exact I/O callbacks,
  without allocating per-row plans or materializing large Engram tables.
- This is a conversion stage, not a complete HF converter: source authentication,
  numeric canonicalization, rank manifests and publication still need integration.

## Implement Qwen stop strings on the native output path

- Share bounded incremental stop matching across streaming/non-streaming output,
  withholding possible prefixes and omitting matched markers and later bytes.
- Separate normal callback-requested stop from disconnect cancellation while
  using native cancel/drain for retirement; retain committed-token accounting.
- Add unexecuted boundary test cases. Source checks do not qualify runtime behavior.

## Preserve Qwen request retirement on output callback failure

- Catch output/cancellation callback exceptions inside generation, suppress
  further publication and continue cancel/drain instead of unwinding ownership.
- Recheck cancellation/deadlines before submission and each token publication,
  and guard cleanup-deadline arithmetic. Native failures still poison the engine.
- Source compilation only; no model or fault-injection test was run.

## Connect Qwen HTTP sampling to the native packed sampler

- Parse bounded temperature/top-p/seed and integer n=1 into the existing native
  sampling descriptor for BF16/INT4, instead of rejecting all non-greedy input.
- Preserve greedy defaults and both EOS IDs; use explicit seed zero when omitted
  and reject conflicting parameters or FP32 underflow before generation starts.
- No stochastic model execution, quality or reproducibility test has run.

## Add standalone native Qwen BF16 source verification

- Share the bounded source-expectation parser between conversion and a new
  CPU-only `pih-qwen-source-verify` command; reject zero/noncanonical authority.
- Wire `verify-qwen-source --build` into deployment tooling and CPU build CI.
  No weights were verified or converted and no model was run locally.

## Make fixed-device deployment preflight explicit about CUDA ordering

- Parse full physical GPU/PCI inventory without silently skipping invalid rows.
- Require management index 0 to be first in PCI order and set CUDA PCI ordering
  before worker startup, retaining the ban on visibility-mask remapping.
- MIG/MPS, topology-race protection and device exclusivity remain unqualified;
  no GPU detection or execution was run locally during this change.

## Select Qwen Kernel Packs by deployment hardware

- Add validated exact Qwen architecture selection to CMake cubin/pack targets.
- Require `build --target` and propagate serve's target into the native build;
  build/install one selected pack and refuse mixing a pre-existing other pack.
- Preserve explicit dual-architecture CMake builds for development. Configuration
  checks do not establish CUDA compilation or hardware qualification.

## Pin the Qwen service tokenizer before engine allocation

- Add a bounded byte-snapshot loader that checks the repository's Qwen3
  tokenizer digest before parsing the same bytes or allocating the GPU engine.
- Reuse the loader for CPU chat-token inspection. Raw tokenizer tooling remains
  separate; weight/configuration provenance and runtime qualification are open.

## Separate Qwen artifact writing from runtime verification

- Extract write-plan and artifact-write functions into private offline writer
  files. The shared artifact I/O header now exposes verification only and no
  longer includes the canonical extent writer.
- Add the writer to the conversion library and historical source-test inventory;
  the model's conversion-source exclusion also covers this new unit.

## Make Qwen conversion interfaces private to offline tooling

- Move five conversion-only headers out of `include/pih/model` and next to
  their offline implementations. Update implementation/test includes without
  exposing a new global include root or retaining old forwarding headers.
- Keep shared artifact-format interfaces separate; their ownership migration
  and historical monolithic test replacement remain unfinished.

## Relocate Qwen conversion implementation to its offline owner

- Move five conversion source files from `src/model` to `plugins/offline-qwen`,
  updating native and historical test build references without forwarding files.
- Preserve the existing public/internal headers for remaining consumers; shared
  format ownership and monolithic test source cleanup are still unfinished.

## Exclude offline conversion from the Qwen model compilation

- Split five conversion/quantization implementation units into
  `pih_qwen_conversion_impl`, removing them from the native model source list.
- Link the verifier to artifact-format code only, not the conversion pipeline.
  Shared format ownership and old monolithic source inventory cleanup remain.

## Connect Qwen artifact verification to deployment commands

- Add `deploy/native.py verify-int4` with explicit expected bindings and optional
  CPU-only build, reusing conversion's native tool selection/build helper.
- Explicitly disable unrelated artifact/tokenizer tools when configuring this
  preparation build. No model, conversion or artifact verification was executed.

## Harden Qwen conversion and expose read-only artifact verification

- Require round-to-nearest and reject fast-math quantizer compilation; disable
  fast-math and contraction in the offline target.
- Check expectation-file growth and receipt delivery, distinguishing failures
  after publication without deleting or overwriting artifacts.
- Add `pih-qwen-int4-verify` for independent structural/payload checks against
  explicit expected file/source/binding/disposition digests. No runtime or
  numerical qualification is implied; the command has not been executed.

## Separate CPU-only Qwen INT4 artifact preparation

- Move the native converter entry to `plugins/offline-qwen` and link an explicit
  offline source set instead of the CUDA model engine. Add the independent
  `PIH_BUILD_QWEN_ARTIFACT_TOOLS` build option.
- Add `convert-int4 --build` for CPU-only preparation without installing the
  inference bundle. Conversion, full linking and hardware qualification have
  not been executed locally as part of this change.

## Correct Qwen chat text tokenization boundaries

- Coalesce adjacent plain-text segments before BPE and encode caller content
  without recognizing added-token spellings; keep trusted template controls
  separate. Raw completion and other model Encode paths are unchanged.
- Share bounded chat token assembly between the service and optional CPU
  format-tool token output. Extend the later reference comparison to five
  cases and optional pinned-tokenizer token-ID comparison; not executed yet.

## Share the native Qwen plain-chat template with offline qualification

- Extract the existing service template into a shared native segment plan;
  retain control-token/text boundaries and add a rendered-byte bound.
- Add CPU-only `pih-qwen-format`, include it in the native build workflow,
  and switch the reference comparator away from the legacy Python renderer.
- Explicitly scope this comparator to non-thinking plain text. Tools, thinking,
  token equivalence and model qualification remain separate unfinished work.

## Point Qwen tokenizer qualification at the native implementation

- Replace the legacy Python candidate in the offline reference comparator with
  an explicitly selected `pih-tokenize` executable; no runtime fallback remains
  in this tool. Keep pinned input bytes, check native output structure and IDs,
  apply per-case deadlines, and report native executable observations.
- Document CPU build and later qualification commands. No tokenizer corpus or
  model test was executed by this migration; other legacy consumers remain.

## Decouple offline evidence tooling from legacy inference imports

- Move the pure Qwen teacher-forced quality compiler to `tools/evidence`,
  update all consumers and remove its old package exports. The standalone
  compiler no longer adds the legacy Python package to its import path.
- Move pure DeepSeek M5 evidence processing out of `python/pih` into
  `tools/evidence`, updating the collection tool and existing tests without
  retaining a compatibility alias or importing the removed native extension.
- Keep offline evidence tooling outside the client wheel and model runtime.

## Remove the legacy Python inference extension entry

- Delete the three pybind extension source files and the `_pih` build/install
  target; explicitly reject enabling the removed Python inference build.
- Remove the obsolete extension-injection runner and its dedicated tests, and
  stop the CPU CI job from generating/running that compatibility extension.
- Keep native C++ contracts and client wheel checks. Remaining old Python/tool
  dependencies and monolithic C++ source cleanup are not yet complete.

## Native V4.1 Linux supervisor link gate

- Add a build-only Linux CI job compiling/linking the CPU supervisor with CUDA,
  NCCL, Python, monolith, plugins and tests disabled, then inspect its ELF direct
  dependencies without executing it.
- Retain compiler/environment, configuration/build logs and linker/dependency
  evidence. Adding this workflow does not establish that a CI run has passed.

## Native V4.1 development build/check/run script

- Add a Linux shell entry for explicit-architecture native builds, digest/schema
  configuration checks and single-request runs without Python/legacy fallback.
- Add a non-launching `--check-config` supervisor mode and bilingual deployment
  guidance distinguishing development commands from qualified support.

## Native V4.1 single-request supervisor CLI

- Connect trusted configuration, request/executable admission, random endpoints,
  delegated cgroup preparation and supervisor orchestration in one native command.
- Add bounded-backpressure raw token output, signal/output-failure cancellation,
  explicit cleanup and dedicated-subreaper descendant reaping; send child logs
  to stderr instead of mixing them into the token stream.
- This is a source-compiled development entry, not a validated model service;
  complete Linux linking, GPU execution and HTTP deployment remain pending.

## Native V4.1 authenticated supervisor configuration

- Add bounded strict-schema parsing and authenticated read-only loading for
  executable identities, library/cgroup placement, model artifacts, requests,
  sampling/stopping budgets and timeout durations.
- Emit typed rank templates with no user-supplied NCCL IDs, process identities
  or channel names. Runtime assembly/CLI wiring remains a separate next step.

## Native V4.1 supervisor operation assembly

- Add a CPU orchestration entry connecting admitted requests/executables,
  isolated NCCL bootstrap, rank launch, generation and publication handoff.
- Route early failures to unlaunched-resource cleanup and post-launch failures
  to group/session retirement using explicit startup custody, not just state names.
- Preserve primary and cleanup failures separately; completion requires the
  existing process/cgroup retirement barriers. CLI/HTTP integration remains pending.

## Native V4.1 broker custody in generation sessions

- Bind rank startup IDs to the live NCCL broker and reject shared rank/helper
  cgroups or insufficient broker lifetime.
- Transfer helper retirement custody with the generation session, check its
  liveness through all-rank readiness, and reap it before submitting inference.
- Include helper retirement in startup failure/cancellation and terminal cleanup;
  do not report completed cleanup while a broker remains unretired.

## Native V4.1 isolated NCCL bootstrap process

- Add a pinned-NCCL helper that returns one private packet ID and remains alive
  for the NCCL bootstrap listener, with a parent-death signal and lifetime bound.
- Add CPU-side broker custody, exec/ID admission, liveness polling and explicit
  cgroup kill, pidfd reap and group removal. Reuse the admitted executable and
  environment launch path without inventing a rank bootstrap envelope.
- Remove in-process communicator ID creation. Service lifecycle integration and
  real process/NCCL validation remain pending.

## Native V4.1 explicit worker environment

- Replace arbitrary environment arrays in worker/group launch with a native
  environment owner: fixed locale/device ordering and bounded explicit library
  directories, without inherited variables, preload/audit or visibility overrides.
- Check library directory components, ownership and shared-write permissions;
  reject loader separators/substitution tokens. This is not shared-object or
  system-wide NCCL/loader configuration authentication.
- Enforce the single-threaded supervisor precondition through procfs immediately
  before clone, instead of leaving the fork/exec lock-safety rule only in comments.

## Native V4.1 sealed worker executable admission

- Authenticate a bounded, read-only, single-link x86-64 ELF against a trusted
  deployment digest and copy it into an executable, fully write-sealed memfd.
- Require the typed executable owner in both rank-group and child launch APIs;
  do not execute a mutable path or accept a raw executable descriptor fallback.
- This covers executable bytes only, not dynamic-library/environment admission;
  actual memfd execution and process tests remain deferred.

## Native V4.1 CPU supervisor request admission

- Separate CPU sampling validation, stopping, ledger and rank-process channels
  from CUDA build/link dependencies; GPU sampling reuses the CPU validator.
- Add an owner for authenticated tokenization, bounded prompt/context admission,
  publication storage and ledger, with canonical sequence bootstrap binding.
- Reject generation handoff when the ledger's completion reservation exceeds
  the launched workers' context. Service entry and GPU validation remain pending.

## Native V4.1 authenticated tokenizer adapter

- Add an independent V4.1 tokenizer family pinned to the official frozen
  tokenizer.json SHA-256, with authenticated read-only artifact loading.
- Own stable raw vocabulary views for generation and bound rendered-prompt
  encoding to the native prefill limit. Chat rendering and execution validation
  remain separate, unfinished work.

## Native tokenizer raw-byte vocabulary access

- Expose exact pre-UTF-8-repair token bytes and bounded dense ID-ordered byte
  tables for controller stop matching without corrupting split code points.
- Reuse the same raw decode implementation for existing text decoding; preserve
  family-specific special-token behavior without declaring V4.1 equivalence.

## Native V4.1 atomic generation-session custody transfer

- Construct generation/session state inside the admitted worker group, matching
  ledger identity, sampling, prompt length and deadlines to launched workers.
- Transfer process/cgroup retirement authority only after session construction
  succeeds; remove raw handoff that left failure cleanup to external assembly.

## Native V4.1 cgroup/process retirement integration

- Carry the launched cgroup owners through rank-group handoff into generation
  sessions; gate success on process reaping plus empty-group removal.
- Kill contained descendants on fault escalation and partial-startup cleanup,
  retaining removed-group masks and reporting unresolved cleanup separately.

## Native V4.1 worker cgroup preparation

- Create fresh delegated cgroup-v2 children and verify memory/swap/OOM-group,
  PID and CPU limits before exposing descriptors to the rank launch group.
- Add explicit subtree kill, populated-state observation and nonrecursive
  removal of the same owned empty member; no destructor mutation.

## Native V4.1 supervised worker group startup

- Validate cross-rank bootstrap agreement and unique devices/cgroups/endpoints,
  prepare all listeners/sealed envelopes, then retain every spawned pidfd.
- Admit all protocol bundles before one-time process-watch handoff; retain
  partial launch masks and explicitly kill/reap failed startup groups.

## Native V4.1 rank worker executable entry

- Add pih-v41-rank-worker, joining sealed FD-3 bootstrap, authenticated channels
  and artifacts, aggregate memory planning, backend activation and rank runtime.
- Require supervisor launch protections and zero-argument invocation; perform
  explicit backend shutdown after runtime release and fail at process boundary.

## Native rank backend plugin activation stack

- Reuse microkernel loaders/registration barriers to activate backend-only
  stacks from digest-authenticated lock snapshots with required binary hashes.
- Resolve unique declared contracts and explicitly shut down after resource
  retirement; quarantine retained host contexts on ambiguous provider failures.

## Native V4.1 worker artifact admission

- Load digest-bound HF configuration, compressed-token map, rank manifest and
  plugin-lock snapshots before GPU initialization; retain authenticated weights.
- Expose snapshot-based deployment-lock parsing without reopening the verified
  lock path; enforce startup deadline checks between admission stages.

## Native V4.1 authenticated weight manifest loading

- Hash bounded rank manifests against independently admitted digests before
  strict JSON parsing and config/topology/shard-identity validation.
- Open fixed manifest members without symlinks, check read-only single-link
  storage and read stability, then authenticate shards through the file owner.

## Native V4.1 sealed bootstrap envelope

- Add strict fixed-size startup encoding for rank/device/sampling/budgets,
  deadlines, channels, NCCL ID and independently admitted artifact digests.
- Create/read immutable memfd envelopes and validate supervisor identity before
  clone; reject noncanonical padding, missing identities and malformed paths.

## Native V4.1 worker spawn primitive

- Launch admitted ELF descriptors with clone3 PIDFD/INTO_CGROUP and execveat,
  sealed bootstrap FD 3, explicit argv/environment and parent-death protection.
- Preserve process custody through setup/exec errors and provide bounded,
  explicit partial-startup kill/reap without destructor signalling or waits.

## Native V4.1 rank channel bundles

- Assemble four independent authenticated streams with fixed controller/worker
  directions, delayed post-spawn peer binding and one shared admission deadline.
- Expose protocol references only after every stream is attached; close
  temporary connectors/listeners while retaining the protocol-owned descriptors.

## Native V4.1 authenticated post-exec sockets

- Add single-use abstract Unix listeners and nonblocking worker connectors,
  validating peer PID/UID before exposing connected descriptors to protocols.
- Reject backlog-full connects, late admission and unexpected peers without
  reconnecting or substituting prefork socketpair credentials.

## Native V4.1 normal worker process reaping

- Require nonblocking pidfd wait/reap and zero exit status for every released
  rank before the supervised generation session reports success.
- Preserve consumed-child masks when normal teardown falls back to fault
  retirement, avoiding duplicate wait/reap and signals against reaped ranks.

## Native V4.1 supervised lifecycle barriers

- Gate generation on authenticated ready notices from every pidfd-bound rank;
  after all terminal receipts, authorize teardown and collect release notices.
- Route startup/teardown channel faults into existing bounded process
  retirement while preserving committed output leases and failure ledgers.

## Native V4.1 worker lifecycle channels

- Add canonical 64-byte ready/retire/released messages over authenticated
  nonblocking Unix endpoints, bound to rank, world and base sequence identity.
- Wire ready publication, retirement authorization and release acknowledgment
  into the worker runtime with explicit startup/execution/teardown deadlines.

## Native V4.1 rank runtime lifecycle

- Connect CUDA handles, nonblocking NCCL initialization, weight/memory startup
  and the rank request loop in a single explicit lifecycle.
- Keep terminal ranks alive for supervisor-authorized teardown; finalize NCCL,
  retire memory and release CUDA handles in order, retaining fault ledgers.

## Native V4.1 worker memory assembly

- Connect authenticated weight upload, exact prompt/decode expert workspaces
  and inference memory initialization under one aggregate allocation budget.
- Validate config/rank/device bindings, retain partial-failure ledgers and
  order normal teardown after completed consumers and a final stream fence.

## Native V4.1 weight allocation/upload ownership

- Allocate exact catalog-sized device weights and bounded pinned staging via
  public plugin memory contracts, then drive authenticated chunked upload.
- Gate access on completed upload and fence final consumers before explicit
  release; preserve allocation records on partial allocation or provider failure.

## Native V4.1 worker communicator ownership

- Add plugin-local NCCL 2.31.2 nonblocking initialization for TP 2/4/8,
  rank/device identity validation and deadline-bound lifecycle polling.
- Separate finalize/destroy from fault abort, preserve ambiguous handles and
  prohibit retries after destructive calls. No legacy model dependency added.

## Native V4.1 worker CUDA handle ownership

- Prepare the selected device and create retained context, nonblocking stream
  and distinct computation/inference/expert retirement events via public APIs.
- Fence normal retirement before reverse-order explicit release; preserve
  handles and per-resource release records after ambiguous provider failures.

## Native V4.1 supervised generation session

- Connect generation failure/cancellation to bounded pidfd process retirement
  and post-reap unaccepted output-credit reconciliation.
- Distinguish successful completion, failed-but-retired and unreconciled
  failure; preserve already committed output leases through cancellation.

## Native V4.1 fault process retirement

- Add explicit pidfd-only SIGTERM/SIGKILL escalation and nonblocking parent
  reaping with retained per-rank completion records and absolute deadlines.
- Bind retirement to the failed ledger and permit unaccepted output-credit
  discard only after all ranks are reaped; never revive a failed ledger.
- Fix failed-ledger cleanup being rejected even after verified retirement.

## Native V4.1 rank pidfd liveness gate

- Retain supervisor-provided pidfds, validate bounded fdinfo process identities
  and bind them to authenticated request/receipt peers.
- Poll rank liveness without blocking or reaping during generation and directly
  before local commit; fail on exit, invalid descriptors or unverifiable checks.

## Native V4.1 normal sequence termination

- Upgrade request framing to strict version 2 with explicit terminal requests;
  terminal frames contain no tokens and cannot enter GPU inference.
- Close generation only after every worker validates the final retired step
  and sends its matching terminal receipt. Preserve explicit resource release.

## Native V4.1 controller generation loop

- Drive output-credit reservation, per-rank request sends, receipt collection,
  decoded-token commit and ledger-derived next-token dispatch.
- Transfer committed output leases explicitly and pause dispatch when output
  credits are exhausted; never regenerate a locally committed token.

## Native V4.1 rank worker execution loop

- Connect authenticated request reception, admitted identity/parameter checks,
  exact prefill/decode workspace selection, inference retirement and receipt send.
- Reject replayed plans and ordinal/shape drift; quarantine sequence resources
  on transport/execution failure without automatic replay.

## Native V4.1 worker request transport

- Add a dedicated authenticated Unix request endpoint with bounded partial
  frame progress and nonblocking sends/receives.
- Decode only complete request frames and permanently fail timed-out,
  disconnected or malformed channels; no automatic resend or execution.

## Native V4.1 worker inference request codec

- Encode bounded token payloads, step identity and sampling parameters in a
  fixed little-endian frame with strict reserved/unused-field checks.
- Validate decoded requests against the worker's local sequence position
  before starting inference through its memory owner.

## Native V4.1 controller rank receipt collector

- Prepare ledger output credit before dispatch and collect one authenticated
  channel receipt per rank with bounded nonblocking polls.
- Expose a candidate only after all ranks complete; check every socket boundary
  before local token commit and fail the ledger on interrupted execution.

## Native V4.1 Linux rank receipt channel

- Attach nonblocking connected Unix stream endpoints with supervisor-supplied
  PID/UID validation and fixed rank binding; own only a CLOEXEC duplicate.
- Incrementally send/receive bounded receipt frames, fail on timeout or EOF,
  and pass complete frames into the rank commit gate.
- Add a nonblocking pre-commit boundary check for queued data/disconnection;
  retain supervisor liveness and global fault ordering requirements.

## Native V4.1 rank receipt wire representation

- Encode completed rank receipts as fixed 256-byte little-endian frames,
  without native struct padding or ABI dependence.
- Decode through the commit gate with exact length/version/reserved-field and
  canonical candidate checks, then apply identity/rank/position admission.
- Require the caller to supply source rank from its authenticated connection;
  payload metadata is never treated as peer authentication.

## Native V4.1 rank completion and token commit gate

- Expose rank receipts only after completed inference and memory retirement,
  including step identity, ordinal, processed length and last-rank candidate.
- Require all TP2/4/8 receipts before local token staging/commit; reject
  duplicate ranks, stale identities, inconsistent positions and expired steps.
- Keep transport authentication and serialized global fault ordering explicit
  controller responsibilities; receipts are not network credentials.

## Native V4.1 explicit inference memory ownership

- Allocate planned device/pinned-host arenas through public CUDA capabilities,
  validate provider identities, and initialize device state asynchronously.
- Gate step execution, output access, reuse and explicit release on completion
  events; preserve partial allocation/release ledgers on provider failure.
- Quarantine failed execution without automatic replay, synchronization or
  destructor frees; retain separate release flags for fault reconciliation.

## Native V4.1 combined inference memory layout

- Compose boundary, phase, cache, attention, compressor, Engram, alternating
  FFN and eight retained indexer arenas into one bounded device layout.
- Bind exact device/pinned-host allocations to borrowed inference resources
  with uploaded identity and weight/workspace overlap checks.

## Native V4.1 ordinary inference-step operation

- Connect embedding, forty-layer backbone, gathered head logits and last-rank
  sampling under one absolute deadline and explicit completion states.
- Bind boundary arenas and shared completion/count storage; reject overlap
  with backbone/weight storage before input submission.
- Keep sampled candidates separate from controller acceptance and stream
  publication. No automatic commit, token loop, resource freeing or retry.

## Native V4.1 forty-layer backbone operation

- Sequence all forty ordinary-text layers after embedding, waiting for block
  completion and expert-workspace retirement before advancing.
- Admit disjoint borrowed arenas with alternating FFN residual/pre storage and
  eight retained index-owner arenas; preserve shared cache/candidate lifetimes.
- Expose output only after the full step completes; poison interrupted or
  failed execution without retrying or freeing in-flight allocations.

## Native V4.1 sequence-derived phase planning

- Plan separate aligned query/compressed position and phase-table regions,
  with token budgets, exact extents and incomplete-group handling.
- Derive block layer/position/token/stream/error metadata from the admitted
  sequence and generate phase descriptors internally; remove the external
  StepPhasesLaunch parameter without retaining a compatibility overload.

## Native V4.1 planned indexer and template-free block entry

- Plan bounded index-query/key, scoring, selection and candidate-mask storage
  with uploaded rank/configuration binding and shared-cache admission.
- Derive all backbone source stages from step phases, layer roles and sequence
  state. Remove the caller-supplied IndexedSourcesLaunch from BlockOperation;
  the new entry has no template compatibility overload.

## Native V4.1 planned compressor execution

- Plan bounded, distinct projection/pooling/normalization/rotation scratch for
  ratio-one and ratio-two compressed KV owners, including incomplete decode
  groups with no emitted output.
- Generate compressor topology and descriptors inside the block entry from
  configuration, sequence cache and planned attention output; retain full
  weight, phase and cross-stage lifetime admission before enqueue.

## Native V4.1 planned Engram execution

- Plan five distinct aligned Engram lookup/projection/gate segments with
  per-rank configuration and token-capacity checks.
- Generate Engram launches inside the block entry only for configured layers
  1/14, using sequence-owned hashes and residuals. Remove the caller-supplied
  optional Engram descriptor; retain full pre-enqueue graph admission.

## Native V4.1 attention output arena and generated block tail

- Plan bounded concatenated KV/indices, sparse-attention/inverse-RoPE output,
  grouped/final projection, FP32 reduction and mHC residual storage.
- Generate the complete attention residual descriptor and remove the caller's
  prepared-block template from BlockOperation::Start; output and FFN descriptors
  now come from their plans with full pre-enqueue graph admission.

## Native V4.1 attention preparation transient plan

- Plan distinct mHC, Q low-rank/expansion, normalization, RoPE and window-KV
  scratch segments with rank/token budgets and uploaded weight binding.
- Generate attention preparation descriptors in the block entry and connect
  generated hidden/low-rank/compressed-latent outputs to compressor/indexer
  inputs; reject absent latent production for index-key stages.

## Native V4.1 FFN transient arena and descriptor generation

- Plan distinct mHC, routing/dispatch, shared FP8 expert and residual segments
  with bounded token prefixes and memory budgets; bind uploaded weights and
  validate generated FFN route/tail descriptors.
- Construct FFN descriptors inside the block entry from the plan and existing
  expert workspace accumulator, replacing manually populated FFN templates.

## Native V4.1 input/head/sampling arena layout

- Add budgeted device and pinned-host boundary layouts with stable addresses
  across prefill/decode token prefixes, separate tensor segments and explicit
  256-byte alignment.
- Generate embedding, token/hash upload, head and sampling descriptors plus
  host counts/error/candidate views. Weight/state admission remains in the
  existing execution entries; this does not allocate a complete model.

## Native V4.1 persistent sequence cache plan

- Plan a budgeted 256-byte-aligned persistent cache arena: all 40 window rings,
  compressed/index-key storage only on KV owners, and ratio-two pooling state.
- Allocate through the plugin memory contract with preserved ownership ledger;
  bind planned cache addresses/capacity inside the block entry and reject
  capacity/config changes or weight-arena overlap.

## Native V4.1 execution consumes uploaded weights

- Replace the block entry's caller-supplied expert array with a completed
  upload object and bind all layer weights internally after wiring data edges.
- Require the same uploaded arena for embedding, blocks and head; protect it
  from input uploads and all retained-sequence writes from the first token.
- Keep wiring distinct from graph admission; validation remains mandatory
  before GPU enqueue. Removed the old entry signatures without overloads.

## Native V4.1 uploaded compressed-KV/indexer bindings

- Bind compressor value/gate/norm and index query/score/key/norm tensors only
  for their configured owner layers, retaining rank-local head partitions.
- Admit incomplete compression steps without inventing outputs; validate
  optional key/candidate stages and protect scratch plus persistent caches
  against the complete uploaded weight arena.

## Native V4.1 uploaded layer bindings

- Bind attention/FFN mHC, canonical text router, both Engram layers and
  attention query/window/output projections from completed uploads.
- Validate configuration/TP/compression geometry and protect all supplied
  writable regions against the full weight arena. Reject visual masks in this
  text-backbone binding path; compressed KV/indexer binding remains separate.

## Native V4.1 uploaded input/head/expert bindings

- Bind admitted embedding, final norm/head, shared FP8 experts and rank-local
  routed FP4 expert arrays by canonical name, with descriptor validation and
  rank/config checks where applicable. No in-place descriptor mutation.
- Preserve config/rank identity in the catalog and reject writable scratch
  overlapping the complete uploaded weight arena, including other layers.

## Native V4.1 staged weight upload

- Connect retained, hash-checked runtime files to bounded pinned-host CUDA
  uploads with per-chunk completion events, deadline/device checks, and no
  staging reuse before copy completion.
- Reject non-finite weight/scale encodings and noncanonical Engram padding;
  expose tensor addresses only after all copies and final file revalidation.
  Resource allocation/retirement and complete model binding remain separate.

## Native V4.1 descriptor-owned runtime shard reads

- Add Linux CPU-only `BackboneWeightFiles`: no-follow directory traversal,
  sealed regular-file admission, exact size/full SHA-256 checking against
  externally supplied identities, and catalog construction from retained FDs.
- Bound tensor staging reads to 1 MiB, resolve offsets from the admitted catalog,
  and check inode/path/stat identity before and after reads. This is not manifest
  authority, numerical payload validation or device-upload completion.

## Native V4.1 runtime shard catalog

- Join parsed safetensors headers to the exact per-rank backbone inventory,
  rejecting duplicate shards/tensors, unknown or missing tensors, incorrect
  storage and dimensions, incomplete ranges and trailing payload.
- Produce owned file locations and budgeted 256-byte-aligned device offsets;
  canonical packed E2M1 uses U8 physical storage. Header admission is not
  payload authentication, conversion or a completed device loader.

## Native V4.1 backbone weight inventory and attention transport precision

- Add a canonical runtime weight inventory for the 40-layer text backbone,
  including TP1/2/4/8 slicing, routed expert ownership, Engram padding, physical
  FP4 shapes, byte accounting and exact metadata-set validation. This is not a
  raw-checkpoint converter or authenticated loader.
- Correct attention wo_b transport to promote local BF16 output into separate
  FP32 scratch, SUM in FP32, and round to BF16 before mHC. Extend cross-stage
  alias/liveness admission to the new scratch; no BF16 collective fallback.

## Native V4.1 unified token/hash upload

- Snapshot host token IDs once, derive both Engram hash planes from the admitted
  contiguous hash state, and upload token IDs plus layer-major hashes using pinned
  sources on the embedding stream before lookup.
- Bind Engram layers to completed sequence-owned hash regions; protect those
  regions against subsequent writes and poison on post-hash/upload failures.
  The integrated input path is text-only and rejects image gating masks.

## Native V4.1 token-to-backbone input

- Added sharded BF16 token embedding, TP SUM, four-stream residual expansion and
  initial one-hot mHC pre coefficients, with bounds/numeric/alias admission.
- Bound layer zero to completed embedding publication instead of caller-supplied
  hidden state. Protected embedding weights/IDs across later stages and rejected
  stale head sampling once the next input has been prepared.

## Native V4.1 reserved token output delivery

- Connected local ledger preparation/commit to admission-allocated publication
  slots backed by the existing output-burst credit implementation, linked without
  the monolithic core. No candidate is staged before the reservation is in flight.
- Return generation-bound queue receipts after acceptance; retain slots until
  explicit consumer release. Separate prepared abort from retired-worker discard;
  preserve pending receipts for fault-owner reconciliation.

## Native V4.1 local accepted-token ledger

- Added admission-reserved record storage, frozen request sampling/stop policy,
  monotonic plan preparation and candidate identity/ordinal/position validation.
- Staged stop transitions without mutation and committed token/logprobs, finish,
  processed length, pending input and stochastic ordinal as one serialized local
  update. Abort does not consume RNG; distributed/output-credit authority remains
  outside this local transaction.

## Native V4.1 transactional token stopping

- Added bounded raw-byte stop matching across tokens, EOS/stop-token suppression
  below min_tokens, earliest-end/longest-match precedence and max-token flush.
- Separated nonmutating preview from local state commit; bound transitions to
  instance identity and predecessor count to reject stale, foreign and repeated
  commits. Distributed ledger/output publication remains controller work.

## Native V4.1 bounded candidate readback

- Added pinned-host candidate transfer before the completion event, followed by
  owned-copy validation of token/suppression, RNG word, logprob fields and padding.
- Replaced the device-only candidate accessor with a value observation carrying
  epoch/plan/generation/config identity, ordinal and processed length. This remains
  tentative data, not an accepted-token ledger or distributed commit proof.

## Native V4.1 tentative GPU sampling

- Added canonical greedy/stochastic parameter validation, stable vocabulary
  ordering, fixed-tree softmax sums, top-k/top-p filtering, Philox4x32-10 and
  selected/top-20 full-distribution logprobs without CPU logits fallback.
- Bound sampling to this step's completed head on the last rank, with sequence
  reservation, cache/scratch admission and zero-error completion. Candidates are
  tentative: controller acceptance, stopping and ordinal advancement remain open.

## Native V4.1 sequence-to-logits head

- Added final mHC collapse/RMSNorm plus FP32 vocabulary projection for the last
  position, with rank-local contiguous vocabulary shards and FP32 all-gather.
- Bound output to a completed backbone sequence and reserved that sequence until
  zero-error gather completion; protected persistent caches and rejected repeated
  head publication. Sampling and model loading remain separate work.

## Native V4.1 completion-owned sequence publications

- Added a nonmovable sequence state to block execution: ordered 40-layer steps,
  contiguous single-token decode, residual/pre carry and completion-only cache
  publication. Failed or abandoned active blocks invalidate the sequence.
- Bound shared KV/index/candidate consumers to their completed producers;
  preserved persistent cache allocation identity and rejected later writes into
  other layers' retained caches/publications. Exposed step output after layer 39.

## Native V4.1 Engram inside block execution

- Required Engram on configured layers 1/14 and bound its completed gate output
  into both attention mHC input and residual, before phase/source preparation.
- Extended whole-block lifetime checks to Engram inputs, weights and scratch;
  retained one workspace reservation across Engram, sources, attention and FFN.

## Native V4.1 RoPE generation inside block execution

- Bound generated query/compressed phase tables to all source consumers and
  inverse RoPE, replacing the earlier prepared-phase block signature.
- Added phase position/output writes to whole-block liveness admission and
  enqueued generation after workspace reservation, before source preparation.

## Native V4.1 step-bound RoPE generation

- Added device-side original-position sequence generation before RoPE tables.
- Bound query/compressed tables to backbone step geometry and owner-layer emitted
  rows, preserving original group positions rather than compressed slot numbers.

## Native V4.1 automatic source-to-block operation

- Added one completion-controlled operation from source/index preparation through
  attention, FFN residual return and workspace retirement on 2/4/8 ranks.
- Ran binding/liveness admission before source enqueue and carried one workspace
  reservation through all phases. Engram/cache publication/model integration remain open.

## Native V4.1 source-to-block lifetime admission

- Added cross-stage live-region checks protecting source/index caches, inputs,
  coefficients and later attention/FFN weights across the combined block.
- Included full expert workspace capacity and ordered weight extents; only the
  common error flag may overlap writes across source and execution stages.

## Native V4.1 source-to-block descriptor binding

- Added explicit binding of produced Q/window/index/cache-prefix and mHC regions
  into attention/FFN block descriptors, retaining declared geometry checks.
- Required external admission for reused shared caches/indices; combined graph
  liveness and automatic source-to-block orchestration remain unfinished.

## Native V4.1 source-to-index operation

- Connected attention sources to index queries, head weights and owner-layer
  index keys, with cross-graph liveness and step/rank/phase admission.
- Added completion-controlled multi-rank execution and explicit shared-index,
  ratio-zero and empty-prefix skips. Full block attachment remains unfinished.

## Native V4.1 combined attention sources

- Joined mHC/query/window preparation with the configuration-owned compressed KV
  producer, binding normalized hidden state and absolute step metadata.
- Added cross-graph protection for qr, coefficients, cache/pool state, weights and
  phases. Shared-cache identity and indexing orchestration remain external.

## Native V4.1 owner-layer compressed KV preparation

- Connected compressor projection/pooling/normalization to compressed RoPE and
  cache writes with frozen layer ownership and step-derived slot admission.
- Handled zero-emission steps without normalization/cache launches and preserved
  unrotated latent for indexing. Full attention orchestration remains open.

## Native V4.1 attention input preparation chain

- Connected mHC input processing to query and window KV generation/cache update
  with backbone layer, rank geometry, phase-region and stream admission.
- Preserved normalized hidden/qr and carry coefficients through whole-chain
  buffer checks. Compressed/indexed attention preparation remains unfinished.

## Native V4.1 prepared attention-to-FFN block

- Connected prepared attention output/residual execution to unified FFN execution
  with one workspace reservation and completion-controlled stage transitions.
- Added layer/rank/residual/pre binding and cross-stage cache/weight protection.
  Upstream Q/KV/index/cache preparation and whole-model scheduling remain open.

## Native V4.1 whole-operation workspace reservation

- Reserved expert workspace before FFN route submission, preventing release or
  a second use while routing/count observation is pending.
- Bound continuation and retirement to non-reused reservation identities and
  retained reservations on partial failure for explicit fault retirement.

## Native V4.1 unified FFN operation

- Added one operation spanning routing, admitted counts, managed expert execution,
  residual completion and workspace retirement on 2/4/8 ranks.
- Copied host descriptors and added pre-route protection for persistent expert
  weights, plus count-independent accumulator binding. Whole-model integration remains open.

## Native V4.1 FFN workspace lifecycle integration

- Replaced manually supplied FFN batches with admitted weights and a workspace
  owner; constructs descriptors and binds the accumulator before submission.
- Connected use/retirement states to FFN completion and protected full allocation
  capacity from aliasing externally retained outputs, coefficients or weights.

## Native V4.1 workspace retirement ledger

- Added plugin-backed workspace ownership states with use fencing, nonblocking
  retirement observation, reuse and explicit checked release.
- Prohibited release while in use, retained original handles on ambiguous
  failures, and avoided destructor-time frees of potentially active storage.

## Native V4.1 plugin-backed workspace allocation

- Connected expert workspace requests to the backend CUDA memory capability,
  validating allocation device, generation, alignment and exact extent.
- Preserved ambiguous provider output for ownership reconciliation instead of
  discarding handles or guessing cleanup. Runtime retirement remains explicit.

## Native V4.1 reusable expert workspace

- Added bounded, aligned workspace sizing and automatic per-expert descriptor
  construction from admitted counts and ordered FP4 weight regions.
- Reused sequential expert scratch with a separate FP32 accumulator, exact
  row-prefix extents and whole-batch validation. GPU allocation remains runtime-owned.

## Native V4.1 observed-route FFN continuation

- Connected admitted routing to local expert batch completion and multi-rank
  reduction/shared-expert/residual completion through a non-replaying state machine.
- Bound hidden/routing/accumulator/coefficient connections and protected retained
  coefficients and persistent expert weights across stages. Allocation remains external.

## Native V4.1 FFN input-to-routing chain

- Connected mHC input processing, normalization, router and asynchronous expert
  dispatch/count observation under backbone layer/connection validation.
- Added whole-chain live-buffer checks while preserving carried versus newly
  computed mHC coefficients. Expert workspace ownership remains caller-managed.

## Native V4.1 expert-to-FFN residual return

- Connected shared-expert merge to mHC post with full cross-stage buffer checks,
  including protection against in-place reduction overwriting residual inputs.
- Updated the native multi-rank expert pipeline to record completion after the
  residual update; removed its prior merge-only descriptor signature.

## Native V4.1 reduction-to-shared-expert pipeline

- Connected FP32 all-reduce enqueue observation to shared expert computation,
  BF16 merge and final error/event admission for 2/4/8 ranks.
- Added whole-chain buffer preflight, device readmission, deadline and non-replay
  failure states. Whole-model batch/residual orchestration remains outstanding.

## Native V4.1 FP32 routed reduction and shared merge

- Added in-place FP32 NCCL SUM with the existing transport's release, device,
  communicator and asynchronous-state checks, without early BF16 conversion.
- Added FP32 routed/BF16 shared addition followed by final BF16 rounding and
  numeric error checks. Reduction/merge scheduler integration remains open.

## Native V4.1 local expert batch

- Connected admitted dispatch counts to ordered execution of every local expert,
  one-time accumulator initialization and completion/error observation.
- Added dispatch descriptor binding and cross-expert live-weight alias checks,
  while permitting sequential scratch reuse. Cross-rank merge remains open.

## Native V4.1 expert count observation

- Added asynchronous dispatch/count readback with pinned storage, device-domain
  admission, completion/error gating and deadline handling.
- Exposed cached per-expert row counts only after successful observation and
  capacity validation. Full expert scheduling and resource ownership remain open.

## Native V4.1 gathered expert chain

- Connected token/weight gather, routed FP4 expert execution and FP32 scatter
  under one preflight-validated native call, including empty-expert handling.
- Added whole-chain alias and connection admission with a standalone native
  target. Count observation, completion ownership and MoE scheduling remain open.

## Native V4.1 expert result scatter

- Added routing-slot-based BF16 expert output accumulation into FP32 token rows,
  without applying routing weights a second time.
- Added count/slot/ownership-range checks, empty-expert handling and nonfinite
  error reporting. Scheduler integration and MoE reduction remain outstanding.

## Native V4.1 expert token gather

- Added paired hidden-state/routing-weight gathering for an owned expert with
  device count, slot ordering and expert-ID validation.
- Added safe invalid-row zero fill and empty-expert count checks. Count readback,
  execution scheduling and result accumulation remain outstanding.

## Native V4.1 expert dispatch planning

- Added per-rank expert counts and stable flattened routing-slot tables with
  explicit unused sentinels and bounded per-expert capacity.
- Added invalid-ID/duplicate-route detection. Gather, expert execution scheduling
  and output accumulation remain outstanding.

## Native V4.1 MoE router

- Added FP32 gate projection, sqrt-softplus scoring, text/image correction bias
  selection and top-6/top-3 routing with unbiased normalized output weights.
- Added exact buffer/mask admission and deterministic tie handling. Dispatch,
  expert ownership, result accumulation and numerical qualification remain open.

## Native V4.1 routed expert chain

- Connected mixed FP8/FP4 gate/up/down projections with required pre-down
  routing-weighted SwiGLU.
- Unified shared/routed chain admission while retaining distinct weight formats.
  Expert selection, token dispatch and result accumulation remain outstanding.

## Native V4.1 mixed FP8/FP4 projection

- Added packed E2M1 weights with per-row block-32 E8M0 scales and FP8 activation
  quantization, sharing the native projection kernel with the FP8 path.
- Added exact mixed-format buffer admission. Routed-expert chaining, dispatch
  and numerical/hardware qualification remain outstanding.

## Native V4.1 shared FP8 expert

- Connected gate/up FP8 projections, clipped SwiGLU and down projection with
  full-chain buffer admission and the frozen 2304-wide intermediate geometry.
- Added optional pre-down-projection route weighting to the standalone activation.
  Routed FP4 expert projection, routing and complete FFN scheduling remain open.

## Native V4.1 mHC sublayer preparation

- Connected mix generation, carried-pre collapse and RMSNorm with complete
  live-buffer checks and explicit separation of current versus carried pre.
- Preserved the BF16 intermediate and next-sublayer coefficients. Complete
  attention/FFN producer scheduling remains outstanding.

## Native V4.1 multi-rank attention residual state machine

- Connected assembled output, NCCL SUM, mHC residual expansion and error/event
  completion with deadline checks and non-replayable failures.
- Rechecked communicator/device before post-reduction work and communicator
  state at completion. Input-producer scheduling and hardware qualification remain open.

## Native V4.1 attention reduction and residual

- Added exact-group/rank admitted BF16 output SUM using the shared NCCL path.
- Connected assembled attention to single-rank mHC residual expansion with
  cross-stage alias checks; partial-rank output is explicitly rejected.
- Multi-rank residual scheduling and hardware qualification remain outstanding.

## Native V4.1 assembled attention output

- Connected KV/index assembly through sparse attention, inverse RoPE and both
  output projections, with whole-chain live-buffer admission.
- Kept rank-local output explicit; final TP SUM, residual expansion and full
  attention scheduling remain outstanding.

## Native V4.1 attention assembly

- Added window/compressed KV concatenation, native window-index construction
  and causal compressed-index guards, connected to sparse attention.
- Supported empty compressed prefixes without dummy keys. Layer/source ownership,
  complete attention scheduling and GPU numerical qualification remain open.

## Native V4.1 indexer layer admission

- Required frozen configuration and layer roles at single-/multi-rank execution
  entry points, enforcing index ownership, key emission and candidate roles.
- Checked prefill/decode concatenated-KV offsets before submission. Buffer
  provenance and full cross-layer sequence ownership remain outstanding.

## Native V4.1 single-rank indexer

- Split shared indexer graph admission from the NCCL translation unit and added
  a single-rank execution target without NCCL dependencies.
- Connected all 32-head stages with event/error completion, deadlines and
  non-replayable failures. Full model scheduling and hardware qualification remain open.

## Native V4.1 indexer input integration

- Connected query preparation and optional index-key/cache updates to the
  multi-rank indexer state machine before scoring and reduction.
- Added whole-graph scratch liveness and compressed-step/cache-prefix checks.
  Cross-layer ownership, single-rank scheduling and full model integration remain open.

## Native V4.1 indexer tail state machine

- Connected weighted scoring, NCCL enqueue polling, candidate generation/top-k
  and event/error completion under an explicit deadline and whole-chain preflight.
- Reused generic error-flag completion and made failed submissions non-replayable.
  Query/key preparation, cross-layer ownership and model integration remain open.

## Native V4.1 index-score reduction

- Factored the existing NCCL operation through a bounded BF16 SUM submission
  entry and reused it for rank-local index scores with exact TP/rank admission.
- Added the opt-in indexer NCCL target. Collective enqueue and GPU completion
  remain distinct; full runtime state-machine integration is still outstanding.

## Native V4.1 two-level candidate selection

- Added causal block-of-eight maxima, newest-block pinning and top-2048
  candidate masks, sharing bounded-heap ordering with index top-k.
- Connected optional candidate masking to ratio-one top-k with mask validation
  and excluded-position fillers. Cross-layer identity, TP score reduction and
  performance qualification remain outstanding.

## Native V4.1 causal top-k

- Added causal compressed-position visibility and top-512 selection with
  position-sorted, offset-adjusted indices and `-1` unreachable fillers.
- Added deterministic tie handling and bounded per-query scratch. Candidate
  block selection, TP reduction and throughput qualification remain outstanding.

## Native V4.1 indexer scoring

- Added BF16 head-weight projection with the global-head normalization factor,
  rank-local query/key scoring and a fully validated connected launch.
- Preserved BF16 dot/product/output rounding boundaries with FP32 accumulation.
  TP score reduction, masking, selection and numerical qualification remain open.

## Native V4.1 index-key input

- Added BF16 index-key projection from unrotated compressor latent and connected
  RMSNorm, RoPE, FP4/E8M0 quantization and asynchronous cache transfer.
- Added full-chain liveness, dimensions and cache-range checks without changing
  the original latent. Runtime ownership/publication and numerical qualification
  remain outstanding.

## Native V4.1 indexer query input

- Connected FP8 query expansion, RoPE and group-32 FP4/E8M0 simulated
  quantization with full-chain geometry and live-buffer checks.
- Shared E2M1 rounding with compressed KV while retaining distinct scale
  formats. Index keys, scoring, candidate selection and top-k remain open;
  no complete-indexer or GPU qualification claim is made.

## Native V4.1 compressed KV cache writes

- Added independent forward-RoPE and cache-write preparation preserving the
  unrotated latent for index-key projection.
- Added group-16 E2M1 quantize/dequantize with E4M3 scales and BF16 cache storage,
  matching the reference storage format rather than claiming packed FP4 memory.
- Added exact extent/alias admission and device error reporting. Full sequence
  ownership, index integration and GPU numerical qualification remain open.

## Native V4.1 compressor chain

- Added BF16 ratio-one and FP32 ratio-two value/gate projections from hidden
  state, with explicit weight formats and no old-runtime or Python fallback.
- Added ratio-two softmax pooling, persistent half-group state and conditional
  RMSNorm, preserving the BF16 rounding boundary before normalization.
- Connected both branches with full preflight and live-buffer checks. Compressed
  cache/index integration and GPU numerical/performance qualification remain open.

## Native V4.1 attention input chains

- Connected low-rank query projection, RMSNorm, head expansion and forward RoPE,
  preserving normalized query state for the indexer.
- Connected FP8 KV projection to window preparation, with whole-chain shape,
  stream/error and live-buffer checks. Compressed selection, cross-chain model
  scheduling and runtime qualification remain outstanding.

## Shared V4.1 FP8 projection and attention wo_b

- Extracted the native block-32 FP8 linear implementation and removed Engram's
  duplicate projection kernel; fixed Engram geometry now calls the shared path.
- Connected FP8 `wo_b` after attention/inverse RoPE/grouped projection to produce
  5120-wide rank-local output with full-chain alias checks. Final TP SUM and
  numerical/model execution qualification remain outstanding.

## Native V4.1 grouped attention output

- Added BF16 `wo_a` block-diagonal projection with FP32 accumulation and explicit
  1/2/4/8 local-group geometry.
- Connected sparse attention, inverse RoPE and grouped projection with full-chain
  admission. FP8 `wo_b`, final TP reduction and engine integration remain open;
  no numerical or performance qualification is claimed.

## Native V4.1 sparse attention

- Added shared-KV sparse attention with block-64 online softmax, BF16-rounded
  value probabilities, learned attention sinks and masked-row handling.
- Added bounds/alias checks and device-side invalid-index guards. This initial
  CUDA implementation still requires numerical/performance qualification and
  complete model/cache integration; no runnable-model capability is registered.

## Native V4.1 sliding-window KV updates

- Added whole-vector block-32 FP8 quantize/dequantize and bounded 128-slot BF16
  ring updates for ordinary prefill/decode, including rotary tail values.
- Connected KV RMSNorm, RoPE and cache update with preflight alias/shape checks.
  Cache ownership, sparse attention and numerical/model execution qualification
  remain outstanding.

## Native V4.1 ordinary attention-step planning

- Added query/causal window indices, ring writes, compressed-group positions and
  tail-slot updates for bounded prefill and single-token decode.
- Distinguished index-key ownership from top-k selection ownership; rejected
  unsupported chunked-prefill/DSpark use instead of inferring ordinary decode.
  Device cache integration and model execution remain outstanding.

## Native V4.1 text position encoding

- Added layer-specific RoPE/YaRN phase generation from explicit original token
  positions and tail-64 BF16 rotation/inverse rotation, including in-place use.
- Added strict bounds/layout/alias admission and device error reporting. Original
  compressed-position planning, numerical qualification and attention integration
  remain outstanding; no hardware capability is registered by these primitives.

## Native V4.1 text normalization

- Added BF16-input RMSNorm with FP32 arithmetic for fixed text widths
  128/512/1280/5120 and explicit BF16/F32 weight storage.
- Connected carried mHC pre-collapse to sublayer RMSNorm while preserving the
  BF16 intermediate. Eleven SDK compatibility syntax checks pass; no numerical
  or model execution qualification is claimed.

## Native V4.1 mHC primitives

- Added FP32 mixing projection/Sinkhorn coefficients, explicit carried-pre
  collapse, residual expansion and one-hot pre initialization for 4 × 5120 streams.
- Kept V4.1's delayed pre-mix semantics separate from the old fused mHC path.
  Launch contracts and CUDA compatibility syntax checks pass; numerical/model
  execution and full engine integration remain outstanding.

## NCCL SDK syntax coverage

- Added checksum-pinned header-only extraction from the official NCCL 2.31.2
  CUDA 13 package and optional NCCL coverage in the Engram syntax checker.
- The reduction adapter and TP scheduler passed checks against actual SDK headers;
  reports distinguish this evidence from linking and multi-rank execution.

## Reproducible CUDA header compatibility checks

- Added checksum-pinned, header-only NVIDIA package extraction and source/header
  receipts without installing a driver, compiler executable or runtime package.
- Added separate Engram CUDA host/device and completion-adapter syntax checks.
  Reports retain Clang's newer-SDK warning and explicitly distinguish SM90/12.9
  feature compatibility from NVCC 13.2 or B300 qualification.

## Native Engram completion admission

- Added pinned-host device-error readback and CUDA-event completion tracking.
  TP output is admitted only after successful event/error/NCCL observations.
- Replaced the enqueue-only terminal observation with waiting/completed states;
  timeout, asynchronous errors and nonzero device flags reject output. CUDA SDK
  compilation and hardware execution remain unverified.

## Native tensor-parallel Engram stage scheduling

- Added full-chain preflight and a move-only deadline-aware operation connecting
  lookup, NCCL reduction and one-time projection/gate submission.
- Pending reductions yield without submitting compute; terminal errors prohibit
  replay. Borrowed-resource cleanup and generation-wide supervision remain the
  runtime owner's responsibility. CUDA/NCCL execution is not yet qualified.

## Independent native Engram NCCL reduction

- Added opt-in native-only BF16 SUM submission with communicator/rank/device
  checks and move-only pending/enqueued/error tracking, separate from the monolith.
- Documented borrowed-resource ownership and the required polling/completion/abort
  protocol. NCCL compilation and distributed execution are not yet qualified;
  transport-plugin registration and supervised model integration remain open.

## Native V4.1 Engram FP8 projection

- Added block-32 activation quantization and a scale-corrected tiled CUDA GEMM,
  producing BF16 KV for the gate primitive with caller-owned scratch buffers.
- Added exact projection buffer and alias admission. This implementation has
  not undergone CUDA compilation, numerical comparison or throughput qualification;
  TP collectives, optimized kernels and model integration remain outstanding.

## Native V4.1 Engram CUDA primitives

- Implemented rank-local FP8/E8M0 embedding lookup and BF16 gated residual kernels,
  explicit stream launches, strict buffer layout/alias checks and device errors.
- Added an independent CUDA build target with no Python or monolithic-model link.
  CUDA compilation/numerics and end-to-end execution remain unverified; projection
  and collective integration are still required.

## Native V4.1 Engram weight layout

- Added six-member dtype/shape/byte admission, replicated projection/gate layout,
  TP table ownership and valid-versus-padded row handling.
- Corrected the unfinished-work description: the pinned Engram has no convolution.
  Layout admission does not authenticate weights or enable model execution.

## Native V4.1 Engram hashing

- Added admitted compressed-map handling, fixed prime buckets/multipliers and
  sequence-local rolling n-gram hashing with image barriers and chunk continuity.
- Added bounded contiguous append/reset APIs and compile-time product checks.
  Static prime ledgers match the reference configuration; no model execution.
- Native tokenizer normalization, Engram weight operations and B300 integration
  remain outstanding; this component does not register an inference backend.

## Native V4.1 configuration and attention-sharing metadata

- Added a separate CPU-native parser for the complete frozen nested HF config,
  with bounded JSON, exact semantic field/type matching and raw config digest.
- Added 43-layer KV/index ownership, candidate/Engram participation and backbone
  versus MTP expert metadata, derived from the pinned reference configuration.
- Static reference-config comparison and 40-unit syntax checks passed. No model
  capability, B300 execution or numerical qualification is claimed.

## Native text startup from an activated store

- Added mutually exclusive direct-artifact/store selection to serve-deepseek.
  Store mode invokes native full target resolution and pins the returned artifact
  root in the Worker Lock; resolution failures do not fall back.
- Added isolated CPU-tool builds, strict observation checks, directory overlap
  guards and bilingual startup guidance. Worker storage admission is unchanged.

## Receipt-bound active generation resolution

- Joined externally pinned pointer/catalog roots, sealed receipt metadata and
  actual target artifact bytes in a shared-lock read-only resolver.
- Added native/deployment resolve commands; no source checkpoint is required,
  and output explicitly does not claim source re-scan or immutable admission.

## Native stored receipt admission

- Added bounded canonical parsing of stored generation receipts against external
  artifact/receipt roots, with exact member ordering and typed/body hash checks.
- Receipt encoding now reparses its result. This enables later pointer-to-artifact
  consumers without treating receipt metadata as actual weight/storage admission.

## Native atomic generation activation

- Added fresh source-bound/receipt verification, expected predecessor checks and
  atomic current-pointer installation with explicit partial-failure outcomes.
- Serialized native commit/activation with a shared nonblocking advisory store
  lock. External mutation exclusion and immutable admission remain separate.
- Added native/deployment activation commands and bilingual operating guidance;
  catalog admission, Worker reload and runtime qualification are not implied.

## Native activation pointer metadata and reading

- Added canonical pointer encoding/parsing with the existing typed root and
  validated predecessor/strict-next-ordinal rules, bounded to INT64_MAX.
- Added bounded no-follow current-pointer reading through an admitted store,
  with file/path/store revalidation. Atomic activation is not yet connected.

## Native verified generation commit

- Connected source-bound pre/post verification, no-replace promotion, read-only
  sealing and exclusive durable receipt creation into a native commit command.
- Preserved partial-mutation outcome flags and retained files on failure; no
  implicit rollback, overwrite, retry or activation.
- Added deployment dispatch and bilingual staged preparation/commit guidance.

## Native generation store structure

- Added retained-descriptor store admission/revalidation and exclusive store
  initialization matching the frozen canonical manifest and directory ABI.
- Added native and deployment commands with explicit structure-only output,
  partial-initialization retention and no automatic cleanup or activation.
- Added the executable to CPU CI build targets and documented both languages;
  real store operations and model tests have not run locally.

## Native generation receipt metadata

- Added bounded canonical receipt encoding with the existing 11-field typed
  root, complete 47-member PP1 ledger and source-bound projection digest.
- Added exact receipt validation by rebuilding from a fresh source-bound
  observation. This metadata component does not publish or activate a directory.
- Included the new source in the CPU artifact library and syntax-check scope.

## Native source-bound verification projection

- Added canonical existing-format verification JSON and its SHA-256 to
  successful source-bound observations; target-only observations cannot emit it.
- Added native projection output and deployment dispatch, with exact-byte
  capture guidance. Projection generation performs no publication or activation.

## Standalone native source-bound observation

- Exposed source-bound verification of existing generations through the native
  verifier and deployment wrapper, without copying or modifying weights.
- Reject partial authority arguments instead of falling back to target-only
  verification; expose byte-equivalence scope explicitly in observation JSON.
- Added bilingual invocation, trust, read-cost and non-publication guidance.

## Native source-bound generation verification

- Added read-only source-bound verification that rebuilds source authority and
  target metadata, checks the exact member set and canonical headers, and hashes
  every target tensor against its freshly scanned source payload.
- Connected preparation's final verification to this stronger entry point and
  added an explicit successful byte-equivalence observation field.
- Publication receipts, activation and immutable admission remain separate;
  shared planning algorithms are not an independent numerical oracle.

## Native preparation preflight and documentation consistency

- Validate staging before the CLI reads the full source checkpoint and before
  owner-bound metadata compilation; retain checks immediately before writes.
- Make source directory traversal descriptors exception-safe.
- Update offline component documentation to distinguish the implemented
  preparation CLI from unfinished publication, activation and qualification.

## Native PP1 preparation command

- Added `pih-deepseek-artifact-prepare` joining admitted source opening, native
  preparation and final full output verification without automatic publication.
- Added deployment build/run dispatch and bilingual commands, trust-root,
  resource and retained-partial-output guidance. Python only orchestrates.
- Added the executable to Linux CI build targets (workflow not run locally).
  32-unit syntax observation passed; real conversion and model tests not run.

## Complete native source inventory aggregation

- Added grammar, namespace-summary and global inventory roots with frozen
  per-namespace ledgers and pinned grammar-root verification.
- Owner-bound preparation now derives its complete source plan internally and
  compares semantic/inventory expectations; caller-authored per-tensor plans
  are removed from that entry point.
- Syntax-only local validation; real checkpoint conversion, complete CLI and
  publication/activation qualification remain outstanding.

## Native source inventory records and shards

- Added source tensor-record and shard-inventory compilation from actual source
  semantics, preserving typed fields, offsets, role suffixes and record order.
- Produces the payload scanner's per-shard authority inputs from retained source
  descriptors and observed roots. Global grammar/namespace/inventory aggregation
  remains unfinished.
- Offline syntax coverage is 31 units; no checkpoint or model tests executed.

## Native source semantic closure

- Added tensor/set/shard/dtype/overall semantic hashing from actual source
  headers, including header-prefix hashes and frozen dtype-byte accounting.
- Owner-bound preparation now recomputes and compares the semantic root rather
  than accepting only a supplied nonzero value. Source inventory compilation
  and the end-to-end CLI remain unfinished.
- Offline syntax coverage is 30 units; no real checkpoint/model tests were run.

## Complete source tensor geometry rules

- Added MTP common and stage-specific checkpoint dtype/shape checks, rejecting
  malformed stages, misplaced special tensors and main-layer-only additions.
- Applied full source geometry validation to header reads, payload observation
  and disposition records, including excluded MTP weights.
- This validates storage, not DSpark execution. Runtime shared-expert migration,
  typed semantic/inventory roots and real checkpoint qualification remain open.

## Native source header/index closure

- Added retained-descriptor header reads joined against the hashed source index,
  with complete tensor coverage, duplicate/placement and count/byte checks.
- Applies main-model geometry rules and revalidates source identities around the
  scan. The owner-bound conversion entry now requires this header/index check.
- MTP geometry and semantic/inventory root construction remain unfinished.
  No real source scan or model tests were executed; validation is syntax-only.

## Native source artifact byte admission

- Added retained no-follow source opening, role-bounded streaming hashes and
  typed object/model-root comparison against an external trusted expectation.
- Validates config and complete source index, rejects inode aliases and exposes
  descriptor/path revalidation. The owner-bound preparation entry checks source
  plan objects against actually hashed identities before conversion.
- Semantic/inventory admission and runnable conversion CLI remain unfinished.
  Syntax-only validation; no actual source checkpoint or model tests executed.

## Native admitted-source preparation orchestration

- Connected full source-payload observation, trusted payload-root comparison,
  disposition derivation, shard copying and metadata writes in one native API.
- Requires empty controlled staging and an exact distinct source descriptor
  set, joins every shard to it, and checks all sources across the whole operation.
- This does not admit source provenance or expose a complete conversion CLI;
  publication/activation remain separate. Offline syntax coverage is 28 units,
  with no real checkpoint conversion or model tests run.

## Native PP1 disposition aggregation

- Added native namespace, owner-summary, owner and overall disposition hashing
  from freshly rebuilt complete source disposition records, including typed
  empty sets and all MTP namespaces in the PP1 owner projection.
- Returns the layout authority and joined per-tensor arrays required by native
  metadata generation. No caller-provided aggregate owner/disposition hashes
  are needed by this entry point.
- Source inventory/payload antecedent admission and the end-to-end CLI remain
  unfinished. Local checks remain syntax-only; no model tests were executed.

## Native PP1 disposition records

- Added complete source-to-target selection records with typed logical and
  disposition roots, plus explicit roots for all DSpark-disabled MTP exclusions.
- Checks full source names, shape/byte ledgers, selected/excluded count and byte
  totals, and selected target geometry. Outputs join the native layout/runtime
  metadata assembler without caller-authored per-target logical roots.
- Aggregate disposition/owner roots and provenance admission remain unfinished.
  Offline syntax coverage is 27 translation units; no model tests were run.

## Complete native source-payload aggregation

- Added fixed 48-shard aggregation of observed payload records into typed
  record-set, shard and closure roots, preserving the existing format.
- Requires the exact 72,317-name source inventory and byte ledger, compares
  actual header dtype/shape/ranges with source authorities, rejects aliased
  shard identities and checks source metadata across the full scan.
- Provenance/semantic/inventory antecedent admission remains unfinished;
  syntax-only checks do not establish checkpoint or model execution correctness.

## Native source payload observation

- Added actual-header range discovery, ordered authority joins and streaming
  per-tensor payload hashing with typed source-payload record roots.
- Corrected source-shard grammar to 48 files and changed generation copying to
  accept the explicit complete 50-descriptor source set instead of counting only
  descriptors used by selected target tensors. All tensor descriptors must join
  that set; all retained source descriptors get final metadata checks.
- Source provenance and aggregate closure admission remain unfinished. No source
  checkpoint or model tests were executed; this is syntax-checked code only.

## Native generation read-only sealing

- Added post-move permission sealing before final full verification: all 47
  members are admitted as distinct owned regular files with one link each,
  then synced at mode 0444; the generation directory is synced at mode 0555.
- Reports sealing completion independently from rename and overall status.
  Partial permission changes are retained on failure; no automatic rollback.
- Permission sealing is not immutable admission and cannot revoke existing
  writable descriptors. Publication receipts and activation remain unfinished;
  validation is syntax-only and no real model directories were modified.

## Verified content-addressed promotion primitive

- Added PP1 full verification before and after Linux no-replace directory
  promotion, exact member-set checks and parent-directory synchronization.
- Reports rename and durability state separately from errors so post-move
  failures cannot be mistaken for untouched staging. No overwrite fallback,
  cleanup, rollback, publication receipt or catalog activation is performed.
- Requires caller-controlled same-filesystem directories. This is not immutable
  admission or complete store publication; local validation remains syntax-only.

## Reusable native generation verification

- Moved full offline PP1 verification from the CLI into the native metadata
  library, retaining file hashes, inventory/geometry and final identity checks.
- Added structured observations and optional progress callbacks; the CLI keeps
  its existing arguments, exit behavior and JSON schema. No immutable-admission
  or publication claim is introduced. Embedded NUL directory components fail.
- Expanded offline syntax coverage to 24 translation units. Linux link/runtime
  validation and the atomic publication integration remain outstanding.

## Native staging metadata writes (publication incomplete)

- Added exclusive metadata writes with bounded input, digest verification,
  fsync, streaming readback and final file/path identity checks.
- Added PP1 generation metadata writing in index/records/manifest order after
  native parser admission and index-to-record joins. Failure retains staging
  members; no overwrite, cleanup or publication occurs.
- Final all-member verification and atomic publication remain unfinished.
  Local validation is syntax-only; no copy or model tests were run.

## Native generation shard copying (publication incomplete)

- Connected complete PP1 metadata planning to sequential copying of all 44
  shards. Artifact object hashes now derive from actual successful copier
  receipts rather than caller-supplied output hashes in this path.
- Validates all ranges and the frozen 50-source-descriptor ledger before writes;
  rejects duplicate source inode aliases and checks source metadata across the
  entire operation. Failure retains staging files without a generation receipt.
- Returns canonical metadata in memory; metadata writes, final all-member
  verification and atomic publication remain open. Syntax-only validation;
  no runtime copy or model tests were executed.

## Native artifact-manifest encoder (publication incomplete)

- Added canonical native artifact-manifest serialization, sharing all derived
  object/conversion/artifact root functions with runtime parsing. Output is
  reparsed and checked against complete family count/byte/shard geometry.
- Requires explicit resource observations matching the frozen converter
  contract; serialization is not resource, payload or provenance evidence.
- Added synthetic-ledger round-trip/rejection test sources, not executed.
  Connecting admitted copy receipts and publishing generations remains open.

## Native generation metadata assembly (converter incomplete)

- Added a complete-inventory PP1 assembler connecting native byte/layout hashes
  to runtime-record serialization. It generates offsets, storage semantics and
  offline ownership, joins named authority arrays, and verifies the encoded
  count/byte/ownership ledger before returning layout and sidecar together.
- Factored shared runtime metadata parsers into a CPU-only converter library
  used by the offline verifier. No Python inference fallback was introduced.
- Expanded offline syntax checks to 22 translation units; these passed, but
  Linux linking, execution and model tests were not run. Source/disposition
  admission, artifact-manifest production and final publication remain open.

## Native typed layout hashing (converter incomplete)

- Added native per-tensor, shard and complete PP1 generation layout binding,
  deriving record/set, weight-map, index, logical-layout, owner-projection and
  final layout roots from rebuilt geometry and caller-admitted authority roots.
- Preserves the existing typed field format, numeric shard ordering and byte
  ledgers. Complete generation binding enforces the frozen 67,612-tensor geometry;
  no source provenance or disposition admission is inferred from supplied hashes.
- Offline C++ syntax checks only; no runtime or model tests. Source/disposition
  production, complete conversion CLI and generation publication remain open.

## Native runtime-record serialization (converter incomplete)

- Added a native encoder for canonical runtime-record sidecars, sharing typed
  record/chunk/set/root hashing with runtime admission and reparsing its output.
- Recomputes derived hashes and byte ledgers; validates record order, storage,
  owner, shape, supplied nonzero roots, integer bounds and nonoverlapping ranges.
- Added frozen-fixture equivalence and rejection test sources (not executed).
  Source provenance, layout/disposition production, complete conversion CLI and
  generation publication remain unfinished; encoding is not admission evidence.

## Native backend closure checks (unqualified)

- Updated the native CUDA-backend source count/hash from 23 to 24 after adding
  the shared-expert adapter; the stale guard would otherwise reject CUDA builds.
- Added `check_native_syntax.py --deepseek-backend`, checking all 24 native PP1
  backend adapters against the pinned CMake source closure without CUDA headers.
  Standalone/combined selections cover 44/205 distinct C++ translation units.
- This remains syntax-only evidence, not a CUDA build, link or model test.
- The wider check caught and fixed a malformed native attention-operation
  factory preprocessor branch that omitted its parameter list/body boundary.
- Added a native-preview CI syntax job with the pinned header setup and combined
  model/backend selection. The workflow has not been run in this local session.

## Shared-expert bootstrap assembly (unqualified)

- CUDA rank bootstrap now creates the mandatory plugin-capability adapter,
  binds shared weights for every owned main layer, and owns drivers/pinned error
  storage through rank infrastructure. Both host-spill and resident lane sets
  bind that provider into the canonical stack before inference.
- MoE preparation uses the mHC layer input, existing expert source/accumulator,
  and mHC FFN branch output; routed and shared work run on the same rank stream.
- Fixed native rank-runtime compilation: explicit transfer-owner conversion and
  SDK context-scheduling constant with a CUDA-backend static assertion.
- Expanded syntax coverage to rank runtime, bootstrap and shared adapter (184
  integration/model translation units). CUDA linking and model execution remain
  unverified; this is not a model correctness or first-release completion claim.

## Main MoE preparation (bootstrap incomplete)

- Added mandatory preparation before routed submission: D2D-copy the mHC layer
  input into routed/shared source storage and zero the FP32 accumulator for the
  actual packed token count, using the same Backend stream capability.
- Reject null/overlapping/overflowing address ranges and out-of-bound token
  counts; preparation failure poisons the owner/stage and prevents routed work.
- Shared computation consumes the same copied input as routed experts. The
  enclosing runtime still must retain/drain resources after partial submission.
- Removed the shared-expert adapter's exclusion from the native CUDA backend
  source set now that it uses mandatory plugin capabilities instead of direct CUDA.
- Added preparation-order/failure source fixtures without running model tests.
  CUDA bootstrap construction/binding remains unfinished.

## Shared-expert resource ownership and stack insertion (bootstrap incomplete)

- Added heap-stable ownership of per-layer shared drivers and one pinned error
  buffer, binding raw resident weights for the owned layer range. Resolution
  rejects busy/poisoned siblings sharing the same scratch and completion event.
- Inserted the shared-expert wrapper into the canonical compute stack. Inference
  without a bound provider is rejected before launch; drain remains available.
  Provider binding is startup-only and cannot replace an already bound provider.
- The CUDA bootstrap still needs to create and bind this resource owner and its
  operation adapter. Inference is deliberately unavailable until that assembly
  is implemented; old routed-only completion is not preserved as a fallback.

## Shared-expert asynchronous completion (integration incomplete)

- Shared driver submission now queues error D2H followed by a completion event;
  polling reads the pinned error only after event completion. In-flight reuse,
  idle polling and reuse after submission/device/event failure are rejected.
- Added a stage wrapper that waits for routed completion, then shared compute
  and finalization, before allowing the enclosing mHC branch to finish.
- Backend async capabilities are mandatory for host validation, error copy and
  event record/query. No CUDA or synchronous fallback is used.
- Added source fixtures for pending work, device errors, submission failures and
  stage ordering. Tests were not executed. Production resource/provider and
  compute-stack assembly still need wiring; this is not an end-to-end claim.

## Shared-expert raw weight binding (integration incomplete)

- Added main-layer shared-expert binding for six raw E4M3/UE8M0 tensors,
  validating exact matrix/scale geometry, contiguous strides, CUDA device and
  allocation generation; packed routed-expert FP4 is not accepted here.
- Exposed shared-expert views over the existing rank expert compute allocation.
  Reuse requires completion of every routed expert; it adds no allocation and
  does not itself schedule shared computation or establish asynchronous safety.
- Converted the shared-expert NVIDIA adapter to mandatory Backend async and
  Kernel Pack capabilities, deleting direct CUDA context/memset and kernel
  fallback paths. Its source passes standalone Linux-target syntax checking;
  production ownership and scheduling integration are still pending.
- Added source fixtures for missing scales, wrong formats/shapes and foreign
  device/generation. Model tests were not run. The MoE backend still needs shared
  execution, finalization and completion/error polling before layer success.

## Native indexer FP8 projection (unqualified)

- Bound raw E4M3 query weights and UE8M0 block scales instead of requiring
  nonexistent BF16 indexer query weights.
- Added separately owned FP8 activation/scale scratch, activation quantization
  and FP8 GEMM before RoPE; rejected base-address aliasing of writable views.
- Updated model/Kernel Pack request layouts and source fixtures together.
  Rebuild both plugins and regenerate the Lock; no old-layout compatibility.
- CUDA compilation and numerical model execution remain unverified. Shared
  expert integration and the broader first-release work remain unfinished.

## Native compressor raw-weight consumption (unqualified)

- Replaced fabricated fused compressor tensor names with the checkpoint's
  separate BF16 wkv/wgate bindings for main and indexer compression.
- Updated the state assembler, launch validation, native adapter, Kernel Pack
  request and CUDA kernel to consume two independent matrices. Updated affected
  test sources but did not execute tests. CUDA compilation remains unverified.
- This changes the development request structure size; rebuild model and pack
  together and regenerate Locks. Mixed old/new requests fail the exact-size
  check; no fused-layout compatibility bridge is retained.
- Indexer FP8 projection and shared-expert consumer alignment remain unfinished.

## Complete PP1 checkpoint geometry rules (unqualified)

- Added the remaining 612 non-routed tensor rules using pinned official source
  and repository artifact contracts, including FP8 shared experts, BF16 raw
  compressor projections, indexer tensors and gate auxiliary storage.
- All 67,612 PP1 names now have dtype/shape checks in planning and verification.
  Static dimensional accounting matches the fixed 156,015,698,140-byte ledger;
  21 source files pass syntax checks. Real artifact/model validation is pending.
- Runtime fused/dequantized consumer alignment, source provenance and complete
  conversion/publication remain unfinished.

## Native dense checkpoint rules (development)

- Added exact dtype/shape rules from existing native endpoint, mHC, attention
  and router-weight bindings to PP1 planning and artifact verification.
- Covers 952 additional names alongside 66,048 routed-expert names. Remaining
  compressor/indexer, shared-expert and gate auxiliary rules are unfinished;
  raw checkpoint layouts are not replaced with runtime-only fused layouts.

## Native routed-expert checkpoint geometry (development)

- Shared exact routed-expert dtype/shape validation between PP1 byte planning
  and offline artifact verification, using the existing native expert-bundle
  checkpoint ABI rather than inferred unpacked FP4 dimensions.
- Together with exact names this checks all 66,048 routed-expert tensors.
  Remaining tensor classes, MTP geometry and provenance are not covered by
  this helper; full source admission and conversion/publication remain open.

## Native frozen tensor-name inventory (development)

- Ported the complete fixed 0731 tensor-name grammar, including the optional
  source MTP names. Native PP1 planning and artifact verification now require
  exact sorted-name equality, not merely matching aggregate tensor counts.
- Expanded artifact syntax coverage to 21 source files. Expected dtype/shape
  and source-provenance admission remain unfinished; MTP name enumeration does
  not add MTP inference or DSpark deployment support.

## Native PP1 generation byte planning (development)

- Added 44-namespace generation planning, fixed PP1 tensor/byte ledgers,
  bounded per-shard grouping and native canonical weight-index generation.
- Retain per-shard source indices/byte offsets and reparse the generated index
  before returning it. Artifact syntax coverage now includes 20 source files.
- This is not source provenance admission or a complete converter: authority
  hashing, runtime-record serialization and generation publication remain open.

## Native single-shard layout (development)

- Implemented canonical PP1 safetensors header/layout construction, namespace
  and shape/byte validation, padded little-endian prefixes, header digest and
  absolute tensor offsets. Reparse generated headers before writing.
- Connected layout construction to the native identity copier and expanded
  artifact syntax tooling to 19 verifier/copy/support source files. The complete
  source admission, layout authority and generation publication are unfinished.

## Native identity payload copy (development)

- Added a descriptor-based native identity-copy component for the DeepSeek
  converter: bounded range streaming, per-payload SHA-256 verification,
  exclusive output creation, durable sync and independent output readback.
- Reject changed source/output metadata and retain failed partial members
  without returning receipts, overwriting files or publishing a generation.
- Component passes Linux-target C++20 syntax checking and is included in the
  Linux artifact CI build. Runtime behavior remains untested; native layout
  generation and end-to-end conversion/publication are not yet connected.

## Native offline artifact verification (unqualified)

- Added CPU-only `pih-deepseek-artifact-verify` and
  `deploy/native.py verify-deepseek-artifact --build`, independent of the old
  Python runtime, monolithic archive, CUDA and ICU.
- Verify externally rooted PP1 manifests, complete file hashes, index/record
  bindings, ownership and real safetensors headers through no-follow file
  descriptors. Detect ordinary concurrent mutation and report an observation,
  not an immutable storage admission or model qualification receipt.
- All 17 tool/support source files pass Linux-target syntax checks. Local
  cross-CMake configuration lacks Linux OpenSSL; linking and execution remain
  unverified. Added Linux CI compilation and bilingual usage documentation.
- Native weight conversion/publication is still unfinished; verification does
  not substitute for that implementation.

## Native PP1 compilation coverage (development)

- Expanded Linux-target syntax checks to the 161 CMake-selected DeepSeek PP1
  model sources plus 20 integration units; all 181 now pass syntax checking.
- Fixed missing direct transaction and capability-lease includes that broke
  standalone native model compilation with NCCL disabled.
- Save full diagnostic logs and add actual PP1 model-object/support-library
  compilation to Linux CI. CI execution, CUDA linking and model tests remain
  unverified; two existing hash-router size-check warnings remain locally.

## Native DeepSeek text integration (unqualified)

- Registered text ABI v2 in the DeepSeek-0731 model plugin, including pinned
  semantic admission, native generation, incremental reasoning/content, strict
  DSML completion, tool-call identities, cancellation and close ownership.
- Added `serve-deepseek --build` to compose native providers with the text HTTP
  surface, without loading diagnostic surfaces or a Python inference backend.
  Requires externally verified PP1 weights and the pinned semantic snapshot.
- Synchronized manifest/Lock capability inventories and ICU build dependencies.
  Code passes syntax checks; linking, CUDA builds and model qualification remain open.

## Native syntax verification (development)

- Added pinned, hash-verified Linux header preparation and a 19-translation-unit
  Clang syntax checker usable from Windows, without installing packages or
  executing models. Linked/CUDA/runtime qualification remains open.
- Fixed DSpark-disabled headers exposing unavailable provider types and an
  implicitly deleted default move assignment declaration in controller ingress.

## Client distribution cutover (development)

- Root wheel is now pure Python `pih_client`; it no longer builds or packages
  `_pih`, `pih.Engine` or historical offline console scripts. No aliases added.
- Moved standalone host preflight into the client distribution; added bounded
  HTTP client/CLI with TLS verification, no redirects and no automatic retries.
- Native Qwen streaming/non-streaming generation shares packed scheduler events,
  with cooperative deadline/disconnect cancellation and drain. Text ABI v2
  replaces v1 without a bridge; rebuild the full bundle and Lock.
- Added incremental UTF-8 decoding, bounded SSE output and client streaming;
  successful streams end with usage and `[DONE]`, failed streams do not.
- Release zero-event prefill output credits internally, since no public event
  exists for consumers to acknowledge those plans.

## Native DeepSeek token generation (unvalidated)

- Added a plugin-private chunked-prefill/decode scheduler and caller-owned token
  generation C ABI; the diagnostic smoke shares the same scheduler.
- Added Worker token generation and `deploy/native.py generate-deepseek-tokens`
  build/staging/launch orchestration, without Python model execution.
- Updated the DeepSeek manifest, sealed capability inventory, SDK headers and
  bilingual launch guide. Existing DeepSeek bundles/Locks must be rebuilt.
- Extracted shared native BPE support with sequential isolated splits for
  DeepSeek-0731 and a CPU-only `pih-tokenize` tool; no model or Python dependency.
  Special-token decoding remains family-specific. Tokenizer equivalence
  qualification is not yet complete.
- Added a native full-conversation DeepSeek codec and completed-turn DSML parser,
  plus `pih-deepseek-format`; the pinned upstream MIT notice is preserved.
  HTTP integration, target compilation and equivalence qualification remain open.
- Added native pinned semantic-closure admission using no-follow descriptors and
  copied bytes, verified in-memory tokenizer construction, an admission CLI and
  `deploy/native.py verify-deepseek-semantics` build orchestration. No Python
  source is executed; this does not qualify weights or model execution.
- DeepSeek's private generation scheduler now supports committed-token/cancel
  callbacks, checks deadlines after device steps, and drains the current
  committed plan before retiring interrupted requests. Cleanup failure retains
  fail-stop ownership; HTTP callback wiring is still pending.
- Added request-owned incremental DeepSeek text/reasoning decoding with partial
  marker retention and strict final DSML validation. Length-truncated tool
  blocks are never emitted as calls; the format CLI exposes a `decode` mode.
  Native HTTP streaming integration and qualification are still pending.
- Native V4.1/SM103, full DeepSeek text services and repository-wide legacy
  removal remain unfinished. No Linux/CUDA build or model test is claimed.

All notable changes to PIH will be documented in this file.

## [0.1.0-alpha] - Unreleased

First code-and-build delivery of the plugin-first inference harness. Hardware
execution remains unverified and is not claimed by this release.

### Included

- Native Qwen3 text inference and SM89/SM90 Kernel Pack targets, bounded C++
  ByteLevel/BPE tokenizer, native loopback HTTP surface, worker serving lifecycle
  and build/seal/launch/probe scripts. No Python inference bridge; target build
  and execution are not yet validated. Full repository migration remains open.

- B300 SM103 host preflight for 1–8 devices, with replayable diagnostic receipts.
- Optional DeepSeek V4.1 Flash CUDA reference package: pinned MIT upstream
  model/kernels, TP checkpoint conversion/loading, sampling, vision, CLI and
  serial HTTP surface. B300 execution and tests remain unqualified/deferred.
- DeepSeek V4.1 Flash configuration inspection and explicit rejection of
  unimplemented V4.1/B300 native runtime profiles.

- Installable header-only plugin SDK and standalone external plugin example.
- Native worker/external-plugin and CPU preview container CI definitions.
- Bilingual public guides and documentation link checks.
- CPU wheel build and isolated import check in the CPU workflow.

### Fixed

- Select the custom deployment profile for the transitional CPU wheel and CI.
- Align the DeepSeek model source-list digest with CMake's bytewise ordering.

### Existing preview components

- Native worker and capability-driven plugin loading.
- Lock-selected deployment graphs and startup validation.
- Platform, storage, backend, execution, memory, model, surface and Kernel Pack
  plugin boundaries.
- Dependency-light Phase 1 native build.
- Linux CPU contract workflow.
- Paired English/Chinese public documentation with an automated documentation
  integrity gate.
- Qwen3 RTX 4090 D SM89 and H100 SM90 native model bundles, including BF16 and
  PIH INT4 artifact paths.
- DeepSeek V4 Flash-0731 RTX 4090 D PP1 native text bundle and offline artifact
  tools.
- DeepSeek V4.1 B300 SM103 native rank Worker, supervisor, NCCL transport and
  offline conversion tools for 2/4/8 ranks.
- Explicit native release components, installation receipts, source/native
  inventories, client-only wheel/sdist and bilingual deployment documentation.

### Support boundary

The production source targets compile, finally link and generate their expected
device images. No GPU/model deployment profile is hardware-qualified in this
release. Real weights, Linux service execution, numerical correctness, capacity,
multi-GPU behavior and performance remain unverified.
