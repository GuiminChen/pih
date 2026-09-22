# Native DeepSeek artifact tools

## Command entry points and current boundary

The Linux CPU-only `pih-deepseek-artifact-prepare` executable admits the source,
derives the complete PP1 conversion plan, writes staging and independently
verifies every output member. `deploy/native.py prepare-deepseek-artifact`
provides build/run orchestration, not Python inference. See the
[deployment guide](../../docs/native-inference.md) for command arguments,
trusted-root requirements and disk-space planning.

The native command validates empty, owned staging before hashing the checkpoint;
the preparation library repeats this check before metadata compilation and
before writes. These checks are not a directory lock. The caller must exclude
concurrent writers. Failure retains any created files; no automatic cleanup.

Preparation is not publication: the separate commit/activation commands below
handle receipts and store pointers. Catalog-byte admission and immutable Worker
storage integration remain unfinished. The code has only local syntax validation, not Linux linked execution,
real checkpoint conversion or model qualification.

## Native components

`ObserveActiveGeneration` joins an externally pinned pointer/catalog pair to
the sealed stored receipt and actual target files. It holds a shared advisory
store lock during verification, checks receipt/path identity across the complete
operation and rereads the pointer before returning. `VerifyReceiptBoundGeneration`
parses the receipt against expected roots, joins every manifest/index/records/
shard hash and byte count, checks conversion authority, exact membership and
sealed ownership/modes, then performs full target verification with retained
descriptors. It does not reread the original source checkpoint. The native store
`resolve` command exposes metadata with explicit receipt-bound/non-immutable
scope. Returned metadata is not a retained lease; Worker still needs its own
artifact/storage admission. `serve-deepseek --store-dir` now uses this result to
select a pinned artifact for the native text Worker; this is startup selection,
not hot reload or a replacement for runtime admission. Catalog-byte admission
and Linux/CUDA qualification remain open.

`ParseGenerationReceipt` admits the stored metadata format against independently
expected artifact and receipt roots without requiring the original checkpoint.
It enforces bounded canonical JSON, the exact 17 fields, PP1 geometry, ordered
47-member names, per-role/aggregate byte budgets, canonical nonzero hashes,
the receipt-body SHA-256 and the existing typed receipt root. Encoding reparses
its own result through this path. This parser reads no generation files and
does not itself prove source equivalence or immutable storage. Worker integration
must still join the pointer, receipt and actual admitted artifact bytes; a parsed
receipt alone is not a capability or a replacement for storage admission.

`ActivateGeneration` connects fresh source-bound verification, sealed receipt
readback, expected predecessor and strict-next-ordinal checks to atomic pointer
installation. Commit and activation both take a nonblocking advisory flock on
the store root; callers must still exclude noncooperating writers. Activation
retains metadata descriptors, exclusively writes/rereads a temporary pointer,
uses no-replace for the first pointer or atomic replacement for a validated
existing pointer, synchronizes directories and rereads current.json. Outcomes
distinguish temporary creation, replacement, sync and completed activation.
Failed operations retain artifacts; no rollback/cleanup is automatic. The native
store command and deployment wrapper expose `activate`. Catalog roots must be
independently admitted: this layer does not inspect catalog bytes, reload Worker
or grant an immutable lease. Real activation/crash recovery remains unqualified.

`generation_pointer.h` implements canonical activation-pointer encoding/parsing
with the existing seven-field typed root. Artifact, receipt and catalog roots
must be nonzero; a null predecessor is allowed exactly at ordinal 1. Successor
construction reparses/rebuilds the previous pointer and requires exactly the
next ordinal, rejecting overflow. The native integer JSON representation bounds
ordinals to INT64_MAX. `ReadCurrentGenerationPointer` reads at most 64 KiB through
the admitted store descriptor, checks ownership/kind/link count/filesystem and
file/path mutation, and revalidates the store. Absence returns an empty optional;
malformed or replaced pointers fail. Neither the codec nor reader verifies the
pointed-to generation/receipt/catalog, changes current.json or grants an immutable
lease. `ActivateGeneration` supplies the separate filesystem coordinator. Syntax only;
no pointer operations or model tests have run locally.

`CommitGeneration` joins the store, source-bound verifier, promotion primitive
and receipt encoder/writer. It accepts only a direct staging member, retains its
descriptor and compares directory identities before/after promotion. Fresh
source-bound scans on both sides of the move must produce identical projections.
Only then is the receipt exclusively created, reread, sealed 0444 and synced.
Final source/store revalidation must succeed before `receipt_committed=true`.
Errors retain promotion flags and a conservative `receipt_may_exist`; no rollback,
overwrite or recursive cleanup occurs. The native store command exposes `commit`
and reports outcome flags even on post-mutation failure. It does not update the
activation pointer or grant an immutable lease. Caller-exclusive mutation control
remains required. Store-pointer activation is separate; catalog/Worker integration remains unfinished.
No real commit, fault-injection or model execution has run locally.

`generation_store.h` owns retained descriptors for a frozen store ABI. Opening
checks canonical `store.json`, the exact five root members, no-follow path
components, caller ownership, non-group/other-writable modes, distinct directory
inodes on one filesystem and bounded child name/type tables. Revalidation checks
paths/descriptors again and rejects ordinary mutation during enumeration/read.
It does not inspect generation payloads or receipt/pointer bodies. Callers must
exclude concurrent writers; this object is not an immutable lease or lock.
Initialization exclusively creates a previously absent root, its four 0700
subdirectories and 0600 manifest, synchronizes files/directories/parent and
reopens for validation. Failure retains partial files without automatic cleanup.
`pih-deepseek-generation-store init|verify ABSOLUTE_STORE_DIRECTORY` exposes the
operations; `deploy/native.py deepseek-generation-store` builds/runs the tool.
No actual store operations or model execution have run locally; syntax only.

`generation_receipt.h` provides native metadata-only receipt encoding and exact
validation against a fresh source-bound observation. It preserves the existing
receipt schema and 11-field typed root: verification-projection digest, artifact,
conversion/layout/disposition/converter roots, 47-member object ledger, canonical
body digest and publication ABI/support state. The encoder requires canonical
projection bytes with a matching SHA-256, complete PP1 counts, fixed shard order,
bounded sizes and nonzero canonical digests. `ValidateGenerationReceipt` rebuilds
the expected bytes and rejects any difference, including noncanonical spelling.
Callers must obtain their own fresh native verification result; a writable C++
observation is not a capability and cannot authenticate caller-supplied data.
Encoding performs no filesystem mutation and does not mean publication occurred.
The future publication coordinator must join verification, promotion, receipt
writing and activation. Only syntax validation has run for this component.

`--source-bound-projection` on the native verifier takes the same arguments as
`--source-bound` and emits the existing
`pih.deepseek_v4_flash_0731_runtime_artifact_verification.v1` projection. It is
canonical ASCII JSON without a trailing newline, constructed from the retained
manifest only after source-derived metadata equality, actual payload comparisons
and final target identity checks pass. Source revalidation must also succeed
before the result escapes the public API. `GenerationObservation` carries the
exact bytes and SHA-256; target-only results leave these empty/zero. This output
preserves the existing non-authorizing scope and is not a publication receipt.
The wrapper's `--verification-projection` selects this mode but prints its own
logs, so use the native executable directly for canonical stdout capture.

The standalone verifier exposes source-bound observation without conversion:
`pih-deepseek-artifact-verify --source-bound SOURCE_DIRECTORY GENERATION_DIRECTORY MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT`.
All roots must be nonzero and independently trusted. The deployment wrapper
selects this mode with `verify-deepseek-artifact --source-dir` plus all five
source/converter roots; partial authorities fail. Output explicitly distinguishes
`source_payload_equivalence=true` from target-only `false`. It does not write,
promote, activate or emit a publication receipt. Full source and target reads
can take substantial time; source MTP tensors are admitted but not executed.

`VerifySourceBoundGeneration` adds source-to-target byte-equivalence observation
to the target-only verifier. It rebuilds semantics/inventory and scans source
payloads against independently supplied roots, derives disposition and metadata,
compares exact index/records/manifest bytes and canonical shard headers, then
hashes every target tensor against its freshly observed source payload digest.
The target directory must contain exactly the 47 expected members, checked before
and after reading. Source and target descriptors/paths receive final mutation
checks. The preparation CLI now calls this entry after copying; the standalone
target-only verifier retains its narrower scope. Shared planning code is reused,
so this is not an independent implementation of those algorithms. It adds full
read passes, does not issue a publication receipt and grants no immutable lease.
No real checkpoint or model execution has been used to qualify this code locally.

`CompileSourceInventory` now completes the native name-grammar, namespace and
overall inventory roots on top of freshly derived shard records. It checks every
namespace's frozen count/byte ledger and compares the computed name grammar to
the pinned `cebbc15b...fb9364f` root. Namespace order is endpoint, numeric main
layers, numeric MTP stages; per-namespace records use lexical tensor ordering.
The owner-bound preparation API now derives this full plan internally and
compares the expected semantic/inventory roots. It no longer accepts a caller's
per-tensor/shard plan. External expected roots still require a trusted origin;
no real checkpoint conversion or numerical model qualification is claimed.

`CompileSourceInventoryShards` now derives source tensor records and per-shard
inventory roots from freshly compiled actual source semantics. It binds each
tensor's namespace, role suffix, dtype/shape, relative and absolute offsets to
its semantic root; record sets retain lexical tensor order. The result uses the
payload scanner's input structure with actual retained descriptors and hashed
object roots. Callers no longer need to author these per-tensor/per-shard roots.
`CompileSourceInventory` adds the name-grammar, namespace-summary and overall
inventory roots; this lower-level shard result alone is not that authority.

`CompileSourceSemantics` derives native tensor, tensor-set, shard, dtype-summary
and overall semantic roots from retained source headers, preserving the existing
typed hash format. It hashes the actual header prefix including the length word,
uses lexical tensor/dtype ordering and numeric shard ordering, and validates the
frozen six-dtype byte ledger. Source paths/descriptors are revalidated at the end.
The owner-bound preparation entry recomputes and compares this semantic root
with its expected authority before payload scanning or output writes. Inventory
construction builds on these semantics; semantic compilation alone is
not publication, immutable-storage admission or numerical model qualification.

`SourceArtifact::ReadShardHeaders` reads headers from retained source descriptors
and checks every actual tensor name/shard against the already-hashed index.
It rejects duplicates, omissions, foreign names, wrong shard placement and
invalid/empty ranges, checks the complete source count/byte ledger, and applies
full source dtype/shape rules including MTP tensors. File and path identities
are revalidated before and after reading. The owner-bound preparation entry
requires this check before payload scanning or writes. Typed semantic/inventory
roots are compiled by the layers described above; headers are
not an immutable storage guarantee.

`ValidateSourceTensorGeometry` includes common MTP attention/mHC/router/expert
storage and the ten stage-specific boundary/head tensors. It rejects MTP-only
weights in the wrong stage, malformed stage numbers and main-only compressor,
indexer or token-id hash-router tensors. Shared experts retain FP8 geometry;
routed experts retain packed FP4 geometry. The same validation is required by
source header reads, payload scans and disposition compilation, including the
MTP tensors explicitly excluded from PP1. This is not DSpark execution support;
its runtime shared-expert consumers still need corresponding native migration.

`SourceArtifact::Open` admits config, index and 48 source shards against an
externally trusted model digest. It opens absolute directory components and
members without following links, retains all 50 descriptors, rejects inode
aliases and enforces role/aggregate byte limits. It streams file hashes, derives
the existing typed object/model roots, validates the frozen config and complete
source index names/shard set, then rechecks retained descriptors and paths.
`Revalidate` repeats the ordinary mutation/replacement checks; it is not an
immutable lease. Extra unlisted files are not admitted. The expected digest
must not be obtained from the same untrusted directory merely to make it pass.

The owner-bound `PreparePp1Generation` overload derives the source plan from
this hashed owner and revalidates its paths before and after conversion. It
compares freshly computed semantic/inventory roots to trusted expectations.
The lower-level owner itself proves byte identity, not model execution.

`PreparePp1Generation` is the native orchestration entry point for an already
admitted source plan. It requires 50 unique regular-file descriptors, all 48
shards joined to that set, nonzero model/semantic/inventory/converter roots and
an independently admitted expected payload root. It requires empty owned staging,
scans payloads, compares the closure root, derives disposition/layout, copies
all target shards and writes index/records/manifest. All source descriptors are
compared against their initial metadata after the complete operation. Failures
retain staging files without returning a success result.

This library API is invoked by the preparation CLI. The descriptor-level
entry requires a previously admitted plan; the owner-bound entry generates it.
Expected root values still require a trusted origin. It does not publish,
activate or certify immutable storage. The CLI follows it with final
all-member verification; the promotion/publication protocol remains separate.
No real checkpoint conversion has been performed locally.

`BuildPp1DispositionRecords` derives target logical and selected/excluded
disposition record roots from the complete sorted source payload observations.
PP1 selects 67,612 endpoint/main-layer tensors and explicitly excludes the
4,705 MTP tensors with `dspark_disabled`; neither group is silently dropped.
It checks each shape/byte ledger, nonzero antecedent roots, exact source names,
both frozen aggregate byte totals and selected target geometry. Results expose
selected tensors and matching layout/disposition authority arrays for metadata
assembly, plus the excluded record roots for subsequent aggregate compilation.
`BuildPp1Disposition` additionally rebuilds these records and derives all 47
namespace roots, the single PP1 owner's namespace summary/root and the overall
disposition root. Empty selected/excluded sets receive typed empty-set roots;
MTP namespaces remain included in the owner summary even with zero ownership.
Ordering is endpoint, numeric main layer, numeric MTP stage. Its output includes
the `GenerationLayoutAuthority` needed by the native metadata assembler. Source
roots still need independent admission; this does not qualify DSpark inference.

`ObserveSourcePayloads` reads an admitted borrowed source descriptor's actual
safetensors header, joins name-sorted source-record authorities, and streams each
tensor through a 1 MiB buffer to compute its payload SHA-256 and typed payload
record root. Header order need not be lexical. It validates the frozen 48-shard
filename grammar and returns copy ranges with observed hashes; final descriptor
metadata checks reject ordinary concurrent changes. It does not authenticate
the caller's artifact-object/source-record roots, inspect source path identity,
or produce the complete source inventory/payload closure. Those admission steps
remain required before issuing publication receipts.

`CopyGenerationShards` now takes the complete 50-descriptor admitted source set
explicitly and checks every tensor descriptor belongs to it. The 50-resource
ledger is not a claim that the checkpoint contains 50 weight shards: there are
48 weight shards plus other source files. All admitted descriptors, including
those not referenced by selected PP1 tensors, are checked before/after copying.
Binding those descriptors to official source object identities remains the
source-admission caller's responsibility.

`ObserveSourcePayloadClosure` scans all 48 shards in fixed filename order and
derives per-shard record-set, shard and global payload roots using the existing
typed format. It requires 72,317 exact source names (including MTP) and
166,878,536,440 payload bytes, rejects aliased shard identities and compares
every actual dtype/shape/range against the supplied source-record authority.
All source descriptors receive a final identity/size/time comparison spanning
the complete scan. Returned tensors are globally name-sorted. Payload memory is
streamed; bounded headers and the complete record metadata remain in memory.
Model, semantic and inventory roots are still supplied antecedents requiring
independent admission. This does not implement MTP inference, authenticate
official source objects or grant publication authority. No actual checkpoint
scan has been executed locally.

`pih-deepseek-artifact-verify ABSOLUTE_GENERATION_ROOT EXPECTED_ARTIFACT_ROOT_SHA256`
is a Linux CPU-only, read-only verifier for existing DeepSeek V4 Flash 0731 PP1
generations. Enable `PIH_BUILD_ARTIFACT_TOOLS=ON`; it has no ICU, CUDA, Python
runtime, Worker or monolithic archive dependency.

The expected nonzero root must come from an independently trusted publication,
not be copied from an untrusted manifest just to make it pass. The tool uses the
native runtime manifest parsers, verifies canonical roots and fixed geometry,
hashes every listed file, compares index/record bindings and ownership, and
checks actual safetensors headers against every runtime record. Reads use a
1 MiB hash buffer, bounded metadata and at most the bounded manifest member set.
All file descriptors remain open through the final identity/size/time check.
Path traversal and symbolic links (including directory components) are refused;
files must be regular. Materialize Hub symlinks before using the tool.

Exit 0 emits one JSON observation on stdout; progress/errors go to stderr.
Exit 2 indicates failure and emits no success observation. Interrupting the tool
does not publish anything. Hashing a complete generation can take substantial
time and disk bandwidth. No model is executed and no file is modified.

The observation is not a signature, fs-verity measurement, immutable lease or
permission to skip Worker storage admission. Timestamp/identity checks detect
ordinary concurrent mutation but cannot prove an adversarial filesystem stayed
immutable. Extra files are not admitted or hashed. This tool does not establish
official source provenance, source-to-target conversion equivalence, numeric
correctness, or hardware qualification. It does not convert or publish weights;
preparation is supplied by the separate native command, while full publication
remains unfinished.

## Native identity-copy component

`identity_copy.h` provides `CopyIdentityShard` for the native converter. It takes
borrowed, already-admitted source descriptors, an encoded header and ordered
payload ranges with expected SHA-256 values. It exclusively creates one member
under an open staging-directory descriptor, streams with a 1 MiB buffer, checks
every range hash, synchronizes and independently rereads the output, then checks
source/output metadata and synchronizes the directory. It never replaces files
or publishes a generation. Failed calls retain the partial member and return no
receipt; callers must not admit that member. No automatic recursive cleanup.

Limits are 512 GiB per shard, 72,317 ranges, 1,024 source descriptors and a
16 MiB header plus its 8-byte prefix. Layout/provenance admission and safetensors
header construction remain the caller's responsibility. Metadata checks are not
an immutable storage guarantee. The component is syntax-checked but not runtime
validated. The preparation CLI invokes it through the generation copier.

`shard_layout.h` now supplies the native PP1 single-shard layout step and
`CopyPlannedIdentityShard` connects it to the copier. It accepts strictly ordered
admitted tensors in `endpoint` or `layers.0` through `layers.42`, validates shape
byte counts and namespace membership, generates canonical ASCII JSON with the
`format=pt` metadata, pads to 8 bytes, prepends the little-endian header length,
and reparses the result with the runtime safetensors parser before writing.
Offsets in the returned layout are absolute file offsets, not payload offsets.
Checkpoint storage bits are preserved; packed FP4 payloads retain their source
storage dtype/shape, without numerical reinterpretation. It does not establish
the complete official tensor inventory, disposition correctness, the multi-shard
index or generation publication on its own. `BindShardLayout` additionally
derives typed per-record, weight-map-set and shard-layout roots from rebuilt byte
geometry and caller-admitted per-tensor source/target roots. Name joins and zero
roots are checked; supplied roots are not independently authenticated here.

`generation_layout.h` extends byte planning across all 44 PP1 namespaces, in
endpoint-then-numeric-layer order. It requires strictly sorted unique input,
67,612 tensors and 156,015,698,140 payload bytes, builds each shard through the
single-shard planner, and emits canonical `metadata.total_size`/`weight_map`
index JSON plus its SHA-256. The generated index is parsed again with the native
runtime parser. Per-shard input indices and absolute byte ranges are retained
for the converter. No files are created by this planner. It assumes previously
admitted tensor inputs: aggregate counts are not a substitute for source
provenance validation. Runtime-record serialization is supplied by the native manifest
encoder described below; assembling its admitted inputs is still converter work.

`BindGenerationLayout` binds the complete rebuilt PP1 geometry to caller-admitted
source inventory, payload closure, disposition and single-owner roots. It derives
the global weight-map, index, logical-layout, owner-projection and final layout
roots using the existing typed format (including its non-authorizing scope and
`hardware_evidence_open` state). Shard roots remain endpoint-then-numeric-layer
ordered; returned tensor record roots are restored to input name order. Header
byte accounting excludes the eight-byte length prefix; artifact bytes include
all shard files and the index. It accepts no precomputed mutable layout object.
These functions are syntax-checked, not runtime-validated. They do not produce
or authenticate source/disposition roots, inspect payloads, publish a generation
or connect a complete conversion CLI. Those steps remain unfinished.

`generation_metadata.h` now supplies `BuildGenerationMetadata`, the native
PP1 assembly entry point joining byte/layout planning to runtime-record
serialization. It requires the complete 67,612-tensor inventory and two equally
ordered, name-matched per-tensor authority arrays. Disposition record roots must
be nonzero; layout roots and geometry are rebuilt internally. Absolute ranges,
checkpoint storage semantics, namespaces and pipeline ownership are generated
from that validated layout, not supplied by the caller. Packed routed I8 storage
is kept distinct from FP8 shared weights and UE8M0 scales. The generated records
are encoded/reparsed and their count, byte sum and PP1/DSpark-disabled ownership
ledger checked against the layout before returning both layout and sidecar.
Ownership name classification is confined to this offline compilation step;
Worker admission still uses the verified runtime-record manifest. The component
is built in the CPU-only `pih_deepseek_generation_metadata` library, with no
Python inference dependency. The preparation CLI invokes it through the copier;
this individual component does not write/publish files. Caller-supplied source
and disposition authority roots still need independent admission. Only syntax
checks have run locally, not linked execution or model tests.

`DeepSeekRuntimeArtifactManifest::Encode` supplies canonical artifact-manifest
serialization for complete family count/byte/shard ledgers. It recomputes
converted shard, index-object, runtime-record-object, conversion, manifest body
and artifact roots, then reparses and checks family geometry. Supplied derived
shard/index-object roots are ignored. Resource observations must explicitly match
the frozen 1 MiB / 50 source descriptors / one output descriptor contract.
This metadata encoder does not observe resource use, read or authenticate source
or target files, verify the sidecar's contents, or prove per-tensor completeness.
The generation copier supplies admitted copy receipts; final publication remains
separate unfinished integration work.
Synthetic-ledger round-trip/rejection test sources were added but not executed.

`CopyGenerationShards` in `generation_copy.h` connects complete metadata planning
to sequential copying of all 44 PP1 shard files. Before writing, it checks the
staging directory, all source range bounds, and exactly 50 distinct regular-file
source descriptors with distinct inode identities. Each shard is exclusively
created using the existing payload-hashing, fsync and reread copier. Only its
successful byte/hash receipt feeds the artifact manifest. A final source
size/identity/timestamp comparison spans the whole generation copy, not merely
individual shards. Failure retains staging files without returning a generation
receipt; no existing file is replaced or removed.

The result includes index, runtime-record and artifact-manifest JSON in memory;
metadata writing follows in `WriteGenerationMetadata`. This function does not publish a
directory, admit source provenance, lock sources, or guarantee earlier outputs
remained unchanged after their individual copy completed. Final all-member
verification is connected by the CLI; immutable storage admission and atomic
publication remain separate. Syntax checks cover this code; no copying or model execution has
been performed locally.

`WriteGenerationMetadata` writes the index, runtime-record sidecar and finally
`pih.manifest.json` into staging. Before any metadata write it reparses the
manifest and records, validates PP1 family geometry, checks the index/record
name-to-shard joins and byte ledgers, and verifies the in-memory object hashes.
`WriteMetadataMember` bounds metadata to 128 MiB, exclusively creates each member
without following links, fsyncs, rereads using a 1 MiB buffer, checks file/path
identity and timestamps, then syncs the directory. Failure retains files; no
overwrite or automatic cleanup. This does not rehash earlier shards or metadata
after subsequent writes. A manifest in staging is not a publication receipt:
the CLI invokes the final all-member verifier, but not atomic publication.

`VerifyGeneration` in `generation_verify.h` exposes the existing full verifier
as a native library operation. The CLI delegates to it and retains its command
syntax and observation JSON. Library calls return the observed artifact root,
tensor/byte/shard counts and directory device/inode; they do not print unless a
synchronous progress callback is supplied. All previous member hashes, frozen
inventory/geometry checks, actual header comparisons and final descriptor/path
identity checks remain in the shared implementation. A result is not an
immutable lease and cannot authorize skipping Worker admission. No directory
publication is performed by this interface.

`PromoteVerifiedGeneration` implements the lower-level content-addressed move,
not the complete store-publication protocol. It verifies staging, requires the
exact 47 PP1 members, checks directory identity, syncs staging and uses Linux
`renameat2` with no-replace into `sha256-<artifact-root>` beneath an existing
generations directory. Both parents must be on the same filesystem, owned by
the effective user and not group/other writable. Absolute path components are
opened without symlinks. Existing destinations fail; ordinary overwrite rename
is never a fallback. Both parents are synced and the final directory is fully
verified again (including all payload hashes).

Its outcome separately reports status, whether rename occurred and whether both
parent syncs completed. On a post-rename error, inspect the destination; do not
blindly rerun conversion. Nothing is automatically removed or rolled back.
Callers must exclusively control staging/store mutation; these checks do not
exclude adversarial same-user writes or confer immutable storage admission.
Publication receipts and catalog activation remain unfinished,
as does the end-to-end publication CLI. This primitive has only been syntax-checked locally;
no real artifact directories were moved or model tests run.

Promotion now seals the moved generation before its final full verification.
All 47 members are opened and checked before the first permission change:
regular files, effective-user ownership, distinct inodes, one link per file and
the generation's filesystem. Member permissions become `0444`, the directory
becomes `0555`; each file and directory is synced and its identity/mode checked.
The returned `read_only_sealed` flag means all sealing operations completed,
not that final verification succeeded: always check overall status as well.
On a sealing failure, permissions can be partially changed at the destination.
No rollback or cleanup is attempted. Owners can restore write permissions and
already-open writable descriptors are not revoked, so this is not an immutable
lease, fs-verity, or protection from concurrent same-user writes.

`DeepSeekRuntimeRecordsManifest::Encode` now serializes admitted record geometry
and per-tensor roots into the runtime's canonical sidecar format. It recomputes
record roots, 4,096-record chunk/set roots, body/root/object digests and byte
counts, then reparses the full result through runtime admission. Duplicate or
unordered names, overlap, wrong ownership/storage/shape, zero supplied roots and
out-of-range integers are rejected. It does not derive layout/disposition roots,
prove source provenance, inspect source payloads or publish files. Metadata is
not a full-inventory receipt: this generic encoder accepts nonempty subsets,
while the generation planner/verifier enforce the complete frozen inventory.
The JSON metadata is
held in bounded memory (up to 72,317 records/128 MiB encoded JSON), not streamed
with the payload copier's 1 MiB buffer. Added fixture-equivalence and rejection
test sources have not been executed; only C++ syntax is checked locally.

`tensor_inventory.h` supplies the frozen 0731 name grammar (67,612 PP1 target
names or 72,317 source names including the three MTP stages). Both generation
planning and offline target verification now compare every sorted tensor name
against that grammar, rejecting omissions, extras and same-count substitutions.
The implementation checks its expected count and uniqueness before returning.
This is not an implementation of MTP inference and does not qualify DSpark.
Routed main-layer expert dtype/shape checks are also shared by planning and
verification. They mirror `DeepSeekExpertBundleManifest`: I8 packed weights
(w1/w3: 2048×2048; w2: 4096×1024), and F8_E8M0 scales (w1/w3: 2048×128;
w2: 4096×64). Combined with exact names this covers 66,048 PP1 expert tensors.
It does not reinterpret packed storage as unpacked numeric dimensions.
Source provenance depends on independently trusted expected roots. The separate
`ValidateSourceTensorGeometry` adds MTP rules for source admission; PP1 target
validation intentionally excludes MTP tensors.

Dense checkpoint checks now additionally mirror the existing endpoint, mHC,
attention and router-weight bindings: six endpoint tensors and 22 direct tensor
rules per main layer. These cover another 952 names. They are applied in both
generation planning and artifact verification after exact name admission.
The remaining 612 compressor/indexer, shared-expert and gate auxiliary tensors
also have explicit rules, completing the PP1 name/dtype/shape rule set. Rules
derive from the [fixed official inference source](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/inference/model.py)
and repository checkpoint contracts. Shared experts are FP8, not packed routed
FP4; compressor projection storage is BF16, not its reference FP32 compute
parameter dtype. Hash-router storage remains I64 per the repository artifact
contract, not the reference's runtime I32 index buffer. Unknown non-routed names
now fail closed. Static dimensional accounting matches 156,015,698,140 bytes;
real header/model validation has not run. This does not establish execution
support for fused/dequantized runtime consumers, which still require alignment.
