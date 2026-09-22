# Native supervisor configuration

`SupervisorConfig::Load(path, trusted_digest)` reads an absolute, non-symlink,
read-only, single-link regular file and authenticates its exact bytes before
parsing. The digest must come from trusted deployment admission, not from
hashing an untrusted configuration and treating that as authorization.
`Parse(bytes, trusted_digest)` applies the same digest/schema checks to a
caller-owned snapshot. Neither method starts workers or mutates cgroups.

The schema name is `pih.deepseek-v41.supervisor.v1`. All fields below are
required; unknown/missing/duplicate keys and wrong types are rejected. JSON is
limited to 2 MiB, depth 8, 4096 nodes and 1 MiB per string. Integer fields are
nonnegative signed-64-bit JSON integers, with the narrower bounds below.

| Root field | Required members / meaning |
| --- | --- |
| `schema` | Exact schema name above |
| `worker`, `helper` | `path`, `sha256`, `byte_budget` for the rank and NCCL-helper ELF |
| `library_directories` | Array of at most 16 unique absolute library directories |
| `cgroup` | `delegated_parent`, `rank_limits`, `helper_limits` |
| `request` | `tokenizer_directory`, `rendered_prompt`, `identity`, `sampling`, `stopping`, `budgets`, `maximum_positions`, `first_plan` |
| `model` | `config_sha256`, `map_sha256`, `plugin_lock`, `plugin_lock_sha256`, `sm_major`, `sm_minor`, `staging_bytes` |
| `ranks` | Ordered array of exactly 2, 4 or 8 rank objects |
| `timeouts` | `startup_ms`, `sequence_ms`, `retirement_ms`, `terminate_grace_ms` |

Paths are canonical absolute POSIX paths, at most 511 bytes, without dot
components, repeated/trailing separators or control characters. Library paths
also reject `:`, `;` and `$`; their combined colon-separated length is at most
4096 bytes. These are lexical checks; `WorkerEnvironment` later performs
directory ownership/permission checks. Digests are nonzero SHA-256 hex values.
Executable byte budgets are 64 bytes–512 MiB; actual ELF snapshots are admitted
separately. This configuration is not proof of shared-library closure.

Both cgroup limit objects require `memory_bytes` (positive multiple of 4096),
`pids` (1–65536), `cpu_quota_us` (1000–1000000000) and `cpu_period_us`
(1000–1000000). Rank limits apply independently to each rank, not as a total
device/host budget. Helper limits apply to its separate cgroup. Creating these
groups requires an explicitly delegated parent; parsing does not enable any
ancestor controller or modify an existing cgroup.

The request objects have these fields:

- `identity`: positive `epoch`, `sequence_generation`, `sampling_config_id`.
  The initial plan identity remains zero; `first_plan` is a separate positive
  value used when generation begins.
- `sampling`: numeric `temperature` and `top_p`, integer `top_k`, `seed`,
  boolean `logprobs`, integer `top_count`. The existing native sampling policy
  validates their combinations. Ordinal and suppressed-token fields cannot be
  supplied here; the ledger owns them.
- `stopping`: `minimum`, `maximum`, `tokens` (at most 17 IDs in 0–129279),
  `strings` (at most 16 nonempty UTF-8 strings of at most 256 bytes each).
  Duplicate/invalid stop entries are rejected by native stop-policy validation.
- `budgets`: `vocabulary_bytes` (1–64 MiB), positive `publication_bytes` and
  `record_bytes`, and `output_slots` (1–4096). Downstream admission verifies the
  actual storage reservations against these budgets; metadata/allocator overhead
  is not counted as payload bytes.
- `maximum_positions`: 2–1048576; maximum completion must be smaller. The
  authenticated tokenizer and request owner subsequently check prompt tokens
  plus completion against this context and the 4096-token prefill limit.
- `rendered_prompt`: a nonempty string, at most 1 MiB. It must already contain
  the correct model prompt framing. The parser does not invent a chat template.

Each rank object has `device`, `artifact_directory`, `weight_manifest_sha256`,
`device_budget`, `host_budget`. Array position assigns rank; every device ordinal
must be distinct and fit a signed 32-bit integer. Device/host budgets are
positive; host budget must cover `staging_bytes`. The common staging size must
be a multiple of 256 in 256–1048576. SM major/minor are 1–99 and 0–9: these
structural bounds are **not** a claim that every architecture is supported.

The rank-worker build must select `103-real` and the
`pih.kernels.deepseek-v41.sm103` plugin; the build script supplies both.
CMake embeds the architecture in the host worker as `PIH_V41_COMPILED_SM`. The sealed
bootstrap's `sm_major * 10 + sm_minor` must match that value before artifact
loading, plugin activation or CUDA/NCCL initialization. The device capability
provider still independently checks the actual GPU. A wrong-architecture worker
is rejected rather than relying on a later kernel-launch failure. Rebuild the
worker and its CUDA dependencies together, then update its trusted file digest
in the supervisor configuration; renaming the executable is not a migration.
Host C++ syntax compilation does not verify any CUDA binary or qualify B300
hardware. Other architectures and multi-architecture worker builds are rejected.

rank worker 必须选择 `103-real` 编译架构及 SM103 Kernel Pack。启动时
先比对密封配置中的架构与可执行文件编入值，再加载制品和初始化插件/CUDA/NCCL；
设备提供者仍需独立验证实际 GPU。跨架构部署必须整体重建并更新受信摘要，不能
只改文件名。C++ 语法检查不代表 CUDA 编译或 B300 真机验证通过。
Actual device/kernel compatibility is admitted by the backend/runtime.

Startup and sequence durations are 1–86400000 milliseconds with sequence
strictly longer than startup. Retirement is 1–300000 milliseconds; termination
grace is nonnegative and strictly shorter than retirement. Runtime orchestration
must convert durations to absolute monotonic deadlines with overflow checks.
PID/UID, fresh private endpoint names and empty NCCL IDs are populated by the
runtime, not accepted from JSON. The broker later supplies the real ID.

The parser emits typed request/deployment values and ordered placement templates
for `SupervisorOperation`. The `pih-v41-supervisor` CLI performs runtime
deadline/endpoint/cgroup assembly; see the README's command section. The text
HTTP service is wired through the native model plugin, but has not been linked
or run on the target hardware. Successful parsing or source compilation is not
runtime inference validation or model qualification.

`--check-config` authenticates the configuration and validates static request
policy only. `--check-request` additionally opens the authenticated tokenizer,
encodes the sealed rendered prompt, checks its token count against the reserved
context, and allocates the bounded CPU output queue and completion ledger. It
emits a `pih.deepseek-v41.request-admission.v1` JSON observation with
`workers_started:false`. Neither mode loads the model, initializes CUDA/NCCL,
starts workers, or modifies cgroups. Both require an independently trusted
nonzero lowercase SHA-256 of the configuration file. Example:

```sh
bash deploy/run-v41-native.sh check-request /absolute/build-dir \
  /absolute/config.json TRUSTED_CONFIG_SHA256
```

Successful request admission does not authenticate the worker ELF, shared
libraries, weight payloads, devices, or delegated cgroup. The actual `run`
path performs those later checks and still requires a complete retirement and
output ledger before reporting success.
# Exceptional retirement behavior

## Final completion record

After successful generation, output writes and final cgroup reconciliation, the
CLI writes one JSON line to stderr with schema
`pih.deepseek-v41.supervisor-completion.v1`. It contains `epoch`,
`sequence_generation`, `world_size`, `prompt_tokens`, `completion_tokens`,
`visible_bytes_written`, `finish_reason` (`stop` or `length`) and
`cleanup` (`reconciled`). Completion tokens are committed ledger records, including
tokens hidden by stopping; visible bytes count successful stdout writes, not
consumer reads or durable storage. This is not a numerical/hardware qualification
or signed receipt. Stderr may contain earlier diagnostic lines.

Require both exit status zero and this final record. A reporting/flush failure
returns nonzero even if execution and cleanup completed. Errors, cancellation or
unreconciled cleanup never deliberately emit a success record. Keep stdout and
stderr separate (for example, `>output.bin 2>run.log`); do not merge the JSON into
the raw token output. No inference was executed to validate this reporting path.

Before emitting completion, the supervisor also requires the number of fully
written, released output publications to equal the committed token-ledger count,
including publications whose visible payload is empty. Accepted-token ordinals
must be consecutive. A promised output-ready state without a retrievable lease
is an error, not an empty poll. These checks prevent omitted/duplicate output
handoffs from being hidden by an otherwise terminal operation state.

The CLI continues consuming queued leases after the process operation becomes
terminal. Released-publication ordinals are tracked separately from successful
delivery, so cancellation can discard and release successive publications without
mistaking an earlier discarded item for an ordinal gap. Success requires both
counts to agree with the ledger. The sequence deadline is checked before output
writes (including empty publications), as well as after a write; an expired
request cannot acquire successful delivery credit during terminal draining.
This path has passed Linux-target syntax compilation, not runtime validation.

成功生成、输出写入及最终 cgroup 核对完成后，stderr 会输出上述 schema 的 JSON
完成记录，包含请求标识、rank 数、token 数、可见输出字节数、结束原因及清理状态。
completion_tokens 来自已提交账本，包含停止规则隐藏的 token；字节数只代表 stdout
写入成功，不证明下游读取或磁盘持久化。必须同时检查退出码为零及最终记录；报告
写入/刷新失败也会返回非零。stderr 可能含前面的诊断行，应与原始 stdout 分开保存。
该记录不是数值/硬件资格或签名证明，本次未运行推理验证。

完成前还要求“已完整写出并释放的输出记录数”等于已提交 token 账本数，包括可见
内容为空的记录；接受 token 序号必须连续。状态声明输出就绪却取不到租约时直接
报错，不能作为空轮询忽略，避免终止状态掩盖漏交付或重复交付。

操作进入终止状态后仍继续处理队列中的租约。释放计数与成功交付计数分开记录，
取消时丢弃并释放后续输出不会因前一条未交付而误报序号缺口；成功时两个计数都
必须与账本一致。输出前及写入后均检查请求期限，空输出也不能在过期后计为成功
交付。本路径仅通过 Linux 目标语法编译，未运行模型或故障注入。

The supervisor catches both standard and nonstandard exceptions while it still
owns the operation and its borrowed request/environment/cgroup objects. Both
paths attempt bounded cancellation/retirement and then final cgroup reconciliation.
If reading a created group's population fails, final cleanup still attempts the
owned group's identity-checked kill operation; a failed observation is never
treated as evidence that the group is empty. Removal still requires an actual
empty observation and member identity validation. An unresolved cleanup returns
failure and leaves the logged namespace for inspection, never a success receipt.
These changes passed Linux-target syntax compilation only; no processes or
cgroups were started for validation here.

Supervisor 现同时捕获标准与非标准异常，并在操作及其借用对象仍存活时尝试有界取消、
回收及最终 cgroup 核对。已创建组的 populated 状态读取失败时，仍尝试终止所拥有的
组；实际写入前由 Kill 检查身份，不能把观测失败当作“组已空”。删除仍要求真实空组
观测及身份验证，未完成清理必须报错并保留日志中的命名空间供检查。本次仅语法编译，
未启动进程或操作 cgroup 进行验证。
