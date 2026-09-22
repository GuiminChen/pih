# Native text support

`bytelevel_tokenizer.{h,cpp}` is shared source compiled into the Qwen model
plugin and the optional CPU-only `pih-tokenize` executable. It does not load
Python, a model runtime, or an external inference service. It is not a new
runtime plugin and does not expose a separate ABI.

The family selector controls vocabulary bounds, special-token identity,
normalizer admission, pre-tokenizer stage count, and decoding policy:

| Family | Pre-tokenization | Decoding |
| --- | --- | --- |
| Qwen3 | One isolated regex split, then ByteLevel | Skip specials; stop on Qwen end-of-text/end-of-message |
| DeepSeek V4 Flash 0731 | Three sequential isolated regex splits, then ByteLevel | Preserve all added-token text, including EOS, for the output parser |

Every isolated split preserves unmatched spans. Later rules operate on each
previous piece separately. Regexes and BPE merge ranks come from the supplied
tokenizer, not from an assumed Qwen-compatible template. The DeepSeek family
requires 128000 dense base IDs and added IDs 0–2 and 128000–129279.
The fixed public metadata reference is
[DeepSeek tokenizer.json at 9e165c30](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/9e165c30e2704aec5d9d593cce3eebd58bbef1cb/tokenizer.json).
No upstream tokenizer payload is redistributed here. The adapted native
conversation codec below has its own preserved upstream notice.

Limits: tokenizer JSON 64 MiB, encode text 1 MiB, each BPE piece 8192 bytes,
decoded bytes 8 MiB, ICU match timeout 1 second per matcher and stack 4 MiB.
The CLI additionally limits token arrays to 65536 and output JSON to 8 MiB.
Incremental decoding retains an incomplete UTF-8 suffix and flushes it with
replacement-character semantics at completion.

Selecting a raw family does **not** authenticate its tokenizer. Use the separate
semantic admission API below for pinned artifacts. The
DeepSeek HTTP/chat adapter now consumes this authenticated path. Token-by-token equivalence
qualification remains separate work; ICU regex/Unicode behavior must be compared with the pinned tokenizer
implementation before claiming equivalence.

## DeepSeek format codec

`deepseek_encoding.{h,cpp}` ports the full-conversation rendering and completed
output parsing rules from the same pinned revision's `encoding/encoding_dsv4.py`.
The upstream MIT notice is preserved in `DEEPSEEK_LICENSE`. `text_json` provides
UTF-8, insertion-order serialization with Python-style separators/float notation;
it is deliberately separate from ASCII authority JSON. These sources are built
by the optional CPU tool `pih-deepseek-format` and are ready to be linked into the
model plugin; semantic-artifact admission is wired into text service loading.

The API requires 1–128 messages, processes an entire conversation, and does not
accept a prefix-cache context. Tool schemas belong to system/developer messages,
as in upstream. User/tool messages are merged and tool results stably ordered by
preceding call IDs before rendering. Thinking history dropping, effort levels,
task transitions, tool argument serialization and response schemas follow the
reference rules. Input/prompt is limited to 1 MiB and output parsing to 8 MiB.

Native hardening intentionally rejects non-finite/duplicate-key request JSON, invalid
UTF-8, duplicate DSML parameters, malformed non-string parameter values, and
completed turns without EOS. Unlike the upstream parser, a zero-parameter call
accepts the empty arguments line produced by its renderer. Length-truncated
turns are not accepted by the completed-turn parser. No tool is executed.
These sources pass Linux-target syntax-only compilation, but have not been
linked or equivalence-qualified; tool support here
must not be advertised as qualified HTTP/tool-calling support.
Historical `function.arguments` is a JSON string: as in upstream, a string that
cannot be parsed is encoded as one string-valued `arguments` parameter rather
than executing or repairing its contents. Valid non-object argument JSON is
rejected.

## Incremental completion decoding

`DeepSeekCompletionDecoder` consumes valid UTF-8 fragments from the tokenizer's
incremental decoder. It holds suffixes that may become structural markers and
distinguishes reasoning, ordinary content, tool calls and EOS. A partial outer
tool marker takes precedence over its inner DSML marker. Text and reasoning
deltas are available as they arrive; tool calls are buffered until the complete
turn passes the strict parser. At a length limit, unfinished DSML is discarded,
not emitted as content or executable calls. Other unfinished ordinary text is
preserved, and a reasoning-only length-limited answer remains reasoning-only.

`Finish(true)` requires a valid completed turn including EOS and checks earlier
emitted text against the final parse. `Finish(false)` handles length exhaustion.
No fragments may follow EOS or finalization. Discard the request-owned decoder
after an exception. The raw generation is bounded to 8 MiB. The offline format
CLI's `decode` operation exposes deltas plus the final message for later manual
qualification; the native model adapter consumes it for HTTP streaming, without
claiming that model execution or end-to-end streaming has been tested.

## Pinned semantic admission

`DeepSeekSemanticArtifacts::Load` reads all 15 original snapshot paths through
Linux `openat` descriptors with `O_NOFOLLOW` at every path component. Each
payload must be a regular bounded file; no symlink-based Hub cache or FIFO is
accepted. The complete domain-separated closure hash includes the pinned
revision and every ordered role/length/content digest. Admission requires
`f77784ede96fb3bc9520622850f94d545ee729bca01bfaecc6d14a29a62ef03c`.
It is the same semantic identity used by the existing artifact contract, not a
new caller-supplied trust root. The copied tokenizer bytes are authenticated
before construction and are never reopened by pathname. Other artifact bytes
are discarded after hashing/config checks. Source and test files are hashed,
never executed. Model weights are not part of this semantic closure.

`pih-deepseek-semantic-verify SNAPSHOT_DIR` emits admission metadata only after
successful loading. `pih-tokenize deepseek-v4-flash-0731-verified SNAPSHOT_DIR
REQUEST.json` uses the admitted in-memory tokenizer. Non-Linux admission fails
explicitly. This implementation has passed syntax-only compilation, not linking
or qualification, and does
not prove equivalence of the native codec to upstream, correctness of model
weights, or successful model execution. Native HTTP loading now uses this API;
target linking and end-to-end qualification remain open.
