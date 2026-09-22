# Native Qwen artifact tools

This directory owns the INT4 conversion pipeline and CPU command entrypoints.
It does not import Python, link CUDA, or provide a model execution fallback.

- `pih_qwen_conversion_impl`: six implementation files in this directory for
  source access, payload quantization, conversion streaming and publication.
- `pih_qwen_artifact_impl`: explicit shared format/manifest source set, currently
  still located under `src/model`; no quantization pipeline dependency.
- `pih-qwen-int4-convert`: trusted source expectation to exclusively published
  INT4 file. Existing output/staging paths are not overwritten.
- `pih-qwen-int4-verify`: read-only format/payload verification against explicit
  file, source, binding and disposition digests; no quality qualification.
- `pih-qwen-source-verify`: read-only BF16 source manifest/digest/tied-weight
  checks against the same strict expectation schema used by conversion.

Configure `PIH_BUILD_QWEN_ARTIFACT_TOOLS=ON` for CPU-only builds. Deployment
commands and environment requirements are documented in the
[English guide](../../docs/native-inference.md) and
[Chinese guide](../../docs/native-inference.zh.md).

The model plugin excludes the conversion target's normalized source paths.
No forwarding implementation remains at their former `src/model` locations.
The five conversion-only headers also live here, using local includes rather
than publishing conversion interfaces under `include/pih/model`. Existing source
tests reference these private headers explicitly; no forwarding headers remain.
The additional `qwen3_int4_artifact_writer.{h,cpp}` owns the write-plan and file
writer functions extracted from the shared artifact I/O module. That shared
header now exposes verification only and does not include the extent writer.
The historical monolithic test source inventory temporarily names the relocated
files; its removal and migration of shared format headers/sources remain open.
Moving these files does not prove complete plugin isolation or a passing Linux
link. Current checks are source compilation only; conversion/model execution
and numerical qualification have not been performed during this migration.
