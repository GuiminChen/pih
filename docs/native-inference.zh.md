# 原生插件推理：构建、启动与观察结果

[English](native-inference.md) | 中文

当前代码/构建结论以[集成状态](integration-status.zh.md)为准：生产链接与设备镜像已验证，硬件运行未验证。构建/发行与系统前提见[部署指南](deployment.zh.md)。下列资格验证命令仅供操作者后续显式使用，不属于本次交付验收。

Qwen 的 `serve`/`seal` 和 DeepSeek 的 `serve-deepseek` 支持 `--generation-timeout-ms`，默认 `600000`，范围
`1..86400000`。该必填模型配置写入新 Lock；旧文本服务 Lock 必须重新生成，不提供
兼容默认值。期限从请求解析/分词前开始，在打开输出流前及原生生成期间检查，
但不能抢占正在执行的 CPU 函数或 CUDA 调用。协作回收使用独立的有界期限，
HTTP 接收/写入超时也独立计算；尚未运行模型验证实际超时效果。

仅构建 Qwen 模型时，不再为了检查源码清单而创建离线转换目标；模型直接排除
`plugins/offline-qwen` 所属源码。需要转换命令时显式启用
`PIH_BUILD_QWEN_ARTIFACT_TOOLS=ON`，文档中的 Bundle 构建脚本已设置该选项。
这项构建边界改动不代表 CUDA、链接或模型执行验证通过。

部署配置命令会先重置可选工具、原生契约测试、历史测试和通信选项，再应用所选模型
配置。Qwen Bundle 明确启用所需 CPU 制品工具，避免请求构建的目标未被配置。
V4.1 Shell 构建也会关闭两类测试。该措施只控制 CMake 缓存选项，不清除已经安装的
旧文件；不同模型配置和架构仍应使用独立构建、安装目录。

## 先明确状态

预发布 CUDA 异步能力表新增 `require_clean_last_error`，模型插件和 CUDA 后端
必须一起重编译，不接受旧能力表大小。Qwen 的事件记录/查询和线程错误检查现
通过该能力调用；线程上下文不匹配会失败，不会切换上下文或清除错误。
类型化 H2D/D2H/D2D 复制也已接入，校验地址范围并保留提交前后错误检查。
内核调用等直接 CUDA 操作仍待迁移，这不代表运行验证完成。

KV 清零/回收也已接入异步能力，校验地址范围并有界等待完成事件；pending 状态
不能视为回收成功。生产 CUDA 目标和原生契约目标已完成编译与最终链接；生成的 Linux
可执行文件及新增回归断言尚未运行。

每步错误缓冲区清零也已独立接入能力接口，校验四字节目标和 rank，不再由具体
矩阵驱动兼任。矩阵计算逻辑未改动，其直接后端依赖仍待迁移。

本指南只使用原生 C ABI 插件，不使用 `_pih`、Python 模型进程、Transformers
推理或自动回退。Python 启动脚本只负责构建、生成部署 Lock、启动原生 Worker
及发 HTTP 请求。**首版代码与 Linux/CUDA 构建交付已完成；这不代表 Linux 执行、
硬件推理或数值正确性验证通过。**

Qwen 还必需包含 `pih.backend.nvidia-cuda` 和 `pih.platform.linux`，启动脚本
自动构建并封装两者及后端注册的全部能力。Qwen 通过能力接口准备选定 SM 的
设备、分配设备/固定页内存，并执行 BF16 权重上传。缺少这些提供者的旧 Lock
必须重新生成；内存回收失败会终止 Worker，不能当作成功或可重试的 HTTP 响应。
流、事件和内核的直接 CUDA 依赖仍待迁移，此改动不代表运行验证通过。

Qwen 还通过 `device.cuda-resources.v1` 创建和回收主上下文、流及事件；构建脚本
已包含该后端能力，无需新增命令行参数。提供者必须存活到全部资源回收结束，
释放失败会终止 Worker。内核调用等操作仍有直接 CUDA 依赖；此资源
绑定仅通过语法检查，未进行设备运行验证。

| 环境与模型 | 当前入口与边界 |
| --- | --- |
| Linux / RTX 4090 D / Qwen3-0.6B | 新原生模型插件 + SM89 Kernel Pack + HTTP；BF16 或 PIH INT4；待验证 |
| Linux / H100 PCIe / Qwen3-0.6B | 同一模型插件 + SM90 Kernel Pack；单卡；待验证 |
| Linux / RTX 4090 D / DeepSeek V4 Flash-0731 | 原生 token/文本/聊天及 SSE 已接线；需要验证权重和固定语义快照；待验证 |
| Linux / B300 / DeepSeek V4.1 Flash | 已有原生模型与 SM103 Kernel Pack、rank worker、supervisor 和 HTTP 源码链；已完成 Linux/CUDA 生产链接，硬件运行未验证，不宣称合格 |
| Windows / macOS / CPU-only | 可以阅读清单、发 HTTP 请求；没有本指南中的本地 GPU 推理路径 |

### DeepSeek V4.1 / B300 开发 Bundle

在具有支持 `sm_103` 的 CUDA 工具链、ICU、OpenSSL、NCCL、CMake/Ninja 的 Linux
B300 主机上，创建全新 Bundle 并封装 rank Lock：

```bash
bash deploy/run-v41-native.sh build-service /absolute/build-dir /absolute/new-bundle 4
bash deploy/run-v41-native.sh seal-rank /absolute/new-bundle
```

按照该 Bundle 的 worker、helper、`v41-rank.lock` 的精确路径与 SHA-256、B300 rank
放置、受委托的 cgroup 父目录及外部已验证模型制品，编写经过认证的
`pih.deepseek-v41.supervisor.v1` 配置。完整字段和准入规则见
[`SUPERVISOR_CONFIG.md`](../plugins/model-deepseek-v41/SUPERVISOR_CONFIG.md)。
配置摘要必须来自独立可信的部署记录，不能对未经审核的输入现算摘要就当作信任依据。
随后执行：

```bash
bash deploy/run-v41-native.sh seal-service /absolute/new-bundle /absolute/config.json TRUSTED_SHA256
bash deploy/run-v41-native.sh serve /absolute/new-bundle 8000
# 若明确要在该主机测试，可从第二个 shell 执行：
bash deploy/run-v41-native.sh probe /absolute/new-observation.json 8000 32
```

`build-service` 只编译安装；`seal-rank` 与 `seal-service` 写入新 Lock，不执行模型。
`serve` 和 `probe` 会执行模型，不属于当前仅静态检查的开发阶段。脚本不自动下载或
转换权重，Python probe 只发送 HTTP。rank Lock 现要求同时包含 CUDA 后端与
`pih.transport.nccl`；NCCL 通信器及 collective 提交由该提供者持有。旧 V4.1
Bundle 必须重新构建封装，rank worker 不再保留原始 NCCL 句柄兼容路径。
只检查代码时可运行
`python3 tools/check_native_syntax.py --v41-model`，但必须先备好固定版本的本地
语法头文件和 sysroot；此命令不编译 `.cu`、不链接 Linux ELF，也不证明 B300 推理正确。

不要把普通 RTX 4090 当成 RTX 4090 D，也不要把 H100 SXM 当成 H100 PCIe。
Qwen 当前不会在 B300 上复用 SM89/SM90 cubin。所有组合均未获得生产资格。

## DeepSeek V4-0731：原生 token 生成

新增 `inference.tokens.v1` / `pih.inference.tokens.v1` C ABI，接收真实 token
序列，由插件内部执行分块 prefill、采样、decode、输出确认及请求/注意力资源
回收。EOS、最大输出长度由现有原生 ledger 判定；C ABI 还提供温度、top-p、
seed、最小输出长度、最多 16 个 stop token 和单调时钟 deadline。
Worker 命令行目前暴露贪心生成，seed=7，生成超时 30 分钟，关闭超时 5 分钟。

这不是字符串/聊天接口：必须提供匹配 **V4 Flash-0731** 官方 tokenizer 的
token IDs（包含所需模板与特殊 token），不能使用 V4.1 或 Qwen 的 tokenizer。
返回原始 `token_ids`、`finish_reason`、prompt/completion 用量；不伪造回答文本。
旧的单 token smoke 只是诊断入口，已复用同一个生成循环，不是另一套推理实现。

准备一个仓库外、已经发布的 PP1 verified generation，包含 `pih.manifest.json`、
`pih.runtime-records.json` 和对应权重。`ARTIFACT_ROOT` 是该制品发布的可信根哈希，
不是目录字符串的哈希；普通 Hugging Face 权重目录不能直接代替。
然后在 Linux / RTX 4090 D 上运行：

```bash
export ARTIFACT_ROOT='替换为已验证制品的64位小写十六进制根哈希'
export PROMPT_TOKEN_IDS='替换为匹配0731模型的逗号分隔token IDs'
python3 deploy/native.py generate-deepseek-tokens --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" \
  --prompt-tokens "$PROMPT_TOKEN_IDS" --max-tokens 128 --jobs 4
```

`--build` 不运行模型测试：先构建封装 Bundle、校验/绑定外部制品，最后执行你明确
请求的 token 生成。默认构建目录为 `out/build/deepseek-4090d-pp1`。
**构建会重建其中的 `bundle/deepseek-4090d-pp1` 子目录，不要在那里保存权重或证据。**
已建好的 Bundle 再次生成时省略 `--build`、`--artifact-dir`、`--artifact-root`。
脚本只编排，不加载 Python 模型；最终进程为原生 Worker。

也可以直接调用：

```bash
out/build/deepseek-4090d-pp1/bundle/deepseek-4090d-pp1/bin/pih-worker \
  --lock out/build/deepseek-4090d-pp1/bundle/deepseek-4090d-pp1/deepseek-v4-flash-rtx4090d-pp1.lock \
  --generate-tokens "$PROMPT_TOKEN_IDS" --max-tokens 128
```

输入加输出长度不能超过 Lock 的 `attention_reserved_tokens_per_sequence`（模板
当前 4096）；逗号之间不接受空项或空格。输出只有在生成和资源关闭均成功后发布。
此接口增加了新的注册能力，必须重新构建整套 Bundle/Lock；不会修补旧 Lock 或
忽略能力清单不匹配。当前代码已完成 Linux/CUDA 生产编译和链接，硬件运行未验证，不承诺数值正确性。

V4.1 已有独立的固定 HF 配置解析器、原生模型插件、SM103 Kernel Pack、rank worker
和 supervisor，位于 `plugins/model-deepseek-v41` 与
`plugins/kernels/deepseek-v41-sm103`。它拒绝 V4-0731 配置，并且仅在封装的 V4.1
Bundle 中注册 `inference.text.v2`。这条源码链已完成生产链接；仍未执行运行集成和模型
验收；源码可编译不等于 B300 推理已经正确运行。

## DeepSeek V4-0731：原生文本服务

模型插件现提供 `inference.text.v2`，使用已验证语义字节和原生 PP1 引擎。
启动命令只组合原生提供者与 `pih.surface.text-http`，不加载诊断 HTTP Surface，
不启动 Python 模型后端。Linux RTX 4090 D 机器需 CUDA、ICU/OpenSSL 开发库：

```bash
python3 deploy/native.py serve-deepseek --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" \
  --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731 \
  --max-context 4096 --port 8000 --jobs 4
```

`ARTIFACT_ROOT` 必须是上一节所述已验证权重 generation 的准确根摘要，不是原始
checkpoint 哈希。语义快照必须满足下文固定 15 文件校验要求；两类制品目录都须
位于源码、构建和安装目录之外。默认构建/安装目录为 `out/build/deepseek-text`
和 `out/install/deepseek-text`；使用新构建目录，重新构建前先停止旧 Worker。
命令挂载外部权重目录、计算插件摘要并生成内容寻址 Lock。只有已构建该 bundle
时才省略 `--build`；此命令不转换原始权重。

出现 `state: serving` 后，由你显式发送请求：

```bash
curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4-Flash-0731","messages":[{"role":"user","content":"你好"}],"thinking_mode":"chat","max_tokens":128,"temperature":0,"top_p":1,"stream":true}'
```

`stream:false` 返回完整 JSON。聊天历史须以 user/developer/tool 输入结束；支持
`thinking_mode` chat/thinking（默认 chat）、`reasoning_effort` low/high/max、
`drop_thinking`、`tools`、提示词级 `response_format`。响应 schema 只是提示词
约束，**不是约束解码或保证符合 schema**。工具调用带 ID 返回，只作为数据，
不会执行；工具块校验完整后才发布调用片段，reasoning 和正文仍逐段输出。长度
耗尽时不发布未完成的工具块。拒绝 `tool_choice`、图像、任意停止字符串、logprobs、
penalty 及其他未实现字段。

聊天和 `/v1/completions` 均支持 max_tokens、temperature 0–2、top_p (0,1]、
非负整数 seed、布尔 stream 和 n=1；贪心 temperature=0 要求 top_p=1。
原始补全使用字符串 prompt，不自动添加聊天模板，输出省略 EOS。输入加输出不得
超过 Lock 中的上下文上限；启动器暂限定 4096，并非已验证容量。服务串行、仅回环
监听，无 TLS/认证；生成预算可配置，默认 10 分钟，中断后最多再等待当前设备计划 5 分钟再
回收，不抢占 CUDA。完成清理的超时返回 504；生成结构错误或清理失败会终止 Worker。
这些代码已完成交叉链接，硬件运行未验证。

也可从已提交并激活的存储库启动文本服务：

```bash
python3 deploy/native.py serve-deepseek --build \
  --store-dir /srv/models/deepseek-store \
  --pointer-root "$POINTER_ROOT" --catalog-root "$CATALOG_ROOT" \
  --semantic-snapshot /srv/models/DeepSeek-V4-Flash-0731 \
  --max-context 4096 --port 8000 --jobs 4
```

该模式先构建/调用 CPU 原生存储库解析器，核对预期指针、回执和全部目标文件，再把
选定 artifact 根写入文本 Worker Lock。`--build` 同时构建服务；解析器使用独立构建目录，
默认 `out/build/deepseek-artifact`，可通过 `--artifact-tools-build-dir` 调整。省略 `--build`
时，原生工具和服务 bundle 都须已构建。存储库、语义快照、构建/bundle 目录不可重叠。
存储库模式不要再传 `--artifact-dir` 或 `--artifact-root`；根缺失、过期或验证失败直接
终止，不回退。这个只读目标验证不要求保留原始权重 checkpoint。

启动固定使用解析时选定的版本，不订阅后续指针变化。Python 仅调度并检查原生工具的
小型观察结果，Worker 仍须独立完成产物/存储准入。catalog 根只与指针比对，并未在此认证
catalog 正文。后续激活不热重载该进程。这条启动路径已实现，但本地未执行 Linux/CUDA 服务。

## 1. 准备 Linux 构建机

需要：支持 C++20 的编译器、CMake 3.26+、Ninja、OpenSSL 3 开发文件、ICU
开发文件、CUDA Toolkit 13.2（由当前 CMake 检查）、匹配驱动，以及 Python
3.11+（仅脚本）。ICU 用于 Unicode 正则预分词，不执行模型。

在项目根目录执行只读工具检查：

```bash
python3 --version
cmake --version
ninja --version
c++ --version
nvcc --version
nvidia-smi --query-gpu=index,name,compute_cap --format=csv
python3 deploy/native.py list
```

安装编译器、CUDA 和开发依赖由运维管理；脚本不会自动使用 sudo、安装系统包、
修改驱动或下载权重。编译找不到 CUDA 时，指定 `CUDACXX` 为实际 nvcc 路径。
运行 Worker 前不要设置 `LD_PRELOAD`、`LD_LIBRARY_PATH` 等 loader 注入变量；
现有 Worker 会拒绝它们。应通过正确的系统库安装/RPATH 提供运行依赖。

## 2. 准备模型目录

权重必须在源码树和构建树之外，例如 `/srv/models/Qwen3-0.6B`。
从可信的官方修订取得制品，并保留下载修订和哈希。脚本不会自动下载。

BF16 目录至少包含：

```text
/srv/models/Qwen3-0.6B/
  config.json
  tokenizer.json
  model.safetensors
```

INT4 使用另一个目录，保留相同的 `config.json` 和 `tokenizer.json`，权重名为
`model.xing-int4`。它是 PIH 自有转换格式，不接受 GGUF、AWQ 或 GPTQ 代替。
已新增不依赖 Python 扩展的原生 INT4 转换工具。先构建 Bundle，再提供绑定可信
原始文件的 expectation JSON（schema、file_bytes、tensor_count、data_bytes、
file_sha256 五个字段，格式沿用已有制品契约）：

```bash
python3 deploy/native.py convert-int4 --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json \
  --output /srv/models/Qwen3-0.6B-int4/model.xing-int4
```

输出目录需已存在，目标文件及 `.staging` 均不得存在；不会覆盖。转换仅发布权重，
将同修订的 config 和 tokenizer 放入输出目录，保存 stdout 的转换收据。
可用下面的命令代替手工复制（输出目录必须已存在）：

```bash
python3 deploy/native.py prepare-qwen-metadata \
  --source-dir /srv/models/Qwen3-0.6B \
  --model-dir /srv/models/Qwen3-0.6B-int4 \
  --config-sha256 "$TRUSTED_CONFIG_SHA256"
```

配置摘要必须来自独立可信的同修订记录；tokenizer 使用原生服务固定的摘要。
命令先认证两个有界文件快照，再独占创建只读副本、回读校验并同步目录。
它不转换权重、不验证配置语义，也不构成模型资格证明。目标目录须由当前用户
拥有且不可被组或其他用户写入；执行期间不得有其他进程修改该目录或其父路径。
已有文件（包括符号链接）均拒绝覆盖。不保证两个文件或权重一起原子发布：
失败可能保留一个或两个文件，不能将部分结果用于服务；检查失败原因后使用新的
输出目录，不要自动删除或盲目重试。此命令尚未实际执行。

若尚未复制配置，可以在转换命令中加上两项参数，一次完成元数据准备和原生转换：

```bash
python3 deploy/native.py convert-int4 --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json \
  --output /srv/models/Qwen3-0.6B-int4/model.xing-int4 \
  --prepare-metadata --config-sha256 "$TRUSTED_CONFIG_SHA256"
```

配置和 tokenizer 取自源权重所在目录，输出文件名必须为 `model.xing-int4`。
先构建或选择原生工具，再认证并复制元数据，最后调用原生权重转换器；元数据失败
不会启动转换。权重转换失败仍可能保留元数据和权重 staging，整个命令不是原子事务，
不会自动清理。不要在同一目录先运行独立元数据复制，再加此选项转换，否则会因已有
文件而拒绝。仍须按下文独立校验 INT4，成功后才能进入服务启动步骤。
不能把任意本地文件的自算哈希称为官方来源证明；没有可信制品时先完成来源确认。

转换前，或不进行转换而部署 BF16 时，也可以独立校验源文件，复用同一份可信
预期文件：

```bash
python3 deploy/native.py verify-qwen-source --build \
  --source /srv/models/Qwen3-0.6B/model.safetensors \
  --expectation /srv/models/qwen-source-expectation.json
```

此 CPU 命令使用原生来源校验器检查固定布局 manifest、整文件摘要、张量及数据
长度、embedding/LM-head 共享权重字节一致性，不转换权重、不加载引擎。它输出
来源观测收据，不是部署锁或数值资格证明。预期文件解析器与转换器共用，严格
要求五个格式字段、正数且有界的计数和非零小写摘要。预期值必须独立可信，
校验期间源文件必须保持不可变。本次实现未实际执行来源校验。

转换后，校验 INT4 制品：

```bash
python3 deploy/native.py verify-int4 --build \
  --artifact /srv/models/Qwen3-0.6B-int4/model.xing-int4 \
  --file-sha256 "$FILE_SHA256" --source-root "$SOURCE_ROOT" \
  --binding-root "$BINDING_ROOT" --disposition-root "$DISPOSITION_ROOT"
```

此入口复用 CPU 构建路径，构建前拒绝格式错误的预期摘要，只调用原生校验器。
它不修改制品、不自行生成预期摘要、不启动服务。不带 `--build` 时使用
`--bundle` 指定安装目录中的工具。也可以直接调用已构建的可执行文件：

```bash
out/build/qwen3-artifact/plugins/offline-qwen/pih-qwen-int4-verify /srv/models/Qwen3-0.6B-int4/model.xing-int4 "$FILE_SHA256" "$SOURCE_ROOT" "$BINDING_ROOT" "$DISPOSITION_ROOT"
```

四个非零小写摘要应来自独立可信的预期记录，分别对应 `file_sha256`、
`source_artifact_root`、`source_binding_root`、`disposition_root`。使用已保存
转换收据中的摘要只证明文件相对该收据未漂移，不是独立官方来源证明。工具检查
布局、载荷摘要及四项预期绑定，输出收据，成功返回 0、失败返回 2；它不评估
量化质量、不运行模型。校验期间必须保持制品不可变；此命令不创建可信挂载，
也不防止其他写入者截断已映射文件。

离线量化编译禁用 fast-math 和浮点融合，运行要求 round-to-nearest 模式，
不支持的舍入模式会明确失败。即使制品已经发布，转换收据输出失败也返回非零。
失败时可能留下目标文件或 `.staging`，重试前先检查并校验，不能假设失败意味着
没有发布文件。本次实现没有执行转换或制品校验。

服务现在会在分配 Qwen GPU 引擎前验证分词器字节快照（最大 12 MiB），要求
与仓库固定 SHA-256
`aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` 一致。
随后解析同一份已校验字节，不重新打开路径。不同格式或修订的分词器，即使
结构看似正确也会拒绝。CPU `pih-qwen-format --tokens` 复用同一加载函数；
原始 `pih-tokenize qwen3` 仍是不固定摘要的格式工具。这项字节绑定不认证权重
及配置，不证明分词等价性，也不建立可信文件系统或依赖闭包，运行资格仍待验证。

新 C++ 分词器从 `tokenizer.json` 读取 ByteLevel/BPE 词表、merge 和正则规则，
检查 Qwen 特殊 token 标识。纯文本非思考模板参考
[官方 tokenizer 配置](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/tokenizer_config.json)。
模板、Unicode 分词及生成结果仍需后续与固定官方修订做对照验证。

Qwen 生成循环现在在提交前、原生步骤后和每个 token 回调前检查取消及截止时间。
输出或取消回调抛异常时会记录错误，停止后续输出并继续正常 cancel/drain，
不会直接遗弃已提交请求。回调失败仅在请求已回收或根本未提交时报告；引擎或
清理失败仍将引擎标记为不可用并要求重启。这是协作式检查，不能抢占阻塞的回调
或 CUDA 操作；故障注入及真实断连、超时验证仍未执行。

## 3. 一条命令构建并启动

仅构建、不启动推理时，必须显式指定目标：

```bash
python3 deploy/native.py build --target rtx4090d --jobs 4
```

部署构建器为 `rtx4090d` 仅选择 SM89，为 `h100-pcie` 仅选择 SM90，覆盖
cubin 生成和 Kernel Pack 安装。直接使用 CMake 可设置
`PIH_QWEN_KERNEL_ARCHITECTURES=89`、`90` 或 `89;90`，空列表、重复架构及
未知架构会被拒绝。架构选择不代表硬件资格已通过。如果安装目录已有另一种
Qwen 架构，请指定新的 `--bundle` 和 `--build-dir`；脚本不会删除已有文件或
静默混装。单独构建不探测或执行 GPU，`serve` 仍执行独立的主机检查。

以后在 RTX 4090 D 上单独验证 CUDA kernel 时，可在生产构建后显式构建资格命令；
该命令不安装进服务 Bundle：

```bash
cmake -S . -B out/build/qwen3-native -DPIH_BUILD_QWEN_QUALIFICATION_TOOLS=ON
cmake --build out/build/qwen3-native --target pih-qwen-cuda-qualify --parallel 4
export PIH_QWEN_QUALIFIER="$PWD/out/build/qwen3-native/plugins/model-qwen3/pih-qwen-cuda-qualify"
export PIH_QWEN_CUBIN_PATH="$PWD/out/install/qwen3-native/lib/qwen3-sm89/sm_89/qwen_bf16_primitives.cubin"
"$PIH_QWEN_QUALIFIER" shape
```

`shape` 输出五组设备摘要的原始 JSON；`gemm` 和 `metric` 分别运行另外两项
CUDA fixture。`tests/hardware/qwen_m2_shape_corpus_smoke.py` 可在将来用这两个
绝对路径额外核对 GPU 与制品身份。本阶段没有运行 fixture，原始命令成功也不等于
发布资格通过。

RTX 4090 D，GPU 0，BF16：

```bash
unset CUDA_VISIBLE_DEVICES
python3 deploy/native.py serve --build \
  --target rtx4090d --precision bf16 \
  --model-dir /srv/models/Qwen3-0.6B \
  --artifact-sha256 "$TRUSTED_BF16_ARTIFACT_SHA256" \
  --config-sha256 "$TRUSTED_CONFIG_SHA256" \
  --max-context 4096 --port 8000 --jobs 4
```

H100 PCIe：

```bash
python3 deploy/native.py serve --build \
  --target h100-pcie --precision bf16 \
  --model-dir /srv/models/Qwen3-0.6B \
  --artifact-sha256 "$TRUSTED_BF16_ARTIFACT_SHA256" \
  --config-sha256 "$TRUSTED_CONFIG_SHA256" --port 8000
```

已有 PIH INT4 制品：

```bash
python3 deploy/native.py serve --target rtx4090d --precision int4 \
  --model-dir /srv/models/Qwen3-0.6B-int4 \
  --artifact-sha256 "$TRUSTED_INT4_ARTIFACT_SHA256" \
  --config-sha256 "$TRUSTED_CONFIG_SHA256" --port 8000
```

`seal` 与 `serve` 必须提供独立可信的 BF16 `model.safetensors` 或 INT4
`model.xing-int4` 及 `config.json` 的完整 SHA-256。封装时校验，原生加载器
在实际消费的字节上再次校验。仅对不可信本地文件自行计算摘要不足以证明官方来源。
封装时还会按 Qwen 插件内置的摘要校验 `tokenizer.json`，不匹配则不会生成服务锁。
启动脚本也会在哈希前拒绝超出原生快照上限的文件（权重 2 GiB、config 1 MiB、
tokenizer 16 MiB）。
tokenizer、config 和权重均通过 storage 插件的 authenticated-snapshot 能力读取。
provider 从受限描述符复制字节、验证可信 SHA-256，并将复制后的内存页设为只读
后才交给模型。权重加载器消费该不可变快照，不再按路径重新打开。此为代码级准入，
不代表真机资格验证通过。

首次 `--build` 配置并编译原生 Worker、模型插件、HTTP 插件和两种 Qwen
Kernel Pack，关闭 monolith、Python extension、NCCL 和测试构建。默认位置是
`out/build/qwen3-native` 和 `out/install/qwen3-native`，可用 `--build-dir`、
`--bundle` 指定。后续启动省略 `--build`；代码改动后需要重新构建。

启动器还要求 `nvidia-smi` 索引 0 在 PCI 地址排序中也为首个设备，拒绝缺失或
重复设备记录，并在启动原生 Worker 前设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。
NVIDIA 文档说明 CUDA 默认使用最快设备优先的启发式枚举，不保证与管理工具
索引一致，见 [CUDA 环境变量说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。
此检查不证明 MIG/MPS 兼容性、设备独占性，也不防止预检和启动之间的拓扑变化。

脚本检查物理 GPU 0 的准确名称和计算能力，生成带插件/Kernel Pack 二进制哈希
的原生开发 Lock，然后以 `exec` 替换为 Worker，不驻留 Python 推理进程。
不要设置 `CUDA_VISIBLE_DEVICES`；容器需暴露完整物理设备命名空间。

看到 `"state":"serving"` 才表示模型加载和监听完成。初次加载可能较慢。
检查 `/readyz` 返回 200 只表示进程已准备接受请求，不证明输出正确。
`GET /health`、`GET /readyz` 和 `GET /v1/models` 必须携带非空 Host，
不接受请求体、传输编码、重复头或同一连接中已收到的流水线请求。
Content-Length 可省略或严格为 `0`。这些端点与推理请求共用头部校验，
不是额外的 GPU 自检；当前串行服务在生成期间也不能独立响应健康检查。

请求解析错误使用明确状态码：未知路径为 404，错误方法为 405 并带 `Allow`，
推理请求缺少 Content-Length 为 411，请求体过大为 413，不支持的推理媒体类型为
415，头部超限为 431，不支持的 HTTP 版本为 505。其他非法格式返回 400，响应中
仅包含固定错误标识，不回显请求内容。文本服务会在等待请求体前拒绝声明长度超过
1 MiB 的请求。这些响应发生在推理之前，不改变 SSE 开始后的错误处理方式。

文本服务未实现 `100 Continue` 临时响应。收到完整头部后，任何 `Expect` 头都会
直接返回 417，不等待请求体；客户端应去掉该头并直接发送有界请求。发现端点也
适用此规则，避免客户端等待发送许可、服务端等待请求体而互相等待。

每个已接受连接有 30 秒绝对请求读取期限，不会因逐字节发送而延长。部分请求超时后
会尽力返回 408 再关闭连接；完全空闲的连接直接关闭。响应写入另有独立的 30 秒
期限。停止信号会中断轮询；监听描述符轮询失败会以错误结束服务，不再对损坏的
描述符无限重试。上述路径已实现，但尚未进行 socket 或故障注入测试。

Qwen 关闭逻辑区分操作锁竞争和底层关闭失败：锁忙时直接返回可重试状态，不在插件
互斥锁上阻塞；BF16/INT4 引擎会缓存关闭结果，因此底层失败属于终止性错误，保留
引擎所有权，不再当作“清理仍在进行”反复重试。Worker 必须退出而不是卸载可能
仍在使用的提供方。这不限制单次 CUDA/引擎关闭调用的耗时，仍需外部监督和关闭
故障注入验收。

Qwen GPU 初始化失败（包括 C ABI 防护层转换的异常）会使当前插件实例进入失败状态，
必须重启 Worker，不能在进程内重新加载。已有引擎对象会阻止再次加载和插件销毁。
选择的 Kernel Pack 能力 ID 必须与其声明的 SM89/SM90 目标一致，不一致时在模型
加载前拒绝。这些准入检查不验证 GPU 二进制的数值行为。

原生文本/token 引擎出现终止性清理错误、非法关闭状态或清理重试超时后，Worker
会失败退出：诊断输出可写时记录 `pih_worker_fail_stop`，随后调用 `_Exit(1)`，
跳过栈展开及全局/动态库析构。这不是正常关闭，不保证最后的缓冲输出或完成收据。
该行为防止进程正常退出时，析构函数再次清理可能仍在使用的设备资源。监督程序必须
将非零退出视为失败，不能当作成功回收。本路径仅完成编译检查，未做故障注入测试。

服务只监听 `127.0.0.1`，串行处理请求，无鉴权/TLS，不应直接暴露公网。
停止使用 Ctrl+C 或向 Worker 发 SIGTERM；在途生成在下一个调度边界提交取消，
完成 drain 后关闭引擎，不抢占 CUDA kernel。SIGKILL 无法执行应用清理。

## 4. 发请求，检查实际生成

另开终端：

```bash
curl -f http://127.0.0.1:8000/readyz
curl -f http://127.0.0.1:8000/v1/models
curl -f http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"请用一句话解释张量并行。"}],"max_tokens":128,"temperature":0,"top_p":1,"stream":false}'
```

原始补全用 `/v1/completions`，把 `messages` 换成字符串 `prompt`。
原生请求采样现在接受 `temperature` 0–2、`top_p` (0,1]、整数 `n=1` 及可选
整数 `seed`（0..INT64_MAX）。默认仍为 temperature=0、top_p=1、n=1、seed=0。
贪心模式拒绝 top_p 不为 1；正 temperature 选择 BF16/INT4 现有的原生 packed
随机采样器，流式和非流式复用同一参数描述。参数转换为采样器的 FP32 表示，
下溢为零会被拒绝。省略 seed 时固定为零，不隐式使用主机随机数；不承诺跨硬件
或跨构建的逐位可复现性。采样行为与质量仍待模型测试验证。
`stream` 接受布尔值，默认 false。
`stop` 接受 null、一个非空 UTF-8 字符串，或 1–4 个互不重复的字符串，每个最多
256 字节。两种模式都在 UTF-8 解码后增量匹配，暂存可能的停止前缀，不输出
命中的停止字符串及后续文本。重叠标记按最先完成的字节匹配处理，同一完成
位置取最早起点，不依赖分块方式。命中后协作式回收原生请求，清理完成才返回
`finish_reason=stop`；断连、超时不会伪装成成功停止。usage 统计实际提交的模型
token，包含未展示的停止 token 及回收期间已提交的 token。停止边界用例代码
已补充但未执行，端到端行为仍未验收。
Chat 接受 system/user/assistant 字符串消息，最后一条必须是 user；不接受
图像、工具、思考历史、结构化输出及上述范围外的采样参数。未知字段会被拒绝，不静默忽略。
如果需要完整官方复杂模板语义，不要声称这个受限模板已经覆盖。

输入加 `max_tokens` 不得超过 `--max-context`。40960 是代码格式上限，
不是已验证容量；底层固定 arena 不会因为设置较小服务上限而全部缩小。
服务限制单个请求约 1 MiB、最多 128 条消息；HTTP 读写有超时。

Qwen 生成现已切换到插件内部的 packed 调度事件循环，非流式请求也走这条路径。
生成预算由 Lock 指定（默认 10 分钟），超时后提交原生取消并等待 drain，清理最多再等 5 分钟；
完成清理的超时请求返回 HTTP 504。截止时间在原生步骤边界检查，不能抢占正在
执行的 CUDA kernel。断连（包括客户端写半关闭）和 Worker 停止信号也会触发
协作式取消；此开发服务不支持 HTTP pipelining 或客户端输入半关闭。

需要逐步输出时，在上面的请求中设置 `"stream":true`，并使用 `curl -N`。
HTTP 返回 `text/event-stream`，每个 `data:` 帧来自原生已提交 token 的增量解码，
跨 token 不完整的 UTF-8 暂存到字节完整；最后输出停止原因、usage 和 `[DONE]`。
不是等待完整回答后拆分。流开始后的错误使用 SSE `error` JSON，然后关闭连接，
不发送成功 `[DONE]`；未开始流的参数错误仍返回 HTTP 400。
读写超时为 30 秒，慢客户端写失败会触发取消和 drain，不无限堆积输出。
原生文本契约已升级到 v2；重新构建 Worker、模型、Surface、SDK 和 Lock，
不接受旧 v1 插件或提供兼容桥。

输出回调违规是终止性错误：重复启动、非法流模式、流启动前写出、非法缓冲区
边界、回调上下文错误、SSE 数据值包含 CR/LF 或不是合法 JSON，都会锁定失败并要求取消。
成功的非流式响应也必须是非空 JSON 对象。提供者不能靠重试或
返回成功清除错误；Surface 会拒绝该契约，不发送 `[DONE]`。Linux 原生契约
测试目标已加入相关回归用例，本阶段尚未执行这些新增测试。

## 5. 留存观察结果（由你后续显式执行）

```bash
python3 deploy/native.py probe \
  --url http://127.0.0.1:8000 \
  --prompt '1+1等于多少？只回答数字。' \
  --max-tokens 32 --output /srv/evidence/qwen-first.json
```

输出目录需事先存在；文件必须不存在，避免覆盖。脚本保存请求、原始响应、HTTP
状态和端到端耗时，只有收到匹配模型的非空生成和正的 completion token 数才
返回 0；HTTP 错误、网络错误和不完整响应返回非零。它**不判定语义正确**，
不测首 token 延迟，也不发布性能/生产资格。

建议逐项检查：中文/英文是否正常；简单算术是否正确；重复请求是否污染上下文；
短输出是否按预算结束；非法 model/采样参数/过长输入是否返回 400；停止后是否
释放显存。再与同修订、同精度、同输入的官方实现比较。INT4 单独验证误差。
保留 `git rev-parse HEAD`、Lock、驱动/Toolkit、GPU UUID、模型修订/哈希、
原始回答和失败日志。不能只凭“接口返回了 200”判断改造成功。

## 6. CPU 原生分词工具

新增的 `pih-tokenize` 与 Qwen 插件共用原生 BPE 代码，也提供 DeepSeek-0731
分词格式支持。它不启动 GPU、不运行模型、不使用 Python 分词器。Linux 构建需
C++20、CMake 3.26+、OpenSSL 3 和 ICU 开发库（Ubuntu：`libicu-dev`）。

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

自行创建 UTF-8 `request.json`：编码用 `{"text":"完整的原始提示词"}`，
解码用 `{"tokens":[0,1,2]}`；标准输出为单个 JSON 对象，错误写入标准错误并
返回 2。Qwen 使用 `qwen3` 家族参数。不会自动加 BOS 或套用聊天模板；DeepSeek
解码保留结构 token，Qwen 解码跳过特殊 token 并在结束 token 处停止。
请求文件最多 1 MiB、输出最多 8 MiB、token 数最多 65536。

工具只读取本地文件，不下载、认证或发布模型制品；家族结构校验不等于固定修订
哈希校验。原生文本插件使用下文已认证的读取路径，不能把原始提示词
编码当作官方聊天编码。ICU 与官方分词器的等价性验证也尚未执行。

### DeepSeek 对话编码和输出解析

同一 CPU 构建还提供 `pih-deepseek-format`，用原生 C++ 迁移固定版本
DeepSeek-0731 编码规则，不导入或执行 Python：

```bash
out/build/tokenizer/plugins/common/pih-deepseek-format conversation.json
```

UTF-8 `conversation.json` 示例：

```json
{"operation":"encode","thinking_mode":"chat","messages":[{"role":"system","content":"简洁回答。"},{"role":"user","content":"你好"}]}
```

返回 `{"text":"..."}`，可作为 `pih-tokenize` 的输入格式。编码可选字段包括
`drop_thinking`（默认 true）、`add_bos`（默认 true）、`reasoning_effort`
（`low`/`high`/`max`，默认 low）；`thinking_mode` 为 `chat` 或 `thinking`。
实现了消息级 `tools`/`response_format`、assistant `tool_calls`、工具结果合并
排序、reminder 和任务标记。这是完整对话接口，不接受单独预编码的 context 或
前缀缓存状态；把全部历史放入 `messages`。最多 128 条消息、每组 128 个工具或
调用，序列化输入和提示词最多 1 MiB。

解析完整生成结果时，请求字段为 `operation:"parse"`、`thinking_mode` 和
`text`。`text` 必须保留原始解码中的 EOS，例如 chat 模式可用
`"Hello<｜end▁of▁sentence｜>"`；thinking 模式还要求最终内容前有 `</think>`。
输出包含 `role`、`content`、`reasoning_content`、`tool_calls`。DSML 工具调用
只解析为数据，不会被执行。截断输出、重复参数、非法非字符串 JSON 参数、尾随
内容及错误分隔符会返回 2；零参数调用允许官方渲染器输出的空参数行。
输出限制为 8 MiB。

格式实现及其生产命令已完成编译、最终链接并进入 PP1 发行组件；渲染器等价性测试、
Linux 执行和 HTTP 端到端验收尚未进行。此工具不是替代推理后端。

需要后续单独观察增量解析时，格式工具还接受：

```json
{"operation":"decode","thinking_mode":"thinking","chunks":["思考内容","</thi","nk>答案<｜end▁of▁sentence｜>"],"stopped":true}
```

返回逐段 `deltas` 和最终 `message`。每段必须已经是有效 UTF-8；分词器负责暂存
跨 token 的不完整字节。长度耗尽时使用 `stopped:false`。reasoning 和正文增量
输出，工具调用则等待完整 DSML 校验；长度上限处未完成的工具块不会作为正文或
调用输出。EOS 后不能再追加数据。离线入口最多 65536 段、请求/输出最多 8 MiB，
只为后续验证提供入口，不表示实时 HTTP 服务已经接通。

### 验证固定语义快照

认证 DeepSeek 分词需准备 `deepseek-ai/DeepSeek-V4-Flash-0731` 固定修订
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb` 的实体文件快照，包含原始字节的
`config.json`、`generation_config.json`、`tokenizer.json`、`tokenizer_config.json`、
`encoding/README.md`、`encoding/encoding_dsv4.py`、`encoding/test_encoding_dsv4.py`，
以及 `encoding/tests/test_input_N.json` / `test_output_N.txt` 的四组文件。
此操作不需要模型权重。包括快照父目录在内的全部路径组件必须是真实目录或普通
文件，符号链接（包括常见 Hub 缓存文件链接）会被拒绝。请将准确文件内容复制到
独立实体快照，不要修改 JSON 空白、换行符或源码文本。

```bash
python3 deploy/native.py verify-deepseek-semantics --build \
  --snapshot-dir /srv/models/DeepSeek-V4-Flash-0731 --jobs 4
out/build/tokenizer/plugins/common/pih-tokenize \
  deepseek-v4-flash-0731-verified /srv/models/DeepSeek-V4-Flash-0731 request.json
```

构建需要 Linux、ICU/OpenSSL 开发库，但不需要 CUDA 或 GPU。Python 命令只编排
原生构建与校验，不下载文件、不执行编码 Python、不运行模型。已有工具时省略
`--build`。需要不带编排日志的纯 JSON 元数据时，直接执行
`out/build/tokenizer/plugins/common/pih-deepseek-semantic-verify SNAPSHOT_DIR`。
成功后输出修订、闭包摘要、文件数和 token 几何；失败返回 2。实现通过禁止跟随
链接的文件描述符读取有界副本，先验证完整语义闭包，再从同一份已验证字节构造
分词器。该记录没有签名，不验证权重，也不证明编码等价性、模型或硬件资格。
该原生接口已接入文本插件加载流程，目标机验收仍未完成。

## 7. 原生离线 PP1 权重验证

已有转换后制品时，可以在启动服务前验证完整原生制品。需要 Linux CPU 环境及
C++20/CMake/Ninja/OpenSSL 开发依赖，不需要 GPU、CUDA 或 ICU：

```bash
python3 deploy/native.py verify-deepseek-artifact --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" --jobs 4
```

非零根摘要须来自独立可信的发布记录。工具读取全部已列出的权重字节，验证哈希、
模型几何、索引、运行记录、张量归属及实际 safetensors 头。目录和文件均拒绝
符号链接。完整哈希会消耗较多时间和磁盘带宽，不修改文件、不运行推理。已构建时
省略 `--build`。需要 stdout 只输出 JSON 时直接运行
`out/build/deepseek-artifact/plugins/offline-deepseek/pih-deepseek-artifact-verify GENERATION_ROOT ARTIFACT_ROOT`。
成功返回 0 并输出观察记录，失败返回 2；该记录不代表不可变存储准入、源到目标
转换等价性或硬件资格。Worker 仍须独立验证存储。本命令没有实现权重转换和发布。

### 原生 PP1 权重准备

Linux CPU 准备命令与上面的验证器使用相同构建依赖。需要独立校验过的 model、
semantic、inventory、payload 根，以及绑定已审查转换器构建的 converter identity 根。
不能编造摘要，也不能为通过检查而直接信任来源不明的根；目前尚未提供自动可信根分发。

```bash
mkdir -m 700 /srv/models/deepseek-pp1-staging
python3 deploy/native.py prepare-deepseek-artifact --build \
  --source-dir /srv/models/deepseek-original \
  --staging-dir /srv/models/deepseek-pp1-staging \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --jobs 4
```

#### 已有产物的源到目标复核

如需对已有产物进行源到目标复核，而不重新转换或写入文件，可使用只读模式：

```bash
python3 deploy/native.py verify-deepseek-artifact --build \
  --artifact-dir /srv/models/deepseek-v4-0731-pp1 \
  --artifact-root "$ARTIFACT_ROOT" --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --jobs 4
```

`--source-dir` 必须与五个源/转换器可信根一起提供，缺项时拒绝运行，不会降级为仅验证
目标。源、目标、构建目录必须互不嵌套。该模式读取完整源权重及目标，包括被 PP1 排除的
MTP 源张量，但不会修改这两个目录。成功 JSON 的 `source_payload_equivalence=true`；
仅验证目标时该字段为 `false`。两者都不是发布回执、不可变租约或模型验证结果。
所有根必须来自独立可信来源，暂不自动提供信任根。

若需 stdout 仅输出 JSON，直接运行原生程序，位置参数如下：
`pih-deepseek-artifact-verify --source-bound SOURCE_DIRECTORY GENERATION_DIRECTORY MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT`。
成功返回 0；失败返回 2，在 stderr 报错且不输出成功 JSON。

如需仓库现有格式的结构化验证结果，将原生标志改为 `--source-bound-projection`，
其余八个位置参数不变。stdout 输出规范化 ASCII JSON，**末尾没有换行**，直接对其
字节求 SHA-256 即为该结果的摘要。内容包括分片表、索引/记录/manifest 的哈希与大小、
转换/布局/处置根，以及 `verification_scope=converted_bytes_and_source_payload_non_authorizing`。
只有源到目标复核全部成功后才返回；C++ 观察结果也带有相同 JSON 字节及摘要，仅目标
验证的这两个字段为空/零。该模式不创建或发布文件。

调度脚本支持与源目录及五个根一起使用 `--verification-projection`，但 stdout 还会
打印调度日志；若要捕获规范字节，请直接调用原生程序。该结果尚不是发布回执或目录
激活记录。生产命令已完成编译和最终链接，但尚未在 Linux 执行这一准备流程。

#### 准备目录与输出要求

源目录须包含 config、索引及全部 48 个原始分片，路径不能含符号链接。暂存目录须
已存在、为空、属于当前用户，且组和其他用户不可写；放在仓库外，并与源目录、构建
目录互不嵌套。原生命令会在哈希源模型前检查暂存目录，并在写入前再次检查；这不是
目录锁，操作者须排除并发写入。输出需预留超过 156 GB 加文件系统开销的空间，原始权重保持不变。
过程会进行多次完整读取，不使用 GPU，不执行推理。脚本只调度原生程序；后者生成
源语义与 inventory、比对预期根、复制 44 个 PP1 分片、写元数据并全量复核输出。
最终复核会重新生成源语义、inventory、载荷哈希、处置规则及目标元数据，检查目标
文件集合、规范化头部，并逐张量比较目标与源载荷哈希。它复用原生规划实现，不是
另一个独立实现的数值正确性判定器；会增加完整源/目标读盘过程，仅成功后报告
`source_payload_equivalence=true`。命令不会发布、激活目录或启动服务。

### 发布存储库初始化与结构验证

Linux CPU 原生工具创建固定的 `store.json`、`.staging`、`generations`、`receipts`、
`pointers` 布局。父目录须已存在、属于当前用户，且组和其他用户不可写。新的存储库
路径必须不存在，即使是空目录或符号链接也不接受。存储库须在仓库外且与构建目录互不嵌套。

```bash
python3 deploy/native.py deepseek-generation-store init --build \
  --store-dir /srv/models/deepseek-store --jobs 4
python3 deploy/native.py deepseek-generation-store verify \
  --store-dir /srv/models/deepseek-store
```

仅输出 JSON 的原生调用为 `pih-deepseek-generation-store init|verify ABSOLUTE_STORE_DIRECTORY`。
成功返回 0，失败返回 2。初始化以 0700 创建目录、0600 创建 manifest，同步目录、文件及
父目录后重新打开验证；不会覆盖已有存储库。失败时保留部分创建结果，请检查原因，不要
盲目重试或删除模型数据，可使用新的目标路径。

结构验证只读，检查规范 manifest、精确根目录成员、无符号链接路径、归属与权限、不同且
同文件系统的子目录、条目名称和类型；每个目录最多 65,536 个条目。它**不会**验证权重
载荷、回执正文或当前指针正文，输出明确标注 `member_contents_verified=false`。调用者须
排除并发写入，持有描述符及变化检查不是目录锁或不可变租约。工具不发布权重、不激活模型。
generation-store 命令和 Worker 存储解析路径已完成编译与最终链接，但尚未对真实 Linux
存储库执行，运行验证仍待完成。下文的提交和激活命令独立于这里的结构验证操作。

### 提交已准备的产物与回执（不激活）

先初始化存储库，再创建私有空目录，例如 `/srv/models/deepseek-store/.staging/run-001`，
将上面的准备命令 `--staging-dir` 指向它。不要手工把未验证目录放进 `generations`。
准备成功后执行：

```bash
python3 deploy/native.py deepseek-generation-store commit --build \
  --store-dir /srv/models/deepseek-store --staging-name run-001 \
  --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" \
  --artifact-root "$ARTIFACT_ROOT" --jobs 4
```

命令重新执行源到目标复核，以原子不覆盖方式移至 `generations/sha256-<artifact-root>`，
将产物文件封存为 0444、目录设为 0555，然后再次执行源到目标复核。移动前后验证结果
完全相同，才独占创建 `receipts/<artifact-root>.json`，读回复核后设为 0444 并同步。
目标产物或回执已存在时拒绝执行。过程需要多次完整读盘，操作者须排除源及存储库并发写入。

原生位置参数接口为
`pih-deepseek-generation-store commit SOURCE_DIRECTORY STORE_DIRECTORY STAGING_NAME MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT`。
返回 0 要求 `success=true`、`receipt_committed=true`，但**不会激活**。失败返回 2，可能
同时输出 `success=false` 的结果 JSON；请检查 `renamed`、`directories_synced`、
`read_only_sealed`、`receipt_may_exist`。它们只表示已完成或可能开始的阶段，不代表整体成功。
仅有非零 receipt 根不能证明发布成功。已移动的目录及部分回执都会保留，没有自动重试、
回滚、覆盖或清理；源准入失败可能发生在结果生成之前。命令不更新 `pointers/current.json`，
不执行模型、不提供不可变租约。本地尚未验证真实提交及崩溃恢复行为。

### 激活已提交产物

激活会重新复核源到目标载荷一致性及已封存回执。使用提交成功时返回的 receipt 根，以及
独立准入的 catalog 根；本命令只绑定 catalog 摘要，不解析、认证或安装 catalog 正文。
首次激活须明确声明没有前一指针：

```bash
python3 deploy/native.py deepseek-generation-store activate --build \
  --store-dir /srv/models/deepseek-store --source-dir /srv/models/deepseek-original \
  --model-root "$MODEL_ROOT" --semantic-root "$SEMANTIC_ROOT" \
  --inventory-root "$INVENTORY_ROOT" --payload-root "$PAYLOAD_ROOT" \
  --converter-identity-root "$CONVERTER_IDENTITY_ROOT" --artifact-root "$ARTIFACT_ROOT" \
  --receipt-root "$RECEIPT_ROOT" --catalog-root "$CATALOG_ROOT" \
  --activation-ordinal 1 --previous-pointer-root none --jobs 4
```

后续激活须提供预期当前指针根，序号恰好加 1，最大 INT64_MAX；过期前驱、跳号、溢出都会
在修改指针前拒绝。提交和激活对存储库根目录使用同一把非阻塞协作锁，忙碌时直接失败。
这不能排除不遵守锁的外部写入，操作者仍须独占管理。复核可能再次完整读取源和目标权重。

原生接口为 `pih-deepseek-generation-store activate SOURCE_DIRECTORY STORE_DIRECTORY MODEL_ROOT SEMANTIC_ROOT INVENTORY_ROOT PAYLOAD_ROOT CONVERTER_IDENTITY_ROOT ARTIFACT_ROOT RECEIPT_ROOT CATALOG_ROOT ORDINAL PREVIOUS_POINTER_ROOT_OR_none`。
工具独占写入并读回 `.current-<ordinal>-<root>.tmp`，再原子安装 `current.json`，首次安装
禁止覆盖；同步 pointers/store 目录后重读指针。成功须返回 0、`success=true`、`activated=true`。
失败时仍可能有 `pointer_replaced=true`，请结合 `temporary_may_exist`、`directories_synced`
判断已完成阶段。遗留临时文件会使结构验证拒绝通过，须检查处理；不会自动回滚或删除。
只有根摘要不能证明激活完成。

这仅切换存储库指针，不重新加载运行中的 Worker，不验证 catalog 正文，不提供不可变
存储准入，也不执行模型。生产命令与 Worker 接线已完成编译和最终链接，但没有执行
真实 Linux 激活；实际运行、硬件及崩溃恢复仍待验证。

### 解析并验证当前激活产物

不保留原始 checkpoint 时，可用以下只读命令核对已激活存储库：

```bash
python3 deploy/native.py deepseek-generation-store resolve --build \
  --store-dir /srv/models/deepseek-store \
  --pointer-root "$POINTER_ROOT" --catalog-root "$CATALOG_ROOT" --jobs 4
```

两个根须来自独立可信的激活/catalog 决策。原生工具在共享协作锁下读取当前指针，按预期
根解析封存回执，将完整对象表与 artifact manifest 对齐，并哈希全部目标文件，同时检查
精确成员集合、封存权限、归属、规范元数据、指针及路径稳定性。不重新扫描源载荷，
也不准入 catalog 正文。需要预留完整目标权重的读盘时间；与原生提交/激活冲突时忙碌失败，
不等待。命令不修改文件。

仅输出 JSON 的调用为 `pih-deepseek-generation-store resolve STORE_DIRECTORY EXPECTED_POINTER_ROOT EXPECTED_CATALOG_ROOT`。
成功返回 0，包含 `generation_name`、artifact/receipt/pointer/catalog 根和激活序号，明确
标注 `receipt_binding_verified=true`、`source_payload_equivalence=false`。失败返回 2，
不输出成功观察结果。目标目录为 `STORE_DIRECTORY/generations/<generation_name>`，但这
不是持有文件描述符的租约，不能绕过 Worker 的产物/存储准入。`serve-deepseek --store-dir`
已将结果接入原生 Worker 启动，仍保留 Worker 准入检查。该路径已完成编译和最终链接，
但尚未使用真实存储库执行。

## 8. 故障定位与迁移边界

压缩器现在直接读取 checkpoint 的两块独立 BF16 权重。开发中的 Kernel Pack
请求布局已改变：模型和 Pack 必须一起重建，并重新生成 Lock；旧融合权重请求
会被精确结构体大小校验拒绝。Indexer query 投影现已绑定原始 E4M3 权重
`[8192,1024]` 与 UE8M0 scale `[64,8]`，先量化 BF16 query 输入，再调用现有
FP8 GEMM 和 RoPE。该请求布局也有变化，模型、Pack、Lock 需要一起重建。
两个独立 scratch buffer 每个最大 query 增加 1,032 字节（未计分配器对齐）。
CUDA 编译和数值执行仍未验证；共享专家等原生权重消费路径仍待运行验证，
制品有效不代表服务已能执行所有权重。

共享专家绑定已能解析主层的六个原始 E4M3/UE8M0 张量，并检查精确形状及统一的
CUDA 设备、分配代次。rank 专家资源现提供不增加分配的共享计算 scratch 视图，
只能在全部路由专家完成后复用。启动器的共享专家资源绑定现已实现，
但尚未在 GPU 上验证。其 NVIDIA 适配器已改为
必须提供 Backend async 与 Kernel Pack 能力，不再直接回退调用 CUDA。
能力的所有者必须比适配器存活更久，适配器由共享运行时资源对象持有。
共享驱动会排入错误回读与完成事件，轮询时拒绝未完成或失败后的复用。
标准计算栈已接入“路由专家完成 → 共享专家计算与合并 → 共享错误/事件完成”的
包装器。未绑定共享 provider 时，推理在提交前失败，drain 清理仍可执行。
只能在启动阶段绑定，不能替换已绑定 provider。堆上稳定的资源所有者持有各层
驱动及 pinned 错误缓冲区，并拒绝并发复用 scratch。CUDA 启动器现已创建该所有者
与适配器并附加到 rank 基础设施；host-spill 和全驻留 lane 都会绑定 provider。
它使用 mHC 本层输入/FFN 输出、专家输入/累加器及 rank 的流/事件。部分提交失败时，外围运行时
必须等待流完成清理后才能释放资源。路由专家提交前，阶段包装器现在要求先在
同一流上复制 mHC 本层输入到路由/共享输入缓冲区，再按实际 packed token 数
清零 FP32 累加器。Backend 适配器拒绝地址范围重叠和溢出。准备失败时禁止路由
提交，并使该所有者/阶段永久失效。以上是代码装配完成，不是 CUDA 链接或推理
正确性的证据。尚未运行模型测试。

- 缺少 ICU/OpenSSL/CUDA：修正开发库和工具链路径，重新配置；不是模型问题。
- 构建链接失败：保存完整日志。生产构建已完成；新的构建失败需要根据日志诊断，不能归咎于
  用户环境或靠重新开启 monolith/Python 绕过。

构建目录与安装目录必须独立且互不包含，不能使用仓库根目录或其上级目录；
允许使用 `out/` 下的两个并列目录。Qwen 模型目录必须与源码、构建、安装目录
完全分离，`serve` 会在构建和安装之前检查。检查使用解析后的路径，防止已有
符号链接掩盖重叠；这不防御其他进程在检查后并发替换路径。
- GPU 身份不符：更换正确目标机，不改 SM 标识或重命名 cubin。
- 模型/分词器格式拒绝：确认是 Qwen3-0.6B 原始 BF16 或 PIH INT4，不跳过校验。
- 二进制变动导致 Lock 摘要不匹配：重新运行启动脚本生成新 Lock，不编辑摘要。
- HTTP 500：此次引擎已视为失败，Worker 退出清理后重新启动；保存失败日志。
