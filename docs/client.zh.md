# Python 客户端

[English](client.md) | 中文

流式客户端遵循原生单结果契约：每个生成事件恰好一个结果，索引必须为整数 `0`，
结束原因出现时必须为非空字符串。结束事件后不得再发送生成结果，只允许至多一个
带 usage 的空 choices 尾部事件，并且仍须以 `[DONE]` 结束。提前断开、非法结果或
结束后继续生成会抛出 `ProtocolError` 并关闭连接。这只是传输校验，不保证模型
内容正确，也不表示服务端资源已完成回收。

非流式响应同样要求一个索引为 `0`、带结束原因的结果，并按接口包含文本或 assistant
消息。成功 HTTP 状态下的错误对象也会被拒绝。两种请求模式均只支持整数 `n=1`。
响应 JSON 拒绝重复键、非有限数值常量，以及 `1e999` 这类浮点溢出，不会悄悄转换为
无穷大。

在仓库根目录执行 `python -m pip install .`，安装的纯 Python wheel 只包含
`pih_client`，不再构建 `_pih` 或提供历史 `pih.Engine`。客户端机器不需要
C++ 编译器、CUDA、PyTorch、tokenizer 或模型权重。服务端需按
[原生推理指南](native-inference.zh.md) 独立构建并启动。

```python
from pih_client import Client

client = Client("http://127.0.0.1:8000", timeout=600)
print(client.ready())
print(client.models())
result = client.chat(
    [{"role": "user", "content": "请用一句话解释张量并行。"}],
    model="Qwen/Qwen3-0.6B", max_tokens=128,
)
print(result["choices"][0]["message"]["content"])
```

地址只填 origin，不加 `/v1`。原始文本补全使用 `complete(prompt, ...)`。
返回字典保留服务端用量与停止原因；客户端不加载模型、不自动启动其他后端。
额外采样参数由服务端验证，当前 Qwen 只实现贪心、单序列文本。
SSE 流式接口为 `stream_chat` 和 `stream_complete`：

```python
with client.stream_chat([{"role": "user", "content": "从一数到五。"}]) as stream:
    for chunk in stream:
        for choice in chunk["choices"]:
            print(choice.get("delta", {}).get("content", ""), end="", flush=True)
```

迭代器校验有大小限制的 UTF-8 JSON 事件，只有收到停止原因和 `[DONE]` 才认为
完整结束；断流不是成功。原生服务的最后一个 chunk 包含 usage。提前退出需关闭
stream/context。CLI 使用 `pih-client chat '从一数到五。' --stream`，逐行输出 JSON
chunk；错误流不会发送表示成功的 `[DONE]`。

```bash
pih-client --url http://127.0.0.1:8000 ready
pih-client --url http://127.0.0.1:8000 models
pih-client --timeout 600 chat '1+1等于多少？' --max-tokens 32
pih-preflight-target-host --hardware-profile rtx4090d --devices 0
```

如果部署了带鉴权的反向代理，可传 `api_key=`，CLI 可使用 `PIH_API_KEY`。
非本地 API key 必须经 HTTPS 发送；不会关闭 TLS 校验、跟随重定向、读取代理
环境变量或自动重试生成。当前内置开发 HTTP 服务没有鉴权/TLS，应保持回环监听。
主机预检仍只是前置诊断，不授予生产资格。

每次请求独占一个连接，成功和异常都会关闭。`timeout` 是 socket 无活动超时，
不是 GPU 执行截止时间；关闭流后，原生服务在下一个调度边界提交协作式取消，
客户端无法确认 GPU drain 已完成，也不能抢占执行中的 kernel。
非 2xx 返回抛出 `HTTPError`（含 `status`、`body`），超限或非法 JSON 抛出
`ProtocolError`，网络失败保留标准库异常。请求限制 1 MiB，响应默认限制 8 MiB。
API key 不写入模型配置或部署 Lock。

旧 `pih` 导入与推理/离线 console scripts 不再随 wheel 安装，没有兼容别名。
Qwen INT4 转换改用原生 `pih-qwen-int4-convert`。其他历史离线绑定仍待拆分，
不能把仓库中仍存留的源文件算作这个客户端包已提供的功能。
