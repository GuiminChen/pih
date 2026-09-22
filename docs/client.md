# Python client

English | [中文](client.zh.md)

Streaming targets the native single-choice contract: exactly one choice with
integer index `0`, and a nonempty string when a finish reason is present. No
further choices are accepted after that finish event. At most one usage-only
empty-choice event may follow it; `[DONE]` is still required. Premature EOF,
malformed choices, or post-finish generation raise `ProtocolError` and close the
connection. This is transport validation, not a guarantee of valid model content
or completed server-side resource retirement.

Non-streaming completions likewise require one terminal choice at index `0`,
with text or an assistant message as appropriate. Error objects under a successful
HTTP status are rejected. Both request modes support only integer `n=1`. Response
JSON rejects duplicate keys, non-finite constants, and numeric overflow such as
`1e999`; it does not silently turn these values into infinity.

Install from the repository with `python -m pip install .`. This produces a pure
Python wheel containing `pih_client`, not `_pih` or the historical `pih.Engine`.
No C++ compiler, CUDA toolkit, PyTorch, tokenizer or model weights are required
on the client machine. Build and start the server separately using the
[native inference guide](native-inference.md).

```python
from pih_client import Client

client = Client("http://127.0.0.1:8000", timeout=600)
print(client.ready())
print(client.models())
result = client.chat(
    [{"role": "user", "content": "Explain tensor parallelism in one sentence."}],
    model="Qwen/Qwen3-0.6B", max_tokens=128,
)
print(result["choices"][0]["message"]["content"])
```

The base URL is an origin without `/v1`. `complete(prompt, ...)` provides raw
text completion. Responses are dictionaries, preserving the server's usage and
finish reason. This client does not choose a local model or start a fallback.
Parameters are sent to the server for validation; current Qwen supports only
greedy, single-sequence text. Use `stream_chat` or `stream_complete` for SSE:

```python
with client.stream_chat([{"role": "user", "content": "Count to five."}]) as stream:
    for chunk in stream:
        for choice in chunk["choices"]:
            print(choice.get("delta", {}).get("content", ""), end="", flush=True)
```

The iterator validates bounded UTF-8 JSON events and requires both a finish reason
and `[DONE]`; a truncated connection is an error, not a successful completion.
The final native chunk includes usage. Close the stream/context when stopping
early. CLI streaming uses `pih-client chat 'Count to five.' --stream` and prints
one JSON chunk per line. Error streams terminate without a successful `[DONE]`.

```bash
pih-client --url http://127.0.0.1:8000 ready
pih-client --url http://127.0.0.1:8000 models
pih-client --timeout 600 chat 'What is 1+1?' --max-tokens 32
pih-preflight-target-host --hardware-profile rtx4090d --devices 0
```

For an authenticated reverse proxy, pass `api_key=` or set `PIH_API_KEY` for the
CLI. Remote API keys require HTTPS. TLS verification remains enabled. The client
does not follow redirects, read proxy environment variables, or retry generation.
The built-in development HTTP surface itself has no authentication or TLS and
must remain on loopback. Host diagnostics do not grant production qualification.

Each request owns and closes one connection, including on errors. `timeout`
is a socket inactivity timeout, not a GPU deadline. Closing a stream disconnects
the native surface, which cooperatively cancels at the next scheduler boundary;
the client cannot acknowledge completion of GPU drain or preempt a kernel. Non-2xx
responses raise `HTTPError` (`status`, `body`); invalid/oversized JSON raises
`ProtocolError`; transport failures retain standard-library exceptions.
Requests are bounded to 1 MiB; responses default to 8 MiB. API keys are not stored
in model configuration or deployment Locks.

The historical `pih` import and its inference/offline console commands are no
longer installed. There are no import aliases. Qwen INT4 conversion uses the
native `pih-qwen-int4-convert` tool. Other offline bindings remain to be extracted
from historical source and are not claimed as part of this client distribution.
