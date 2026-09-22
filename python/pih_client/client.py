"""Bounded, synchronous HTTP transport; no redirects, proxies, or retries."""
from __future__ import annotations

import http.client
import json
import math
import ssl
from typing import Any
from urllib.parse import urlsplit


class ProtocolError(RuntimeError):
    """The peer did not return the expected bounded JSON protocol."""


class HTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"PIH server returned HTTP {status}: {body[:2048]}")


class Stream:
    """Bounded SSE iterator. Use as a context manager or explicitly close it.

    Closing disconnects the native surface, which cooperatively cancels at its
    next scheduler boundary. It is not an acknowledgement of completed GPU drain.
    """

    def __init__(self, connection: http.client.HTTPConnection,
                 response: http.client.HTTPResponse, model: str, limit: int):
        self._connection, self._response = connection, response
        self._model, self._limit = model, limit
        self._received = 0
        self._closed = False
        self._finished = False
        self._usage_tail = False

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._response.close()
            finally:
                self._connection.close()

    def __iter__(self) -> Stream:
        return self

    def __next__(self) -> dict[str, Any]:
        if self._closed:
            raise StopIteration
        try:
            data: list[bytes] = []
            while True:
                line = self._response.readline(min(self._limit - self._received + 1, 1024 * 1024 + 1))
                if not line:
                    raise ProtocolError("stream ended without [DONE]")
                self._received += len(line)
                if self._received > self._limit or len(line) > 1024 * 1024:
                    raise ProtocolError("SSE response exceeds configured byte limit")
                if not line.endswith(b"\n"):
                    raise ProtocolError("unterminated SSE line")
                line = line.rstrip(b"\r\n")
                if not line:
                    if not data:
                        continue
                    payload = b"\n".join(data)
                    if payload == b"[DONE]":
                        if not self._finished:
                            raise ProtocolError("stream completed without a finish reason")
                        self.close()
                        raise StopIteration
                    value = _json_object(payload)
                    if "error" in value:
                        raise ProtocolError(f"server terminated stream: {value['error']}")
                    if value.get("model") != self._model or not isinstance(value.get("choices"), list):
                        raise ProtocolError("stream model/choices mismatch")
                    choices = value["choices"]
                    if not choices:
                        # At most one usage-only tail after the terminal choice.
                        if not self._finished or self._usage_tail or not isinstance(value.get("usage"), dict):
                            raise ProtocolError("unexpected empty-choice stream event")
                        self._usage_tail = True
                        return value
                    if self._finished:
                        raise ProtocolError("stream emitted choices after its finish reason")
                    if len(choices) != 1 or not isinstance(choices[0], dict):
                        raise ProtocolError("native stream requires exactly one choice")
                    choice = choices[0]
                    if type(choice.get("index")) is not int or choice["index"] != 0:
                        raise ProtocolError("native stream choice index must be zero")
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        if not isinstance(finish, str) or not finish:
                            raise ProtocolError("invalid stream finish reason")
                        self._finished = True
                    return value
                if line.startswith(b":"):
                    continue
                field, separator, value = line.partition(b":")
                if field == b"data":
                    if separator and value.startswith(b" "):
                        value = value[1:]
                    data.append(value)
        except BaseException:
            self.close()
            raise


def _json_object(data: bytes) -> dict[str, Any]:
    def constant(value: str) -> None:
        raise ProtocolError(f"non-finite JSON number: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ProtocolError("JSON number overflows finite floating-point range")
        return number

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProtocolError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        value = json.loads(data.decode("utf-8"), parse_constant=constant,
                           parse_float=finite_float, object_pairs_hook=unique)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ProtocolError("response is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ProtocolError("response must be a JSON object")
    return value


class Client:
    """One connection per request; closing it does not promise GPU cancellation.

    ``base_url`` is an origin, not an OpenAI ``/v1`` URL. API keys are sent only
    to that exact origin; redirects and environment proxy settings are ignored.
    Never retries a generation request automatically.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", *,
                 api_key: str | None = None, timeout: float = 600,
                 max_response_bytes: int = 8 * 1024 * 1024):
        url = urlsplit(base_url)
        if (url.scheme not in {"http", "https"} or not url.hostname or
                url.username is not None or url.password is not None or
                url.path not in {"", "/"} or url.query or url.fragment or
                any(ord(char) <= 32 for char in base_url)):
            raise ValueError("base_url must be an HTTP(S) origin without credentials or path")
        port = url.port
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("invalid server port")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or not 0 < timeout <= 3600:
            raise ValueError("timeout must be finite and between 0 and 3600 seconds")
        if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_response_bytes must be 1 KiB..64 MiB")
        if api_key is not None and (not isinstance(api_key, str) or not api_key or
                any(ord(char) < 33 or ord(char) > 126 for char in api_key)):
            raise ValueError("API key must contain printable non-whitespace ASCII")
        if api_key and url.scheme != "https" and url.hostname not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("remote API keys require HTTPS")
        self._scheme, self._host, self._port = url.scheme, url.hostname, port
        self._api_key, self._timeout, self._limit = api_key, timeout, max_response_bytes

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 *, stream_model: str | None = None) -> dict[str, Any] | Stream:
        headers = {"Accept": "text/event-stream" if stream_model else "application/json", "Connection": "close"}
        encoded = None
        if body is not None:
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            if len(encoded) > 1024 * 1024:
                raise ValueError("request exceeds 1 MiB")
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._scheme == "https":
            connection = http.client.HTTPSConnection(self._host, self._port, timeout=self._timeout,
                                                     context=ssl.create_default_context())
        else:
            connection = http.client.HTTPConnection(self._host, self._port, timeout=self._timeout)
        handed_off = False
        response = None
        try:
            connection.request(method, path, body=encoded, headers=headers)
            response = connection.getresponse()
            if stream_model and 200 <= response.status < 300:
                if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
                    raise ProtocolError("server did not return text/event-stream")
                result = Stream(connection, response, stream_model, self._limit)
                handed_off = True
                return result
            data = response.read(self._limit + 1)
            if len(data) > self._limit:
                raise ProtocolError("response exceeds configured byte limit")
            if not 200 <= response.status < 300:
                raise HTTPError(response.status, data.decode("utf-8", errors="replace"))
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != "application/json":
                raise ProtocolError("server did not return application/json")
            return _json_object(data)
        finally:
            if not handed_off:
                try:
                    if response is not None:
                        response.close()
                finally:
                    connection.close()

    def ready(self) -> bool:
        """Readiness is availability, not numerical or production qualification."""
        result = self._request("GET", "/readyz")
        if type(result.get("ready")) is not bool:
            raise ProtocolError("readiness response is missing a boolean ready field")
        return result["ready"]

    def models(self) -> dict[str, Any]:
        return self._request("GET", "/v1/models")

    def complete(self, prompt: str, *, model: str = "Qwen/Qwen3-0.6B",
                 max_tokens: int = 128, **parameters: Any) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        return self._completion("/v1/completions", {"prompt": prompt}, model, max_tokens, parameters)

    def chat(self, messages: list[dict[str, str]], *, model: str = "Qwen/Qwen3-0.6B",
             max_tokens: int = 128, **parameters: Any) -> dict[str, Any]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        return self._completion("/v1/chat/completions", {"messages": messages}, model, max_tokens, parameters)

    def stream_chat(self, messages: list[dict[str, str]], *, model: str = "Qwen/Qwen3-0.6B",
                    max_tokens: int = 128, **parameters: Any) -> Stream:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        return self._completion("/v1/chat/completions", {"messages": messages}, model, max_tokens, parameters, stream=True)

    def stream_complete(self, prompt: str, *, model: str = "Qwen/Qwen3-0.6B",
                        max_tokens: int = 128, **parameters: Any) -> Stream:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a nonempty string")
        return self._completion("/v1/completions", {"prompt": prompt}, model, max_tokens, parameters, stream=True)

    def _completion(self, path: str, payload: dict[str, Any], model: str,
                    max_tokens: int, parameters: dict[str, Any], *, stream: bool = False) -> dict[str, Any] | Stream:
        if not isinstance(model, str) or not model or type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("model and positive integer max_tokens are required")
        if parameters.keys() & {"model", "max_tokens", "prompt", "messages", "stream"}:
            raise ValueError("reserved request fields cannot be overridden; use stream_chat/stream_complete")
        if "n" in parameters and (type(parameters["n"]) is not int or parameters["n"] != 1):
            raise ValueError("native client supports exactly n=1")
        payload.update({"model": model, "max_tokens": max_tokens, "temperature": 0, "top_p": 1,
                        "stream": stream})
        payload.update(parameters)
        result = self._request("POST", path, payload, stream_model=model if stream else None)
        if stream:
            return result
        if "error" in result:
            raise ProtocolError("server returned an error object with a successful HTTP status")
        if result.get("model") != model or not isinstance(result.get("choices"), list) or len(result["choices"]) != 1:
            raise ProtocolError("completion response model/choices mismatch")
        choice = result["choices"][0]
        if (not isinstance(choice, dict) or type(choice.get("index")) is not int or
                choice["index"] != 0 or not isinstance(choice.get("finish_reason"), str) or
                not choice["finish_reason"]):
            raise ProtocolError("completion requires a terminal choice at index zero")
        if path == "/v1/completions":
            if not isinstance(choice.get("text"), str):
                raise ProtocolError("completion choice is missing text")
        else:
            message = choice.get("message")
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ProtocolError("chat completion is missing an assistant message")
            content = message.get("content")
            if not isinstance(content, str) and not (content is None and isinstance(message.get("tool_calls"), list)
                                                    and message["tool_calls"]):
                raise ProtocolError("assistant message is missing content or tool calls")
        return result
