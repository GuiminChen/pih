"""Stubbed transport contract cases, without network or model execution."""

import pytest

from pih_client import Client, ProtocolError
from pih_client.client import _json_object


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_rejects_nonfinite_and_overflowed_json_numbers(number):
    with pytest.raises(ProtocolError):
        _json_object(('{"value":' + number + '}').encode())


@pytest.mark.parametrize("choice", [None, {}, {"index": True, "finish_reason": "stop", "text": "x"},
    {"index": 0, "finish_reason": None, "text": "x"},
    {"index": 0, "finish_reason": "stop"}])
def test_rejects_nonterminal_or_malformed_completion(monkeypatch, choice):
    client = Client()
    monkeypatch.setattr(client, "_request", lambda *a, **k: {"model": "test", "choices": [choice]})
    with pytest.raises(ProtocolError):
        client.complete("prompt", model="test")


def test_accepts_empty_but_terminal_text(monkeypatch):
    client = Client()
    response = {"model": "test", "choices": [{"index": 0, "finish_reason": "stop", "text": ""}]}
    monkeypatch.setattr(client, "_request", lambda *a, **k: response)
    assert client.complete("prompt", model="test") == response


def test_rejects_success_status_error_object(monkeypatch):
    client = Client()
    monkeypatch.setattr(client, "_request", lambda *a, **k: {"error": "failed"})
    with pytest.raises(ProtocolError, match="error object"):
        client.complete("prompt")


@pytest.mark.parametrize("n", [True, 0, 2, 1.0])
def test_rejects_unsupported_n_before_transport(monkeypatch, n):
    client = Client()
    monkeypatch.setattr(client, "_request", lambda *a, **k: pytest.fail("transport must not run"))
    with pytest.raises(ValueError, match="n=1"):
        client.complete("prompt", n=n)
