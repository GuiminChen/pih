"""In-memory transport cases; no service or model execution."""

import io
import json

import pytest

from pih_client import ProtocolError, Stream


class Connection:
    closed = False

    def close(self):
        self.closed = True


def event(choices, **fields):
    value = {"model": "test-model", "choices": choices, **fields}
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def stream(payload):
    connection = Connection()
    return Stream(connection, io.BytesIO(payload), "test-model", 4096), connection


def test_finish_then_done_closes_transport():
    iterator, connection = stream(event([{"index": 0, "finish_reason": "stop"}]) + b"data: [DONE]\n\n")
    next(iterator)
    with pytest.raises(StopIteration):
        next(iterator)
    assert connection.closed


def test_choices_after_finish_are_rejected():
    iterator, connection = stream(event([{"index": 0, "finish_reason": "stop"}]) +
                                  event([{"index": 0, "finish_reason": None}]))
    next(iterator)
    with pytest.raises(ProtocolError):
        next(iterator)
    assert connection.closed


@pytest.mark.parametrize("choices", [[], [{}], [{"index": True}], [{"index": 1}],
                                     [{"index": 0, "finish_reason": False}],
                                     [{"index": 0}, {"index": 1}]])
def test_invalid_choice_does_not_establish_success(choices):
    iterator, connection = stream(event(choices))
    with pytest.raises(ProtocolError):
        next(iterator)
    assert connection.closed


def test_usage_only_tail_does_not_replace_done():
    iterator, connection = stream(event([{"index": 0, "finish_reason": "length"}]) +
                                  event([], usage={"completion_tokens": 1}))
    next(iterator)
    next(iterator)
    with pytest.raises(ProtocolError, match="without"):
        next(iterator)
    assert connection.closed
