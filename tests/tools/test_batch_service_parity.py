import importlib.util
from pathlib import Path
import threading

spec = importlib.util.spec_from_file_location("batch_parity", Path(__file__).parents[2] / "tools/verify_batch_service.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


def test_parity_submits_variable_prompts_concurrently_and_compares_each_result():
    gate = threading.Barrier(4)
    calls = 0
    lock = threading.Lock()
    def complete(prompt, seed):
        nonlocal calls
        with lock:
            calls += 1
            concurrent = calls > 4
        if concurrent:
            gate.wait(timeout=2)
        return {"choices": [{"index": 0, "text": prompt + str(seed), "finish_reason": "length"}],
                "usage": {"prompt_tokens": len(prompt), "completion_tokens": 3}}
    report = parity.compare(complete, ["a", "bb", "ccc", "dddd"], seed=7)
    assert calls == 8
    assert report["all_equal"] is True
    assert report["batch_size"] == 4
    assert report["execution_plan_verified"] is False


def test_parity_reports_sequence_specific_difference():
    count = 0
    def complete(prompt, seed):
        nonlocal count
        count += 1
        text = "changed" if count == 4 else prompt
        return {"choices": [{"index": 0, "text": text, "finish_reason": "stop"}], "usage": {}}
    report = parity.compare(complete, ["a", "bb"], seed=7)
    assert report["all_equal"] is False
    assert sum(not item["equal"] for item in report["sequences"]) == 1
