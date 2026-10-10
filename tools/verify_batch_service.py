"""Compare deterministic serial/concurrent HTTP outputs on a running native model.

This checks numerical output parity, not GPU plan membership or performance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request


def signature(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("expected exactly one completion per sequence")
    choice = choices[0]
    if choice.get("index") != 0 or not isinstance(choice.get("text"), str) or choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("malformed or unfinished completion")
    usage = response.get("usage", {})
    return {"text": choice["text"], "finish_reason": choice["finish_reason"],
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens")}


def compare(complete, prompts, seed=7):
    if not 1 <= len(prompts) <= 32 or any(not isinstance(p, str) or not p for p in prompts):
        raise ValueError("provide 1..32 nonempty prompts")
    baseline = [signature(complete(prompt, seed + index)) for index, prompt in enumerate(prompts)]
    with ThreadPoolExecutor(max_workers=len(prompts)) as workers:
        futures = [workers.submit(complete, prompt, seed + index) for index, prompt in enumerate(prompts)]
        observed = [signature(future.result()) for future in futures]
    sequences = [{"index": index, "equal": expected == actual,
                  "serial": expected, "concurrent": actual}
                 for index, (expected, actual) in enumerate(zip(baseline, observed))]
    return {"schema": "pih.batch-output-parity.v1", "batch_size": len(prompts),
            "all_equal": all(s["equal"] for s in sequences),
            "execution_plan_verified": False, "sequences": sequences}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, choices=range(1, 33), default=4)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    def complete(prompt, seed):
        payload = json.dumps({"model": args.model, "prompt": prompt, "max_tokens": args.max_tokens,
                              "temperature": 0, "top_p": 1, "seed": seed}).encode()
        request = urllib.request.Request(args.url.rstrip("/") + "/v1/completions", payload,
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = response.read(8 * 1024 * 1024 + 1)
            if len(data) > 8 * 1024 * 1024:
                raise ValueError("response exceeds byte budget")
            return json.loads(data)
    prompts = ["Explain matrix multiplication." + " Give an example." * index
               for index in range(args.batch_size)]
    report = compare(complete, prompts, args.seed)
    report["model"] = args.model
    # Never overwrite an earlier hardware observation.
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"all_equal": report["all_equal"], "batch_size": args.batch_size,
                      "execution_plan_verified": False}))
    return 0 if report["all_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
