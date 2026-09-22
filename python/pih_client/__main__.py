from __future__ import annotations

import argparse
import http.client
import json
import os
import sys

from . import Client, HTTPError, ProtocolError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Client for an already-running native PIH worker")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=600)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("ready")
    commands.add_parser("models")
    for name in ("chat", "complete"):
        request = commands.add_parser(name)
        request.add_argument("prompt")
        request.add_argument("--model", default="Qwen/Qwen3-0.6B")
        request.add_argument("--max-tokens", type=int, default=128)
        request.add_argument("--stream", action="store_true")
    args = parser.parse_args(argv)
    try:
        client = Client(args.url, api_key=os.environ.get("PIH_API_KEY"), timeout=args.timeout)
        if args.command in {"chat", "complete"} and args.stream:
            operation = client.stream_chat if args.command == "chat" else client.stream_complete
            prompt = [{"role": "user", "content": args.prompt}] if args.command == "chat" else args.prompt
            with operation(prompt, model=args.model, max_tokens=args.max_tokens) as stream:
                for chunk in stream:
                    print(json.dumps(chunk, ensure_ascii=False), flush=True)
            return 0
        if args.command == "ready":
            result = {"ready": client.ready()}
        elif args.command == "models":
            result = client.models()
        elif args.command == "chat":
            result = client.chat([{"role": "user", "content": args.prompt}],
                                 model=args.model, max_tokens=args.max_tokens)
        else:
            result = client.complete(args.prompt, model=args.model, max_tokens=args.max_tokens)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if args.command == "ready" and not result["ready"] else 0
    except (ValueError, OSError, http.client.HTTPException, HTTPError, ProtocolError) as error:
        print(f"pih-client: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
