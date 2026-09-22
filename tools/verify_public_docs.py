from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
FORBIDDEN_RE = re.compile(
    r"XingInfer|Codex|ChatGPT|superpowers|development slice|P0/P1|"
    r"writing prompts|collaboration instructions|"
    r"[A-Za-z]:[\\/](?:Users|github|ProgramData|AppData)[\\/]",
    re.IGNORECASE,
)


def markdown_files() -> list[Path]:
    # Public entry points only. Historical design records contain local build
    # provenance; scanning generated build trees is neither bounded nor useful.
    paths = list(ROOT.glob("*.md")) + list(DOCS.glob("*.md"))
    paths.extend((ROOT / "examples").rglob("*.md"))
    return sorted(paths)


def main() -> int:
    errors: list[str] = []
    docs = sorted(DOCS.glob("*.md"))
    english = [path for path in docs if not path.stem.endswith(".zh")]
    chinese = [path for path in docs if path.stem.endswith(".zh")]

    for name in ("index", "architecture", "development", "plugins", "deployment", "support"):
        for suffix in (".md", ".zh.md"):
            if not (DOCS / (name + suffix)).is_file():
                errors.append(f"missing public guide: docs/{name}{suffix}")

    for path in english:
        peer = path.with_name(f"{path.stem}.zh.md")
        if not peer.is_file():
            errors.append(f"missing Chinese pair: {path.relative_to(ROOT)}")
            continue
        expected = f"English | [中文]({peer.name})"
        if expected not in path.read_text(encoding="utf-8"):
            errors.append(f"missing language link: {path.relative_to(ROOT)}")

    for path in chinese:
        name = path.name.removesuffix(".zh.md") + ".md"
        peer = path.with_name(name)
        if not peer.is_file():
            errors.append(f"missing English pair: {path.relative_to(ROOT)}")
            continue
        expected = f"[English]({peer.name}) | 中文"
        if expected not in path.read_text(encoding="utf-8"):
            errors.append(f"missing language link: {path.relative_to(ROOT)}")

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        match = FORBIDDEN_RE.search(text)
        if match:
            errors.append(
                f"private/process content in {path.relative_to(ROOT)}: {match.group(0)!r}"
            )
        for match in LINK_RE.finditer(text):
            link = match.group(1)
            if link.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative = unquote(link.split("#", 1)[0])
            if relative and not (path.parent / relative).resolve().exists():
                errors.append(f"broken link in {path.relative_to(ROOT)}: {link}")

    if errors:
        print("public documentation verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"public documentation verification passed: {len(english)} bilingual pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
