#!/usr/bin/env python3
"""
CI Code Review Script — uses Anthropic Claude to review PR diffs for bugs.

Reads the PR diff, sends it to Claude with a structured review prompt,
formats the response as markdown, and saves it for the CI comment step.

Requires: ANTHROPIC_API_KEY set in environment (GitHub secret).
Optional: ANTHROPIC_MODEL (default: claude-sonnet-4-20250514).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent

REVIEW_PROMPT = """You are a senior Python code reviewer. Review the git diff below for
**correctness bugs only**. Do NOT report style, formatting, naming, or simplification
suggestions. Only report things that could cause wrong results, crashes, or silent
data corruption.

Focus on:
1. **Changed behavior**: removed guards, narrowed conditions, changed defaults that
   callers depended on.
2. **Cross-file breakage**: function signature changes, return type changes, new
   exceptions, changed preconditions that break callers.
3. **Python pitfalls**: mutable defaults, falsy-zero checks, bare except swallowing
   errors, late-binding closures, `is None` vs `or` fallback errors.
4. **Import / registration**: new files not registered in __init__.py, missing
   config entries, parser choices mismatched with method maps.
5. **Tensor / device bugs**: tensors on wrong device, missing .detach(), scalar
   tensors that would break torch.cat, shape mismatches in indexing.
6. **Training correctness**: loss computation changes, gradient flow issues,
   optimizer config changes that silently alter convergence.

For each finding, give: file path, line number (approximate from diff), one-line
summary, a concrete failure scenario (inputs/state → wrong output/crash), and
severity (critical/high/medium/low).

IMPORTANT: Only report bugs that are actually present in the diff, not
hypothetical issues. If you see no bugs, say so clearly.

Return your response as a JSON object with a single key "findings" containing
an array of objects with keys:
  - file (string)
  - line (integer, approximate)
  - summary (string, one line)
  - failure_scenario (string)
  - severity (string: critical/high/medium/low)

Return {"findings": []} if there are no bugs.

GIT DIFF:
"""

MAX_DIFF_CHARS = 80_000  # keep well under context limits


def get_pr_diff() -> Optional[str]:
    """Get the git diff for this PR."""
    base_ref = os.environ.get("GITHUB_BASE_REF", "origin/main")
    try:
        result = subprocess.run(
            ["git", "diff", f"{base_ref}...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass

    # Fallback: diff against HEAD~1
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1...HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout
    except Exception:
        pass

    return None


def call_claude(diff: str) -> Optional[str]:
    """Send the diff to Claude and get the review response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[review] ANTHROPIC_API_KEY not set — skipping LLM review")
        return None

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Truncate diff if needed
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated, too large)"
        print(f"[review] Diff truncated to {MAX_DIFF_CHARS} chars")

    prompt = REVIEW_PROMPT + diff

    print(f"[review] Sending {len(prompt)} chars to {model} ...")

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            system="You are a code reviewer. Return only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract text from the response
        if message.content and len(message.content) > 0:
            return message.content[0].text

    except ImportError:
        print("[review] anthropic package not installed — falling back to requests")
        return _call_claude_http(api_key, model, prompt)
    except Exception as e:
        print(f"[review] Claude API error: {e}")
        return None

    return None


def _call_claude_http(api_key: str, model: str, prompt: str) -> Optional[str]:
    """Fallback: call Claude API via raw HTTP (no anthropic SDK needed)."""
    import urllib.request
    import urllib.error

    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "system": "You are a code reviewer. Return only valid JSON.",
            "messages": [{"role": "user", "content": prompt}],
        }
    ).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            if data.get("content") and len(data["content"]) > 0:
                return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        print(f"[review] HTTP {e.code}: {e.read().decode()[:500]}")
    except Exception as e:
        print(f"[review] HTTP request error: {e}")

    return None


def parse_findings(text: str) -> list[dict]:
    """Extract findings JSON from Claude's response text."""
    # Try to find JSON block in the response
    import re

    # Look for ```json ... ``` code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("findings", [])
        except json.JSONDecodeError:
            pass

    # Try raw JSON in the text
    try:
        data = json.loads(text)
        return data.get("findings", [])
    except json.JSONDecodeError:
        pass

    # Try to find any {...} containing "findings"
    match = re.search(r'\{[^}]*"findings"\s*:\s*\[', text)
    if match:
        # Find the matching closing brace
        start = match.start()
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        return data.get("findings", [])
                    except json.JSONDecodeError:
                        break

    return []


def format_markdown(findings: list[dict], diff_stats: str) -> str:
    """Format findings as a nice markdown PR comment."""
    if not findings:
        return (
            "## 🤖 Claude Code Review\n\n"
            "**No bugs found** in the changed code. :white_check_mark:\n\n"
            f"<sub>Reviewed {diff_stats}. "
            "Powered by [Claude Code](https://claude.com/claude-code).</sub>"
        )

    sev_emoji = {
        "critical": ":red_circle:",
        "high": ":orange_circle:",
        "medium": ":yellow_circle:",
        "low": ":white_circle:",
    }

    lines = [
        "## :robot: Claude Code Review",
        "",
        f"Found **{len(findings)}** potential bug(s) in the changed code.",
        "",
        "| | Severity | File | Line | Summary |",
        "|---|---|---|---|---|",
    ]

    for f in findings:
        emoji = sev_emoji.get(f.get("severity", "low"), ":white_circle:")
        file = f.get("file", "?")
        line = f.get("line", "?")
        summary = f.get("summary", "")[:100]
        lines.append(
            f"| {emoji} | **{f.get('severity', 'low')}** | `{file}` | {line} | {summary} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    for i, f in enumerate(findings, 1):
        lines.append(f"### {i}. {f.get('summary', 'Untitled')}")
        lines.append("")
        lines.append(
            f"**File:** `{f.get('file', '?')}` — **Line:** {f.get('line', '?')} — **Severity:** `{f.get('severity', 'low')}`"
        )
        lines.append("")
        if f.get("failure_scenario"):
            lines.append(f"**Failure scenario:** {f['failure_scenario']}")
        lines.append("")

    lines.append("---")
    lines.append(
        f"<sub>:robot: Automated review of {diff_stats}. "
        "Powered by [Claude Code](https://claude.com/claude-code).</sub>"
    )

    return "\n".join(lines)


def main() -> int:
    print("=" * 60)
    print("LLM Code Review (Claude)")
    print("=" * 60)

    # 1. Get diff
    diff = get_pr_diff()
    if not diff:
        print("[review] No diff available — skipping")
        return 0

    file_count = diff.count("diff --git")
    diff_lines = diff.count("\n")
    diff_size = len(diff)
    diff_stats = f"{file_count} files, {diff_lines} lines, {diff_size // 1024}KiB"
    print(f"[review] Diff: {diff_stats}")

    # 2. Call Claude
    response = call_claude(diff)
    if response is None:
        # Check if this is a local run (no API key) — don't fail
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("[review] No ANTHROPIC_API_KEY — skipping (not an error)")
            return 0
        print("[review] Claude API call failed — exiting with error")
        return 1

    # 3. Parse findings
    findings = parse_findings(response)
    print(f"[review] Claude found {len(findings)} finding(s)")

    # 4. Format and save
    markdown = format_markdown(findings, diff_stats)

    output_path = Path("/tmp/review_findings.md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[review] Saved review to {output_path} ({len(markdown)} chars)")

    # Also print to stdout for CI logs
    print("\n" + markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
