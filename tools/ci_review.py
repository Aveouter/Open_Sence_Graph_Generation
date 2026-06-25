#!/usr/bin/env python3
"""
CI Code Review Script — uses LLM to review PR diffs for bugs.

Supports multiple backends (auto-detected from available API keys):
- DeepSeek V4-Pro (DEEPSEEK_API_KEY)  — recommended, OpenAI-compatible
- Anthropic Claude (ANTHROPIC_API_KEY)

Reads the PR diff, sends it to the LLM with a structured review prompt,
formats the response as markdown, and saves it for the CI comment step.

GitHub repo secrets needed (set at least one):
  DEEPSEEK_API_KEY  (preferred)
  ANTHROPIC_API_KEY (fallback)
"""

from __future__ import annotations

import json
import os
import re
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


# ================================================================
# Backend: DeepSeek (OpenAI-compatible)
# ================================================================


def call_deepseek(diff: str) -> Optional[str]:
    """Call DeepSeek API (OpenAI-compatible)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
    return _call_openai_compatible(
        api_key=api_key,
        model=model,
        endpoint="https://api.deepseek.com/chat/completions",
        diff=diff,
    )


# ================================================================
# Backend: Anthropic Claude
# ================================================================


def call_anthropic(diff: str) -> Optional[str]:
    """Call Anthropic Claude API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return _call_anthropic_http(api_key, model, diff)


# ================================================================
# Generic: OpenAI-compatible (used by DeepSeek, can extend to others)
# ================================================================


def _call_openai_compatible(
    api_key: str, model: str, endpoint: str, diff: str
) -> Optional[str]:
    """Call any OpenAI-compatible chat completions API."""
    import urllib.request
    import urllib.error

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
        print(f"[review] Diff truncated to {MAX_DIFF_CHARS} chars")

    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a code reviewer. Return only valid JSON with a 'findings' key.",
                },
                {"role": "user", "content": REVIEW_PROMPT + diff},
            ],
        }
    ).encode()

    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            # OpenAI format: choices[0].message.content
            choices = data.get("choices", [])
            if choices and choices[0].get("message"):
                return choices[0]["message"]["content"]
    except urllib.error.HTTPError as e:
        print(f"[review] HTTP {e.code}: {e.read().decode()[:500]}")
    except Exception as e:
        print(f"[review] HTTP request error: {e}")

    return None


def _call_anthropic_http(api_key: str, model: str, diff: str) -> Optional[str]:
    """Call Anthropic API via raw HTTP."""
    import urllib.request
    import urllib.error

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + "\n... (diff truncated)"
        print(f"[review] Diff truncated to {MAX_DIFF_CHARS} chars")

    body = json.dumps(
        {
            "model": model,
            "max_tokens": 4096,
            "system": "You are a code reviewer. Return only valid JSON.",
            "messages": [{"role": "user", "content": REVIEW_PROMPT + diff}],
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
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            if data.get("content") and len(data["content"]) > 0:
                return data["content"][0]["text"]
    except urllib.error.HTTPError as e:
        print(f"[review] HTTP {e.code}: {e.read().decode()[:500]}")
    except Exception as e:
        print(f"[review] HTTP request error: {e}")

    return None


# ================================================================
# Response parsing
# ================================================================


def parse_findings(text: str) -> Optional[list[dict]]:
    """Extract findings JSON from the LLM response text.

    Returns:
        list[dict] — findings parsed successfully (may be empty).
        None       — JSON parsing failed; the LLM response was malformed.
    """
    # Try ```json ... ``` code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("findings", [])
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    try:
        data = json.loads(text)
        return data.get("findings", [])
    except json.JSONDecodeError:
        pass

    # Try to find any {...} containing "findings"
    match = re.search(r'\{[^}]*"findings"\s*:\s*\[', text)
    if match:
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

    # If we reach here, JSON parsing failed
    return None


# ================================================================
# Markdown formatting
# ================================================================


def format_markdown(
    findings: Optional[list[dict]], diff_stats: str, backend: str
) -> str:
    """Format findings as a nice markdown PR comment.

    Args:
        findings: list of finding dicts, empty list (no bugs), or
                  None (JSON parsing failed).
    """
    if findings is None:
        return (
            f"## :warning: LLM Code Review ({backend})\n\n"
            "**Unable to parse the review response.** The LLM returned "
            "malformed output. Check the CI logs for the raw response.\n\n"
            f"<sub>Reviewed {diff_stats}.</sub>"
        )

    if not findings:
        return (
            f"## :robot: LLM Code Review ({backend})\n\n"
            "**No bugs found** in the changed code. :white_check_mark:\n\n"
            f"<sub>Reviewed {diff_stats}.</sub>"
        )

    sev_emoji = {
        "critical": ":red_circle:",
        "high": ":orange_circle:",
        "medium": ":yellow_circle:",
        "low": ":white_circle:",
    }

    lines = [
        f"## :robot: LLM Code Review ({backend})",
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
            f"**File:** `{f.get('file', '?')}` — "
            f"**Line:** {f.get('line', '?')} — "
            f"**Severity:** `{f.get('severity', 'low')}`"
        )
        lines.append("")
        if f.get("failure_scenario"):
            lines.append(f"**Failure scenario:** {f['failure_scenario']}")
        lines.append("")

    lines.append("---")
    lines.append(
        f"<sub>:robot: Automated review of {diff_stats}. Powered by {backend}.</sub>"
    )

    return "\n".join(lines)


# ================================================================
# Main
# ================================================================


def main() -> int:
    print("=" * 60)
    print("LLM Code Review")
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

    # 2. Pick backend and call LLM
    # Priority: DeepSeek → Anthropic
    response = None
    backend = "unknown"

    if os.environ.get("DEEPSEEK_API_KEY"):
        backend = "DeepSeek"
        print(
            f"[review] Using DeepSeek ({os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-pro')})"
        )
        response = call_deepseek(diff)
    elif os.environ.get("ANTHROPIC_API_KEY"):
        backend = "Anthropic Claude"
        print(
            f"[review] Using Anthropic ({os.environ.get('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')})"
        )
        response = call_anthropic(diff)
    else:
        print(
            "[review] No API key set. "
            "Set DEEPSEEK_API_KEY or ANTHROPIC_API_KEY in repo secrets."
        )
        return 0

    if response is None:
        print(f"[review] {backend} API call failed")
        return 1

    # 3. Parse findings
    findings = parse_findings(response)
    if findings is None:
        print(f"[review] {backend} response could not be parsed as JSON")
        print("[review] Raw response (first 500 chars):")
        print(response[:500])
        # Write a failure comment so the PR knows the review broke
        markdown = format_markdown(None, diff_stats, backend)
    else:
        print(f"[review] {backend} found {len(findings)} finding(s)")
        markdown = format_markdown(findings, diff_stats, backend)

    output_path = Path("/tmp/review_findings.md")
    output_path.write_text(markdown, encoding="utf-8")
    print(f"[review] Saved review to {output_path} ({len(markdown)} chars)")

    print("\n" + markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
