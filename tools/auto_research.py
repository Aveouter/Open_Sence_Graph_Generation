#!/usr/bin/env python3
"""File-based coordination helper for the automatic research workspace.

The tool is deliberately small and dependency-free. It manages the fixed files
under research_workspace/control so Codex and Claude Code can hand off work,
recover after interruptions, and keep an auditable trail.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT / "research_workspace"
CONTROL = WORKSPACE / "control"
REPORTS = WORKSPACE / "reports"
STATE_PATH = CONTROL / "state.json"
LOCK_PATH = CONTROL / "lock.json"
EVENT_LOG = CONTROL / "event_log.md"
DECISION_LOG = CONTROL / "decision_log.md"
CODEX_TO_CLAUDE = CONTROL / "codex_to_claude.md"
CLAUDE_TO_CODEX = CONTROL / "claude_to_codex.md"
TASK_QUEUE = CONTROL / "task_queue.md"


REQUIRED_FILES = [
    STATE_PATH,
    LOCK_PATH,
    EVENT_LOG,
    DECISION_LOG,
    CODEX_TO_CLAUDE,
    CLAUDE_TO_CODEX,
    TASK_QUEUE,
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utc_now()).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    keep = []
    for ch in value.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {"-", "_", ".", "/"}:
            keep.append("-" if ch == "/" else ch)
        elif ch.isspace():
            keep.append("-")
    out = "".join(keep).strip("-._")
    return out or "task"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    tmp.replace(path)


def append_event(actor: str, task_id: str, event: str, message: str) -> None:
    EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG.exists():
        EVENT_LOG.write_text(
            "# Event Log\n\n| UTC Time | Actor | Task | Event | Message |\n"
            "| --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    safe_message = message.replace("\n", " ").replace("|", "\\|")
    line = f"| {iso()} | {actor} | `{task_id}` | {event} | {safe_message} |\n"
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(line)


def run_git(args: list[str], check: bool = False) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if check and proc.returncode != 0:
        raise SystemExit(proc.stdout)
    return proc.stdout.rstrip()


def load_state() -> dict[str, Any]:
    return read_json(STATE_PATH)


def save_state(state: dict[str, Any]) -> None:
    state["last_update_utc"] = iso()
    git_info = state.setdefault("git", {})
    git_info["branch"] = run_git(["branch", "--show-current"]) or None
    git_info["last_known_commit"] = run_git(["rev-parse", "--short", "HEAD"]) or None
    write_json(STATE_PATH, state)


def lock_is_active(lock: dict[str, Any]) -> bool:
    if not lock.get("locked"):
        return False
    expires = lock.get("expires_at_utc")
    if not expires:
        return True
    try:
        expires_dt = datetime.fromisoformat(expires.replace("Z", "+00:00"))
    except ValueError:
        return True
    return expires_dt > utc_now()


def cmd_status(_: argparse.Namespace) -> int:
    state = load_state()
    lock = read_json(LOCK_PATH)
    print("Automatic Research Workspace")
    print(f"  task_id: {state.get('task_id')}")
    print(f"  phase:   {state.get('phase')}")
    print(f"  status:  {state.get('status')}")
    print(f"  owner:   {state.get('owner')}")
    print(f"  updated: {state.get('last_update_utc')}")
    print(f"  next:    {state.get('next_action')}")
    print("")
    print("Lock")
    if lock_is_active(lock):
        print(
            f"  active:  {lock.get('actor')} until {lock.get('expires_at_utc')}"
        )
    else:
        print("  active:  no")
    print("")
    print("Git")
    print(run_git(["status", "--short", "--branch"]))
    return 0


def cmd_validate(_: argparse.Namespace) -> int:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_FILES if not p.exists()]
    if missing:
        print("Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    for path in [STATE_PATH, LOCK_PATH]:
        try:
            read_json(path)
        except json.JSONDecodeError as exc:
            print(f"Invalid JSON in {path}: {exc}")
            return 1

    lock = read_json(LOCK_PATH)
    if lock.get("locked") and not lock_is_active(lock):
        print("Lock exists but is expired; it may be reclaimed.")
    print("Workspace validation passed.")
    return 0


def cmd_lock(args: argparse.Namespace) -> int:
    lock = read_json(LOCK_PATH)
    if lock_is_active(lock) and not args.force:
        print(
            "Active lock held by "
            f"{lock.get('actor')} until {lock.get('expires_at_utc')}. "
            "Use --force only if that agent is known to be stopped."
        )
        return 1

    state = load_state()
    task_id = args.task_id or state.get("task_id") or "unknown"
    now = utc_now()
    new_lock = {
        "locked": True,
        "actor": args.actor,
        "task_id": task_id,
        "acquired_at_utc": iso(now),
        "expires_at_utc": iso(now + timedelta(minutes=args.ttl_minutes)),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "note": args.note or "",
    }
    write_json(LOCK_PATH, new_lock)
    state["owner"] = args.actor
    state["status"] = "locked"
    save_state(state)
    append_event(args.actor, task_id, "lock", f"claimed for {args.ttl_minutes} minutes")
    print(f"Lock claimed by {args.actor} for task {task_id}.")
    return 0


def cmd_unlock(args: argparse.Namespace) -> int:
    lock = read_json(LOCK_PATH)
    if lock.get("locked") and lock.get("actor") != args.actor and not args.force:
        print(
            f"Lock is held by {lock.get('actor')}; use --force if reclaiming intentionally."
        )
        return 1

    task_id = lock.get("task_id") or args.task_id or "unknown"
    write_json(
        LOCK_PATH,
        {
            "locked": False,
            "actor": None,
            "task_id": None,
            "acquired_at_utc": None,
            "expires_at_utc": None,
            "host": None,
            "pid": None,
            "note": "Use tools/auto_research.py lock/unlock. Expired locks may be reclaimed.",
        },
    )
    state = load_state()
    state["status"] = "idle"
    save_state(state)
    append_event(args.actor, task_id, "unlock", "released lock")
    print("Lock released.")
    return 0


def cmd_set_state(args: argparse.Namespace) -> int:
    state = load_state()
    for key in ["run_id", "task_id", "phase", "status", "owner", "summary", "next_action"]:
        value = getattr(args, key)
        if value is not None:
            state[key] = value
    if args.success_criteria:
        state["success_criteria"] = args.success_criteria
    if args.open_question:
        state["open_questions"] = args.open_question
    if args.artifact_root:
        state["artifact_root"] = args.artifact_root
    save_state(state)
    append_event(args.actor, state.get("task_id", "unknown"), "state", "updated state")
    print(f"Updated state for task {state.get('task_id')}.")
    return 0


def handoff_path(sender: str, receiver: str) -> Path:
    pair = (sender.lower(), receiver.lower())
    if pair == ("codex", "claude"):
        return CODEX_TO_CLAUDE
    if pair == ("claude", "codex"):
        return CLAUDE_TO_CODEX
    raise SystemExit("--from/--to must be codex->claude or claude->codex")


def cmd_handoff(args: argparse.Namespace) -> int:
    path = handoff_path(args.sender, args.receiver)
    task_id = args.task_id
    criteria = "\n".join(f"- {item}" for item in args.criteria) or "- Not specified."
    notes = args.notes or "None."
    artifacts = "\n".join(f"- {item}" for item in args.artifact) or "- None."

    body = f"""# {args.sender.title()} to {args.receiver.title()} Handoff

Status: {args.status}

## Task

- Task ID: `{task_id}`
- Updated UTC: {iso()}
- Summary: {args.summary}

## Next Action

{args.next}

## Success Criteria

{criteria}

## Artifacts

{artifacts}

## Notes

{notes}
"""
    path.write_text(body, encoding="utf-8")

    state = load_state()
    state["task_id"] = task_id
    state["owner"] = args.receiver
    state["status"] = args.status
    state["phase"] = args.phase
    state["summary"] = args.summary
    state["next_action"] = args.next
    if args.criteria:
        state["success_criteria"] = args.criteria
    save_state(state)
    append_event(args.sender, task_id, "handoff", f"sent to {args.receiver}: {args.summary}")
    print(f"Wrote {path.relative_to(ROOT)}")
    return 0


def cmd_log(args: argparse.Namespace) -> int:
    append_event(args.actor, args.task_id, args.event, args.message)
    print("Logged event.")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    if not DECISION_LOG.exists():
        DECISION_LOG.write_text(
            "# Decision Log\n\n| UTC Time | Actor | Task | Decision | Rationale |\n"
            "| --- | --- | --- | --- | --- |\n",
            encoding="utf-8",
        )
    safe_decision = args.decision.replace("\n", " ").replace("|", "\\|")
    safe_rationale = args.rationale.replace("\n", " ").replace("|", "\\|")
    with DECISION_LOG.open("a", encoding="utf-8") as f:
        f.write(
            f"| {iso()} | {args.actor} | `{args.task_id}` | "
            f"{safe_decision} | {safe_rationale} |\n"
        )
    append_event(args.actor, args.task_id, "decision", args.decision)
    print("Logged decision.")
    return 0


def snapshot_text(task_id: str, summary: str) -> str:
    state = STATE_PATH.read_text(encoding="utf-8")
    codex = CODEX_TO_CLAUDE.read_text(encoding="utf-8")
    claude = CLAUDE_TO_CODEX.read_text(encoding="utf-8")
    return f"""# Research Workspace Snapshot

- Task ID: `{task_id}`
- UTC: {iso()}
- Summary: {summary}
- Branch: `{run_git(["branch", "--show-current"]) or "detached"}`
- Commit: `{run_git(["rev-parse", "--short", "HEAD"])}`

## Git Status

```text
{run_git(["status", "--short", "--branch"])}
```

## Git Diff Stat

```text
{run_git(["diff", "--stat"])}
```

## Recent Commits

```text
{run_git(["log", "--oneline", "--decorate", "-8"])}
```

## State

```json
{state}
```

## Codex to Claude

```markdown
{codex}
```

## Claude to Codex

```markdown
{claude}
```
"""


def cmd_snapshot(args: argparse.Namespace) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    name = f"{iso().replace(':', '').replace('-', '')}_{slug(args.task_id)}_snapshot.md"
    path = REPORTS / name
    path.write_text(snapshot_text(args.task_id, args.summary), encoding="utf-8")
    append_event(args.actor, args.task_id, "snapshot", str(path.relative_to(ROOT)))
    print(path.relative_to(ROOT))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    REPORTS.mkdir(parents=True, exist_ok=True)
    status = args.status
    path = REPORTS / f"{slug(args.task_id)}_{slug(args.title)}.md"
    commands = "\n".join(args.command) if args.command else "# no commands recorded"
    artifacts = "\n".join(f"- {item}" for item in args.artifact) or "- None."
    body = f"""# {args.title}

## Metadata

- Task ID: `{args.task_id}`
- Actor: {args.actor}
- UTC: {iso()}
- Status: {status}
- Branch: `{run_git(["branch", "--show-current"]) or "detached"}`
- Commit: `{run_git(["rev-parse", "--short", "HEAD"])}`

## Summary

{args.summary}

## Commands

```bash
{commands}
```

## Artifacts

{artifacts}

## Result

{args.result}

## Next Action

{args.next}

## Git Status

```text
{run_git(["status", "--short", "--branch"])}
```
"""
    path.write_text(body, encoding="utf-8")
    append_event(args.actor, args.task_id, "report", str(path.relative_to(ROOT)))
    print(path.relative_to(ROOT))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("status", help="Show state, lock, and git status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("validate", help="Validate required workspace files")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("lock", help="Claim the workspace lock")
    p.add_argument("--actor", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id")
    p.add_argument("--ttl-minutes", type=int, default=120)
    p.add_argument("--note")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_lock)

    p = sub.add_parser("unlock", help="Release the workspace lock")
    p.add_argument("--actor", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_unlock)

    p = sub.add_parser("set-state", help="Update state.json")
    p.add_argument("--actor", default="codex", choices=["codex", "claude"])
    p.add_argument("--run-id")
    p.add_argument("--task-id")
    p.add_argument("--phase")
    p.add_argument("--status")
    p.add_argument("--owner", choices=["codex", "claude"])
    p.add_argument("--summary")
    p.add_argument("--next-action")
    p.add_argument("--success-criteria", action="append")
    p.add_argument("--open-question", action="append")
    p.add_argument("--artifact-root")
    p.set_defaults(func=cmd_set_state)

    p = sub.add_parser("handoff", help="Write an agent handoff")
    p.add_argument("--from", dest="sender", required=True, choices=["codex", "claude"])
    p.add_argument("--to", dest="receiver", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--next", required=True)
    p.add_argument("--criteria", action="append", default=[])
    p.add_argument("--artifact", action="append", default=[])
    p.add_argument("--notes")
    p.add_argument("--status", default="handoff")
    p.add_argument("--phase", default="execution")
    p.set_defaults(func=cmd_handoff)

    p = sub.add_parser("log", help="Append to event_log.md")
    p.add_argument("--actor", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--event", required=True)
    p.add_argument("--message", required=True)
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("decision", help="Append to decision_log.md")
    p.add_argument("--actor", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--decision", required=True)
    p.add_argument("--rationale", required=True)
    p.set_defaults(func=cmd_decision)

    p = sub.add_parser("snapshot", help="Write an audit snapshot report")
    p.add_argument("--actor", default="codex", choices=["codex", "claude"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--summary", default="Manual snapshot")
    p.set_defaults(func=cmd_snapshot)

    p = sub.add_parser("report", help="Write a compact task report")
    p.add_argument("--actor", required=True, choices=["codex", "claude"])
    p.add_argument("--task-id", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--status", default="complete")
    p.add_argument("--summary", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--next", required=True)
    p.add_argument("--command", action="append", default=[])
    p.add_argument("--artifact", action="append", default=[])
    p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
