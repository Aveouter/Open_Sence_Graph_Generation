# Automatic Research Workspace

This directory defines a file-based operating protocol for long-running research work with two agents:

- Codex owns planning, result judgement, scientific direction changes, and PR hygiene.
- Claude Code owns implementation, experiment execution, bug fixing, and raw result collection.

The agents do not need to copy messages between chat sessions. They communicate through the fixed files under `research_workspace/control/`, and they use git status, diffs, reports, and event logs as the audit trail.

## Directory Layout

```text
research_workspace/
  agents/
    CODEX.md              # Codex operating contract
    CLAUDE_CODE.md        # Claude Code operating contract
  control/
    state.json            # Current task, owner, phase, status, and next action
    lock.json             # Best-effort agent lock with expiry
    task_queue.md         # Prioritized backlog and active task
    codex_to_claude.md    # Planner to implementer handoff
    claude_to_codex.md    # Implementer to planner handoff
    event_log.md          # Append-only operational log
    decision_log.md       # Scientific and engineering decisions
  reports/
    .gitkeep              # Commit milestone reports here
  templates/
    experiment_report.md  # Report template for completed attempts
```

Large runtime artifacts, raw command logs, checkpoints, and generated plots should go under `outputs/research_workspace/<task_id>/` or another ignored output directory. Commit only compact reports, code, configs, and small metadata needed for audit.

## Basic Loop

1. Codex reads `control/state.json`, `control/task_queue.md`, and `control/claude_to_codex.md`.
2. Codex writes a precise implementation or experiment request to `control/codex_to_claude.md`.
3. Claude Code claims the lock, executes code and experiments, then writes results to `control/claude_to_codex.md`.
4. Codex judges whether the result supports the hypothesis, updates direction, and either creates a new handoff or prepares a PR.
5. Both agents append short entries to `control/event_log.md`; scientific decisions go in `control/decision_log.md`.

## CLI

Use the stdlib helper:

```bash
python tools/auto_research.py status
python tools/auto_research.py lock --actor codex --ttl-minutes 120
python tools/auto_research.py handoff --from codex --to claude --task-id exp001 --summary "Run RelTR smoke" --next "Implement and run the command listed below"
python tools/auto_research.py log --actor claude --task-id exp001 --event experiment --message "Finished smoke run"
python tools/auto_research.py snapshot --task-id exp001
python tools/auto_research.py validate
python tools/auto_research.py unlock --actor codex
```

The helper intentionally does not launch background jobs. It writes state, lock, handoff, and report files so either agent can resume after interruption.

## Recovery Rules

- If interrupted, start with `python tools/auto_research.py status` and read both handoff files.
- If `lock.json` is expired, a new agent may claim it. If it is not expired, do not overwrite it unless the prior agent is known to be dead.
- If an experiment partially ran, Claude Code must write what command started, where outputs are, and whether artifacts are trustworthy.
- If state and git disagree, trust git status/diff first, then repair `state.json` and append an event.
- If a direction changes, Codex must record the reason in `decision_log.md`.

## PR Rules

- Keep research-control updates separate from large artifacts.
- Do not commit raw checkpoints, core dumps, or full logs.
- Before opening a PR, run `python tools/auto_research.py snapshot --task-id <id>` and include the report path in the PR body.
- Use one PR per coherent code/documentation change. If an experiment produces many local-only files, summarize them in `research_workspace/reports/`.
