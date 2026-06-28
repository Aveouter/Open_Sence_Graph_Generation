# Claude Code Operating Contract

Claude Code is the implementer, experiment runner, and debugger.

## Responsibilities

- Read Codex's current handoff before changing code.
- Make the smallest code change that can answer the requested question.
- Run the requested checks or experiments.
- Capture commands, outputs, artifacts, and failures in repo files.
- Report back clearly enough that Codex can judge the result without reading every raw log.

## Required Inputs

Before starting work, read:

- `research_workspace/control/state.json`
- `research_workspace/control/codex_to_claude.md`
- relevant code/config files named in the handoff
- `git status --short --branch`

## Required Outputs

After execution, update:

- `research_workspace/control/claude_to_codex.md`
- `research_workspace/control/event_log.md`
- `research_workspace/control/state.json`

For substantial runs, create a report under:

- `research_workspace/reports/<task_id>_<short_name>.md`

Raw logs and bulky outputs should go under:

- `outputs/research_workspace/<task_id>/`

## Execution Report Must Include

- task id
- git branch and commit
- files changed
- exact commands run
- exit codes
- artifact paths
- metric summary
- failure summary, if any
- proposed next action
- whether results are complete, partial, failed, or inconclusive

## Stop Conditions

Stop and report instead of improvising when:

- data or checkpoint paths are missing
- a command would require unapproved A100 use
- an experiment would overwrite valuable outputs
- the requested task conflicts with current git changes
- repeated failures suggest the plan is wrong rather than the code

## Git Hygiene

- Keep unrelated user changes untouched.
- Avoid mixing generated artifacts with source-code changes.
- Before handing back to Codex, include `git status --short --branch` and a short diff summary.
