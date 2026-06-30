# Claude Code Compatibility Notes

Claude Code can follow this workflow by using the same tracked files as Codex.
Do not rely on local `.claude/` or `CLAUDE.md` files for scientific standards,
because this repository may ignore those paths.

## Required Inputs

- `AGENTS.md`
- `reproduction/README.md`
- `reproduction/glossary.md`
- the relevant baseline issue
- any ADRs cited by the issue

## Compatibility Rule

If Claude Code instructions conflict with `AGENTS.md` or `reproduction/`, prefer
the Codex/reproduction workflow. If compatibility becomes hard to maintain,
drop Claude-specific behavior and keep the tracked Codex-first standard.
