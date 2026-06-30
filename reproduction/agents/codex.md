# Codex Workflow

Codex is the primary agent for this repository.

## Before Acting

1. Read `AGENTS.md`.
2. Read `reproduction/README.md` and `reproduction/glossary.md`.
3. If working on a baseline, create/update an issue from
   `reproduction/issues/baseline_reproduction_issue.md`.
4. Do not run or patch code until the required evidence and claim boundary are
   clear.

## During Work

- Use `conda run -n hsg` for Python/PyTorch commands.
- Prefer official repository and checkpoint evidence over local approximations.
- If the method cannot be aligned, write an audit/gap report and stop.
- Record decisions in `reproduction/adr/`.

## Before PR

- Fill `reproduction/templates/pr_checklist.md`.
- Run `python tools/reproduction/check_reproduction_claims.py --changed-from <base> AGENTS.md reproduction <pr-body-or-release-note-files>`.
- If the PR adds or changes reproduction tooling, read and update
  `reproduction/USAGE.md` so later agents can run the same path.
- Use PR titles that match the claim type, for example:
  - `fix(metric): align RelTR evaluator with official protocol`
  - `docs(tde): record checkpoint-unavailable deferred reproduction`
  - `audit(ra-sgg): compare OpenSGG adapter with official ReTAG`
