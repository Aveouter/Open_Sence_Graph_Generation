# Relational-emergence research extraction

[Issue #113](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/113)
separates two responsibilities that had grown together: this repository holds the
stable SGG system, and the relational-emergence research belongs in a repository
of its own.

## Current state

The extraction is **local only**. It has not been published, and no remote
research repository has been created.

| | |
|---|---|
| Preservation tag | `research-preservation/phase1-pre-extraction` |
| Preservation commit | `e2613e24466833570bc162afe738ea80414d1b2d` |
| Extracted copy | `.research-extraction/Relational_Representation_Research` (gitignored) |
| Extracted tip | `c312c54cb036a9a5639735f983e5ac1f09c0f8df` |
| Planned remote | `Aveouter/Relational_Representation_Research` — not created |

Machine-readable records live beside this file:

- `relational_emergence_manifest.json` — the migration index: preserved
  tag and commit, the five selected path groups, renames, both integrity
  records, and what was deliberately left behind.
- `relational_emergence_paths.txt` — the directive list handed to
  `git filter-repo`.
- `relational_emergence_raw_verification.json` — the extracted tree as
  filtered, hashed against the source blobs.
- `relational_emergence_extracted.json` — the extracted tree after link
  rewriting, with post-rewrite hashes.

## Why there are two integrity records

The extraction copies history and then deliberately rewrites part of it: 17
relative links were repointed at the new layout, the `CONTEXT.md` pointer to
`reproduction/glossary.md` became a citation because that document stays here,
and every `tools.relational_emergence` reference became `src.relational_research`
— including the documented `python -m` run commands. Comparing rewritten blobs
against source hashes would compare two different things, so the two stages are
recorded separately and never cross-checked.

`tests/_optional.py` was copied in and a `tests/__init__.py` added, so the
extracted suite runs without anything from this repository on the path. It is
checked with `unittest discover -s tests -t .` from the extracted root: 196
tests, no dependency on a file that exists only here.

`--stage raw` must run against the tree *as filtered*, before any content
changes. Running it afterwards fails, by design, and it will not overwrite a
passing record with a failing one.

## What has not happened

The research surface is still in this repository. Nothing was deleted:
`tools/relational_emergence/` (53 files), the eight research test modules,
`outputs/reports/relational_emergence/` (7 files), ADRs 0005–0010, and
`CONTEXT.md` are all present and unchanged.

They are frozen by `tools/reproduction/check_repository_boundary.py`, which
records each path's blob hash in
`reproduction/evidence/repository_boundary_baseline.json`. The check fails when
that surface changes — when a path is added, and equally when a frozen file is
edited, staged or not. Removing the surface later means deleting its lines from
that baseline; that is the one edit the ratchet is built to require.

Removal is deferred. It is gated on the extraction verifying against both
records above and on explicit approval — not on publishing a remote repository.

## Related work

- [PR #106](https://github.com/Aveouter/Open_Sence_Graph_Generation/pull/106)
  at `148b25663a40f8f4b2385956166859efebec8a81` — the immutable Phase I v1
  diagnostic record, preserved by ADR 0010.
- [#107](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/107) —
  Phase I v2 development boundary.
- [#108](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/108) —
  related Phase I work.
- [#110](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/110),
  [#111](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/111),
  [#112](https://github.com/Aveouter/Open_Sence_Graph_Generation/issues/112) —
  the research roadmap that follows the extraction.

## Deliberately left behind

- `tools/ontology_probe/` and its tests — a separate research line, tracked as
  its own issue rather than folded into this migration.
- `reproduction/adr/0001`–`0004` and `reproduction/glossary.md` — the
  reproduction standard and its vocabulary, which the stable system relies on.
- `outputs/analysis/relational_emergence/` — generated artifacts that were never
  tracked, so there was nothing in git to extract.
