# ADR 0013: Port the Apache-2.0 release preceding the license-only change

## Status

Accepted source boundary; strict checkpoint loading and public inference parity
have been verified. Full-split evaluation compatibility remains unverified.

## Context

The issue fixes reference commit `06766fdf56752ca535fc9b971fca99ce563676d0`,
which changed the upstream license to AGPL-3.0-only. Its parent
`4a07de9d06f2e3f14309753b7907cf1d3a263b08` distributes code under Apache-2.0.
An external checkout was inspected on 2026-10-09. The command below produces
no differences in the model package, evaluator, benchmark, tests or dependency
configuration:

```bash
git diff --name-only 4a07de9d06f2e3f14309753b7907cf1d3a263b08 \
  06766fdf56752ca535fc9b971fca99ce563676d0 -- relsgg tests benchmark pyproject.toml
```

The later commit changes LICENSE, adds NOTICE, updates attribution documents,
and changes the Gradio demo. We do not import any of that later licensed material.

## Decision

Port the reusable `relsgg` package from the explicitly Apache-2.0 parent revision
under `src/modules/relateanything/`. Keep the upstream license, copyright and
third-party notices beside it. Record source identity and the narrow namespace
change required by the official evaluator's absolute imports. Do not copy
research/benchmark/deploy trees into main. The complete network, text student,
losses, pack loader and evaluator are supported reusable components of this port.

The original parent license applies to this component; OpenSGG's root MIT file
does not replace it. The pinned issue revision remains the numerical reference
because its relevant files are identical. Future upstream changes must be audited
for both code parity and license provenance; no automatic AGPL-to-MIT conversion
or whole-repository relicensing is authorized by this decision.

Keep the source code in this repository but verify it in an isolated Python 3.13
environment with the official dependency series. The existing OpenSGG environment
must not be silently upgraded or have its pins loosened. Weight, tokenizer and
dataset assets stay external; DINOv3 and dataset terms remain applicable.

## Evidence

- [Apache parent LICENSE](https://github.com/Maelic/RelateAnything/blob/4a07de9d06f2e3f14309753b7907cf1d3a263b08/LICENSE)
- [Apache parent notices](https://github.com/Maelic/RelateAnything/blob/4a07de9d06f2e3f14309753b7907cf1d3a263b08/THIRD_PARTY_NOTICES.md)
- [License-change commit](https://github.com/Maelic/RelateAnything/commit/06766fdf56752ca535fc9b971fca99ce563676d0)
- External inspected checkout: `<external-official-checkout>`.

## Consequences

This permits a provenance-preserving source port without using the later AGPL
code revision as its license source. It does not establish a successful strict
weight load or reproduction. Checkpoint-backed full-split metrics remain gated
by issue #124 and the baseline manifests. Namespace/adapter parity must be tested
against the official implementation, and every behavioral change recorded.

Pinned source files are checked against the Apache Git revision by an optional
external-reference integration test. Scoped Ruff exceptions retain inherited
import layout, unused imports, non-strict zips and postponed type annotations;
CI auto-fixes must not silently invalidate the source manifest. OpenSGG facade
and integration code retain the normal lint checks. Method classes are resolved
on selection so the optional runtime does not import unrelated Lightning models;
the all-method CI check still resolves every class when iterating mapping items.
