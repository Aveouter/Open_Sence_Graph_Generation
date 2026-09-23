# Phase I: Predictive Relational Emergence — result reports

**Status: research instrument, not a reproduction.** Nothing in this directory
loads an SGG checkpoint or reports a benchmark number. The Phase IA artifacts it
cites are generated into `outputs/analysis/relational_emergence/` (gitignored)
and are reproducible from the frozen config recorded inside them.

The decisions fixing the data-generating process, the development/audit
separation, the evidence criteria and the Phase IB endpoints live in
[`reproduction/adr/0005`](../../../reproduction/adr/0005-phase1-data-generating-process.md),
[`0006`](../../../reproduction/adr/0006-development-vs-audit-world-protocol.md),
[`0007`](../../../reproduction/adr/0007-emergence-evidence-criteria.md),
[`0008`](../../../reproduction/adr/0008-phase1b-endpoint-specification.md) and
[`0009`](../../../reproduction/adr/0009-phase1b-protocol-freeze.md).

| Report | Covers |
|---|---|
| [01_phase1a_gate.md](01_phase1a_gate.md) | Ontology audit, M0 leakage probes, the Level-1 `J_ab` matrix, the two floors, horizon selection, the dataset splits, and the Phase IA gate verdict |
| [02_phase1a_level2.md](02_phase1a_level2.md) | The Level-2 oracle utility check (`delta_M`), in both oracle parameterisations, measured by the torch job on the frozen dataset |
| [03_phase1b_factorial.md](03_phase1b_factorial.md) | The seven arms, CCGP on unseen appearance families, role equivariance, label efficiency, the intervention-richness comparison, and the transplant endpoint with its must-fire control |
| [04_phase1b_audit_world.md](04_phase1b_audit_world.md) | The same factorial run once under the ADR 0009 freeze on `audit_world`, and what the seal changes about the claim |

Reports state measurement, protocol and limitation. Interpretation is confined
to its own section, and each report ends with what is explicitly **not** claimed.
