# Reproduction PR Checklist

## Claim Type

Choose one:

- [ ] code alignment fix
- [ ] checkpoint-backed evaluation
- [ ] protocol alignment documentation
- [ ] implementation audit
- [ ] deferred reproduction report
- [ ] guardrail against misleading claims

## Evidence

- [ ] official paper/repo cited
- [ ] checkpoint provenance recorded or explicitly unavailable
- [ ] config alignment recorded
- [ ] inference flow alignment recorded
- [ ] evaluator semantics alignment recorded
- [ ] no random-init/tiny-slice result is presented as reproduction
- [ ] deviations and blockers are explicit

## Safety Confirmations

- [ ] Confirmed: no label edits
- [ ] Confirmed: no ground-truth edits
- [ ] Confirmed: no evaluator semantic changes to improve numbers
- [ ] Confirmed: no failed sample deletion
- [ ] Confirmed: no paper-alignment claim without evidence

## Verification

- Commands run:
- Outputs:
- Known limitations:
