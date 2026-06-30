# FREQ Smoke Report

Command:

```bash
python tools/ci_smoke_test.py --methods freq
```

Result:

- Method instantiation passed.
- Synthetic forward pass passed.
- Synthetic forward loss: `3.9318`.
- Minimal CPU train subprocess completed under the smoke harness after CPU/CUDA
  startup fixes.

Diagnose/fix/retry notes:

- Parser initially rejected lowercase `freq`; fixed by adding parser choice and
  config alias.
- CPU train smoke initially crashed in method-info display due to CUDA driver
  mismatch; fixed by using the existing `--no_display_method_info` flag.
- Environment collection then crashed when querying CUDA device names; fixed by
  recording the CUDA query error instead of raising.
- Lightning CUDA RNG isolation then crashed because CUDA was visible but unusable;
  fixed by hiding CUDA only for the CPU smoke subprocess.
