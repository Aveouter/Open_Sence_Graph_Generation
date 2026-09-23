"""Why does the transplant show no contrast? A diagnostic kept for the record.

The endpoint returned a null, and a null is only worth reporting if the device
that produced it can produce a positive. This runs the identical machinery with a
code that *is* the mechanism -- a one-hot label -- and beside it the learned code,
and reports the paired alignment as a function of how far the decoder is rolled.

It answers three questions in one run:

1. can the decoder turn a known-correct code into a prediction that moves the way
   the law moves the world? (oracle row: if not, the instrument is broken and no
   null about the representation is reportable)
2. at what rollout length does that stop being true? (the horizon at which the
   endpoint is meaningful)
3. how does the learned code compare at that horizon?

Not part of the pipeline. Kept because it is the evidence behind how the
transplant endpoint is specified.
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from ..common import DEFAULT_OUTPUT_ROOT
from ..eval import design
from ..simulator.splits import DEFAULT_SPLIT_SEED, build_split
from .run_phase1a import build_groups
from .run_phase1b import TRANSPLANT_HORIZON, _rows_for, oracle_codes, standardise_latents

MAX_STEPS = TRANSPLANT_HORIZON


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups-per-tuple", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--decoder-epochs", type=int, default=80)
    parser.add_argument("--arm", default="P-R")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_ROOT / "phase1b_diagnostic")
    args = parser.parse_args(argv)

    import torch

    from ..config import Phase1AConfig
    from ..eval.transplant import cosine, flatten
    from ..models.batch_stream import (
        StreamSpec,
        action_dim,
        action_vector,
        build_part,
        frame_series,
        struct_dim,
    )
    from ..models.representation import ARMS, ModelConfig, train_arm
    from ..models.transplant import DecoderConfig, freeze_latents, roll_from_state, train_decoder
    from ..simulator.dataset import build_rows
    from ..simulator.geometry import GravityFrame

    config = Phase1AConfig(groups_per_tuple=args.groups_per_tuple)
    groups = build_groups(config)
    rows = [row.as_dict() for row in build_rows(groups, config.horizon)]
    split = build_split(groups, "unseen_family", seed=DEFAULT_SPLIT_SEED)
    partition = design.partition_contexts(split.train, split.test, seed=DEFAULT_SPLIT_SEED + 1)
    fit_rows = _rows_for(rows, partition.fit)
    test_rows = _rows_for(rows, partition.test)
    spec = StreamSpec(history=4, horizon=MAX_STEPS)

    record = train_arm(
        args.arm,
        {
            "train": build_part(
                fit_rows,
                spec,
                with_relation=ARMS[args.arm].uses_relation_label,
                future_steps=ModelConfig().rollout_steps,
            ),
            "val": build_part(
                _rows_for(rows, partition.validation),
                spec,
                with_relation=ARMS[args.arm].uses_relation_label,
                future_steps=ModelConfig().rollout_steps,
            ),
        },
        state_dim=8,
        struct_dim=struct_dim(),
        action_dim=action_dim(),
        config=ModelConfig(epochs=args.epochs, seed=0),
    )
    model = record["model"]

    steps = DecoderConfig().rollout_steps
    fit_part = build_part(fit_rows, spec, with_relation=False, future_steps=steps)
    test_part = build_part(test_rows, spec, with_relation=False)
    raw_fit = freeze_latents(model, fit_part)

    row_index = {f"{row['group']}|{row['regime']}|{row['mechanism']}": row for row in test_rows}
    # Flat index of each episode's first step; see run_phase1b._transplant.
    part_index = {
        f"{row['group']}|{row['regime']}|{row['mechanism']}": position * spec.horizon
        for position, row in enumerate(test_rows)
    }
    keys = sorted(part_index)

    for code in ("oracle", "learned"):
        if code == "oracle":
            fit_codes = oracle_codes(fit_rows, spec)
            codes = oracle_codes(test_rows, spec)
        else:
            fit_codes = standardise_latents(raw_fit, raw_fit)
            codes = standardise_latents(raw_fit, freeze_latents(model, test_part))
        fitted = train_decoder(
            fit_part,
            fit_codes,
            state_dim=8,
            struct_dim=struct_dim(),
            action_dim=action_dim(),
            latent=int(fit_codes.shape[1]),
            config=DecoderConfig(epochs=args.decoder_epochs, seed=0),
        )
        decoder = fitted["model"]
        print(f"[diag] {code}: decoder final loss {fitted['final_loss']:.5f}")

        # Every (target, regime, factual, counterfactual) cell, with the source
        # chosen as a different-filler context of the same tuple.
        cells: dict[tuple, list[float]] = {}
        skipped = 0
        by_tuple: dict[str, list[str]] = {}
        filler = {row["group"]: row["filler_index"] for row in rows}
        for key in keys:
            by_tuple.setdefault(row_index[key]["tuple"], []).append(key)
        for target in sorted({row_index[key]["group"] for key in keys}):
            sibling = next(key for key in keys if row_index[key]["group"] == target)
            eligible = [
                key
                for key in by_tuple[row_index[sibling]["tuple"]]
                if row_index[key]["group"] != target and filler[row_index[key]["group"]] != filler[target]
            ]
            if not eligible:
                print(f"[diag]   no eligible source for {target}")
                continue
            # `by_tuple` holds full (group|regime|mechanism) keys; the source is
            # named by its context, and the regime and mechanism are supplied by
            # the cell being evaluated.
            source = row_index[eligible[0]]["group"]
            for regime in sorted({row_index[key]["regime"] for key in keys if row_index[key]["group"] == target}):
                mechanisms = sorted(
                    {
                        row_index[key]["mechanism"]
                        for key in keys
                        if row_index[key]["group"] == target and row_index[key]["regime"] == regime
                    }
                )
                for counterfactual in mechanisms:
                    for factual in mechanisms:
                        if factual == counterfactual:
                            continue
                        target_cf = f"{target}|{regime}|{counterfactual}"
                        target_fa = f"{target}|{regime}|{factual}"
                        source_cf = f"{source}|{regime}|{counterfactual}"
                        source_fa = f"{source}|{regime}|{factual}"
                        if not all(k in part_index for k in (target_cf, target_fa, source_cf, source_fa)):
                            continue
                        row = row_index[target_cf]
                        factual_row = row_index[target_fa]
                        frame = GravityFrame(row["frame_angle_rad"])
                        series = frame_series(frame, [row["x0"], *row["trajectory"]])
                        factual_series = frame_series(
                            GravityFrame(factual_row["frame_angle_rad"]),
                            [factual_row["x0"], *factual_row["trajectory"]],
                        )
                        actions = [action_vector(row, t) for t in range(MAX_STEPS)]
                        structural = torch.tensor([row["filler_structural"]], dtype=torch.float32)

                        for steps in (1, 2, 3, 5, MAX_STEPS):
                            roll = partial(
                                roll_from_state,
                                decoder,
                                initial_state=list(series[0]),
                                structural=structural,
                                actions=actions,
                                steps=steps,
                            )
                            moved = [
                                a - b
                                for a, b in zip(
                                    flatten(
                                        roll(latent_vector=codes[part_index[source_cf]])
                                    ),
                                    flatten(
                                        roll(latent_vector=codes[part_index[source_fa]])
                                    ),
                                    strict=True,
                                )
                            ]
                            law_moved = [
                                a - b
                                for a, b in zip(
                                    flatten([list(e) for e in series[1 : steps + 1]]),
                                    flatten([list(e) for e in factual_series[1 : steps + 1]]),
                                    strict=True,
                                )
                            ]
                            nmoved = sum(v * v for v in moved) ** 0.5
                            nlaw = sum(v * v for v in law_moved) ** 0.5
                            if nmoved < 1e-9 or nlaw < 1e-9:
                                skipped += 1
                                continue
                            cells.setdefault((regime, steps), []).append(cosine(moved, law_moved))

        print(f"[diag] {code}: skipped {skipped} degenerate cells")
        print(f"[diag] {code}: mean alignment by regime and rollout length")
        for key in sorted(cells):
            values = cells[key]
            print(
                f"[diag]   {key[0]:8s} steps={key[1]:2d} "
                f"mean={sum(values) / len(values):+.4f} n={len(values)}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
