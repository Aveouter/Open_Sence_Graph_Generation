"""Does the decoder read a code it was trained on? A two-minute check.

The alignment diagnostic skipped far more cells under the one-hot code than under
the learned one, which can only happen if the two rolls differ by exactly zero --
i.e. if the decoder ignored an input it was trained on. That would break the
must-fire control, so it is checked directly rather than assumed: train the same
decoder twice, once per code, and report how far swapping the code moves the
prediction at the first step.
"""

from __future__ import annotations

import argparse

from ..eval import design
from ..simulator.splits import DEFAULT_SPLIT_SEED, build_split
from .run_phase1a import build_groups
from .run_phase1b import TRANSPLANT_HORIZON, _rows_for, oracle_codes, standardise_latents


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups-per-tuple", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--decoder-epochs", type=int, default=80)
    parser.add_argument("--arm", default="P-R")
    args = parser.parse_args(argv)

    import torch

    from ..config import Phase1AConfig
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
    spec = StreamSpec(history=4, horizon=TRANSPLANT_HORIZON)

    model = train_arm(
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
    )["model"]

    steps = DecoderConfig().rollout_steps
    fit_part = build_part(fit_rows, spec, with_relation=False, future_steps=steps)
    test_part = build_part(test_rows, spec, with_relation=False)
    raw_fit = freeze_latents(model, fit_part)

    for code in ("oracle", "learned"):
        if code == "oracle":
            fit_codes, codes = oracle_codes(fit_rows, spec), oracle_codes(test_rows, spec)
        else:
            fit_codes = standardise_latents(raw_fit, raw_fit)
            codes = standardise_latents(raw_fit, freeze_latents(model, test_part))
        decoder = train_decoder(
            fit_part,
            fit_codes,
            state_dim=8,
            struct_dim=struct_dim(),
            action_dim=action_dim(),
            latent=int(fit_codes.shape[1]),
            config=DecoderConfig(epochs=args.decoder_epochs, seed=0),
        )["model"]

        row_index = {f"{r['group']}|{r['regime']}|{r['mechanism']}": r for r in test_rows}
        # Flat index of each episode's first step; see run_phase1b._transplant.
        part_index = {
            f"{r['group']}|{r['regime']}|{r['mechanism']}": position * spec.horizon
            for position, r in enumerate(test_rows)
        }
        moves: list[float] = []
        zero = 0
        for key, position in list(part_index.items())[:200]:
            row = row_index[key]
            siblings = [
                k
                for k in part_index
                if k.startswith(key.rsplit("|", 1)[0] + "|") and k != key
            ]
            if not siblings:
                continue
            other = part_index[siblings[0]]
            frame = GravityFrame(row["frame_angle_rad"])
            series = frame_series(frame, [row["x0"], *row["trajectory"]])
            structural = torch.tensor([row["filler_structural"]], dtype=torch.float32)
            actions = [action_vector(row, t) for t in range(TRANSPLANT_HORIZON)]

            def roll(index: int) -> list[float]:
                return [
                    value
                    for state in roll_from_state(
                        decoder,
                        initial_state=list(series[0]),
                        structural=structural,
                        actions=actions,
                        latent_vector=codes[index],
                        steps=1,
                    )
                    for value in state
                ]

            delta = [a - b for a, b in zip(roll(position), roll(other), strict=True)]
            norm = sum(value * value for value in delta) ** 0.5
            if norm == 0.0:
                zero += 1
            moves.append(norm)

        moves.sort()
        print(
            f"[codes] {code}: n={len(moves)} exactly-zero={zero} "
            f"median move={moves[len(moves) // 2]:.4e} max={moves[-1]:.4e}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
