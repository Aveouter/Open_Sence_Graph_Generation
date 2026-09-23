"""At what horizon does the mechanism start to matter to a decoder?

The transplant endpoint came back null and the must-fire control *passed* -- the
decoder reads a one-hot mechanism label and ignores the learned code. Before that
can be reported as a fact about the representation, the alternative has to be
ruled out: that the mechanism simply does not affect the trajectory over the
horizon the decoder is trained on, so no code could have helped.

The test is a pair of training runs at each horizon, identical except that the
second is fed a *deranged* one-hot -- the same code alphabet, the wrong label.
Where those two losses separate, the mechanism is doing work; where they do not,
no code can be read and the endpoint has no power there.
"""

from __future__ import annotations

import argparse

from ..eval import design
from ..simulator.splits import DEFAULT_SPLIT_SEED, build_split
from .run_phase1a import build_groups
from .run_phase1b import _rows_for, oracle_codes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups-per-tuple", type=int, default=12)
    parser.add_argument("--decoder-epochs", type=int, default=60)
    args = parser.parse_args(argv)

    from ..config import Phase1AConfig
    from ..models.batch_stream import StreamSpec, action_dim, build_part, struct_dim
    from ..models.transplant import DecoderConfig, train_decoder
    from ..simulator.dataset import build_rows

    config = Phase1AConfig(groups_per_tuple=args.groups_per_tuple)
    groups = build_groups(config)
    rows = [row.as_dict() for row in build_rows(groups, config.horizon)]
    split = build_split(groups, "unseen_family", seed=DEFAULT_SPLIT_SEED)
    partition = design.partition_contexts(split.train, split.test, seed=DEFAULT_SPLIT_SEED + 1)
    fit_rows = _rows_for(rows, partition.fit)

    print("[horizon] loss of the same decoder under a true label and a deranged one")
    for steps in (1, 2, 4, 6, 8, 12, 20):
        spec = StreamSpec(history=4, horizon=steps)
        part = build_part(fit_rows, spec, with_relation=False, future_steps=steps)
        honest = oracle_codes(fit_rows, spec)
        # The control shifts the labels by one whole episode, so same code
        # alphabet and same marginal with the pairing broken. It is a blunt
        # derangement rather than a within-episode one, which is the conservative
        # direction: it leaves some rows correctly labelled by accident, so the
        # gap it reports is a lower bound on what a true derangement would show.
        import torch

        deranged = honest[torch.arange(honest.shape[0]) - spec.horizon]
        losses = {}
        for name, codes in (("true", honest), ("deranged", deranged)):
            losses[name] = train_decoder(
                part,
                codes,
                state_dim=8,
                struct_dim=struct_dim(),
                action_dim=action_dim(),
                latent=int(honest.shape[1]),
                config=DecoderConfig(epochs=args.decoder_epochs, rollout_steps=steps, seed=0),
            )["final_loss"]
        gap = losses["deranged"] - losses["true"]
        print(
            f"[horizon]   steps={steps:2d} true={losses['true']:.6f} "
            f"deranged={losses['deranged']:.6f} gap={gap:+.6f} "
            f"({100 * gap / max(losses['true'], 1e-12):+.1f}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
