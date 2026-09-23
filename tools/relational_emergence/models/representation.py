"""The predictive representation models, and their factorial controls.

The comparison the phase is built on is between a model that predicts through an
explicit **pairwise** relation bottleneck and one that predicts just as well
without it. Everything else is held fixed: same data, same split, same optimiser,
same parameter count, same checkpoint rule. Only the bottleneck and the temporal
ordering change between arms, which is what makes a difference in the
*abstraction* endpoints attributable to them rather than to capacity or to data.

Arms, as the plan's factorial:

======================  ==========  ===========  ==========
arm                     temporal    pairwise     labels
======================  ==========  ===========  ==========
``S-G``                 no          no           yes (ceiling)
``S-R``                 no          yes          yes (ceiling)
``StaticSSL-G``         no          no           no
``StaticSSL-R``         no          yes          no
``Shuffle-R``           destroyed   yes          no
``P-G``                 yes         no           no
``P-R``                 yes         yes          no
======================  ==========  ===========  ==========

The supervised arms are a reference, not a target: the phase does not ask that
``P-R`` beat ``S-R``, only that the predictive arms be comparable to each other.

Two properties are structural rather than learned, and both exist so the
endpoints have something to measure:

* the pair encoder sees **one snapshot**, never the future and never the history.
  A ``z_ij`` that had seen the future would make the transplant endpoint circular.
* ``z_ji`` comes from the *same* encoder with its arguments swapped, so a shared
  role transform is a real hypothesis about the encoder rather than a
  restatement of how the latents were built.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

ARM_NAMES: tuple[str, ...] = (
    "P-R",
    "P-G",
    "Shuffle-R",
    "StaticSSL-R",
    "StaticSSL-G",
    "S-R",
    "S-G",
)


@dataclass(frozen=True)
class ArmSpec:
    """What one arm is allowed to see. Everything else is shared."""

    name: str
    temporal: bool
    pairwise: bool
    shuffle_history: bool = False
    supervised_relation: bool = False

    @property
    def uses_relation_label(self) -> bool:
        return self.supervised_relation


ARMS: dict[str, ArmSpec] = {
    "P-R": ArmSpec("P-R", temporal=True, pairwise=True),
    "P-G": ArmSpec("P-G", temporal=True, pairwise=False),
    "Shuffle-R": ArmSpec("Shuffle-R", temporal=True, pairwise=True, shuffle_history=True),
    "StaticSSL-R": ArmSpec("StaticSSL-R", temporal=False, pairwise=True),
    "StaticSSL-G": ArmSpec("StaticSSL-G", temporal=False, pairwise=False),
    "S-R": ArmSpec("S-R", temporal=False, pairwise=True, supervised_relation=True),
    "S-G": ArmSpec("S-G", temporal=False, pairwise=False, supervised_relation=True),
}


@dataclass(frozen=True)
class ModelConfig:
    latent: int = 32
    hidden: int = 64
    depth: int = 2
    history: int = 4
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    relation_weight: float = 1.0
    """Weight on the auxiliary relation-classification head, for the ``S-*`` arms."""
    rollout_steps: int = 4
    """Steps of autoregressive prediction the loss is taken over.

    The plan forbids training on ``t -> t+1`` alone (section 18), and the reason is
    measurable rather than stylistic. A one-step residual is largely determined by
    the current state, so an encoder trained on it has no pressure to carry the
    mechanism at all -- it can leave the code nearly useless and the predictor
    will still fit. The consequence shows up as an endpoint that cannot move: with
    a one-step objective the transplant decoder produced bit-identical
    predictions under different codes, i.e. the harness reported the *code's*
    irrelevance when what it had actually measured was the *objective's*.

    Over a horizon the constraint is what decides the trajectory, and the state
    alone stops determining it.
    """

    def __post_init__(self) -> None:
        for name in ("latent", "hidden", "depth", "history", "epochs", "batch_size", "rollout_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.history < 2:
            raise ValueError("a history of one frame is not a history")


def _mlp(widths: Sequence[int]) -> Any:
    """A ReLU MLP over an explicit width list: ``[in, h1, ..., hk, out]``.

    The width list is explicit rather than a ``(in, hidden, out, depth)`` tuple
    because the tuple form is ambiguous about whether ``depth`` counts the output
    layer, and an off-by-one there silently adds a degenerate
    ``latent -> latent`` block to every arm -- which is what the first version of
    this function did.
    """
    import torch

    if len(widths) < 2:
        raise ValueError("an MLP needs an input and an output width")
    layers: list[Any] = []
    for index in range(len(widths) - 2):
        layers.append(torch.nn.Linear(widths[index], widths[index + 1]))
        layers.append(torch.nn.ReLU())
    layers.append(torch.nn.Linear(widths[-2], widths[-1]))
    return torch.nn.Sequential(*layers)


def _layer_parameters(widths: Sequence[int]) -> int:
    """Parameter count of :func:`_mlp` for the same width list."""
    return sum(
        widths[index] * widths[index + 1] + widths[index + 1]
        for index in range(len(widths) - 1)
    )


def _matched_global_hidden(
    pair_width: int, hidden: int, latent: int, depth: int
) -> int:
    """The hidden width that makes the global arms' pair block cost the same.

    The global arm encodes the scene once and emits two latents, so at the same
    width it would carry about 45% more parameters than the relational arm -- and
    a comparison between a bottleneck and a *larger* model measures capacity, not
    factorization. Solved by bisection because the width appears linearly and
    quadratically in the count.
    """
    target = _layer_parameters([pair_width, *([hidden] * depth), latent])
    low, high = 4, 4 * hidden
    for _ in range(40):
        middle = (low + high) // 2
        if _layer_parameters([pair_width * 2, *([middle] * depth), latent * 2]) < target:
            low = middle
        else:
            high = middle
    return max(4, (low + high) // 2)


def build_model(
    state_dim: int,
    struct_dim: int,
    action_dim: int,
    spec: ArmSpec,
    config: ModelConfig,
    geometry_dim: int = 4,
) -> Any:
    """One arm's network. Capacity is matched across arms by construction.

    The global arms are not smaller: they receive the same pair features
    concatenated rather than pooled through a shared encoder, and the width is
    chosen so the parameter count matches the relational arm's. A comparison
    between a bottleneck and a *smaller* model would measure capacity.

    ``state_dim`` is the *pair* width -- both bodies -- while the two object
    columns that go into the pair features are half of it each, and the relative
    geometry is a third block of its own size. Getting that decomposition wrong is
    not a shape error that shouts: a wrong width somewhere upstream is exactly how
    an arm silently trains on a different input from the one the report describes.
    """
    import torch

    if state_dim % 2 != 0:
        raise ValueError("the state must split into two objects of equal width")
    object_dim = state_dim // 2

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.spec = spec
            self.state_dim = state_dim
            self.struct_dim = struct_dim
            self.action_dim = action_dim
            self.object_dim = object_dim
            self.geometry_dim = geometry_dim
            self.history = config.history
            self.latent = config.latent
            self.pair_width = 2 * object_dim + struct_dim + geometry_dim
            if spec.pairwise:
                # One encoder, applied to each direction. The two latents are
                # then the same function of swapped arguments.
                self.pair_encoder = _mlp(
                    [self.pair_width, *([config.hidden] * config.depth), config.latent]
                )
            else:
                # The unordered scene, encoded once and split, at a width chosen
                # so this block costs what the relational one does.
                self.pair_encoder = _mlp(
                    [
                        self.pair_width * 2,
                        *(
                            [_matched_global_hidden(
                                self.pair_width, config.hidden, config.latent, config.depth
                            )]
                            * config.depth
                        ),
                        config.latent * 2,
                    ]
                )
            # The dynamics encoder reads the *probe's* own recent trajectory, so
            # its width is the object width times the history length -- not the
            # pair width. The supporter's motion reaches the predictor through the
            # pair latent, which is the separation the architecture is claiming.
            history_dim = self.history * object_dim if spec.temporal else object_dim
            self.dynamics_encoder = _mlp(
                [history_dim, *([config.hidden] * config.depth), config.latent]
            )
            predict_in = 2 * config.latent + action_dim
            self.predictor = _mlp(
                [predict_in, *([config.hidden] * config.depth), state_dim]
            )
            if spec.uses_relation_label:
                self.relation_head = _mlp(
                    [2 * config.latent, *([config.hidden] * config.depth), 4]
                )

        def pair_features(self, snapshot: Any, swap: bool = False) -> Any:
            """``(probe_state, supporter_state, structural, relative geometry)``.

            The relative geometry is expressed in the frame's own coordinates, so
            it is a property of the configuration rather than of the world axes.
            """
            si = snapshot["state_i"]
            sj = snapshot["state_j"]
            structural = snapshot["structural"]
            geometry = snapshot["geometry_i_to_j"]
            if swap:
                geometry = torch.stack(
                    [-geometry[:, 0], -geometry[:, 1], -geometry[:, 2], -geometry[:, 3]], dim=1
                )
                return torch.cat([sj, si, structural, geometry], dim=1)
            return torch.cat([si, sj, structural, geometry], dim=1)

        def pair_latents(self, snapshot: Any) -> tuple[Any, Any]:
            """``(z_ij, z_ji)``.

            The relational arms apply **one** encoder to each direction, so the
            two latents are the same function of swapped arguments and a shared
            role transform is a hypothesis rather than a restatement of how they
            were built. The global arms encode the unordered scene once and split
            it, so the two halves are not related by any shared function of the
            arguments -- which is exactly what "no pairwise bottleneck" means.

            Note the honest limitation of a two-object world: both arms receive
            the same inputs, differing in *factorization* rather than in
            information. The plan's ``P-G`` contrast presumes more objects, and
            with two it can only test whether parameter sharing across directions
            matters.
            """
            if self.spec.pairwise:
                return (
                    self.pair_encoder(self.pair_features(snapshot, swap=False)),
                    self.pair_encoder(self.pair_features(snapshot, swap=True)),
                )
            scene = self.pair_encoder(
                torch.cat(
                    [
                        self.pair_features(snapshot, swap=False),
                        self.pair_features(snapshot, swap=True),
                    ],
                    dim=1,
                )
            )
            return scene[:, : self.latent], scene[:, self.latent :]

        def encode(self, batch: dict[str, Any]) -> dict[str, Any]:
            """Everything that does not depend on the prediction horizon.

            Split out so the multi-step loss can roll the predictor without
            recomputing the encoders on every step -- and so the checkpoint rule
            and the endpoints both read exactly the same code.
            """
            z_ij, z_ji = self.pair_latents(batch)
            if self.spec.temporal:
                history = batch["history"]
                if self.spec.shuffle_history:
                    # Destroy the ordering but keep the same frames, so the arm
                    # differs from P-R in temporal order and nothing else.
                    order = torch.arange(history.shape[1] - 1, -1, -1, device=history.device)
                    history = history[:, order, :]
                dynamics_in = history.reshape(history.shape[0], -1)
            else:
                dynamics_in = batch["state_i"]
            return {
                "z_ij": z_ij,
                "z_ji": z_ji,
                # Both arms predict from both directions, so the difference
                # between them is how the pair was factorized and not how much of
                # it reached the predictor.
                "pooled": z_ij + z_ji,
                "m_i": self.dynamics_encoder(dynamics_in),
            }

        def forward(
            self, batch: dict[str, Any], steps: int = 1, ablate_pooled: bool = False
        ) -> dict[str, Any]:
            """One step, or ``steps`` of autoregressive prediction.

            ``steps > 1`` is what the loss uses and what the phase's section 18
            asks for. The encoders are evaluated once, on the true snapshot at
            ``t``: the code describes the relation *now*, and the action sequence
            is exogenous and given, so nothing about the pair needs to be
            re-derived as the rollout proceeds. That is also what keeps the
            temporal arm's history honest -- shuffling it can only matter through
            ``m_i``, which the rollout holds fixed.
            """
            encoded = self.encode(batch)
            # Zeroing the code is the arm's own sensitivity check: it answers
            # whether the predictor reads the bottleneck *at all*, independent of
            # whether the code is portable. Without it, a null on the transplant
            # endpoint cannot be told apart from a predictor that ignores the code
            # and a code that is not portable.
            if ablate_pooled:
                encoded["pooled"] = torch.zeros_like(encoded["pooled"])
            current = torch.cat([batch["state_i"], batch["state_j"]], dim=1)
            actions = batch["action_sequence"][:, :steps] if steps > 1 else batch["action"].unsqueeze(1)
            predictions = []
            for step in range(steps):
                current = self.predictor(
                    torch.cat([encoded["m_i"], encoded["pooled"], actions[:, step]], dim=1)
                )
                predictions.append(current)
            out = {
                "prediction": current,
                "predictions": predictions,
                "z_ij": encoded["z_ij"],
                "z_ji": encoded["z_ji"],
            }
            if self.spec.uses_relation_label:
                out["relation_logits"] = self.relation_head(
                    torch.cat([encoded["z_ij"], encoded["z_ji"]], dim=1)
                )
            return out

    return Model()


def parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_arm(
    arm: str,
    batch_stream: dict[str, Any],
    state_dim: int,
    struct_dim: int,
    action_dim: int,
    config: ModelConfig,
) -> dict[str, Any]:
    """Train one arm; the checkpoint is the minimum validation prediction loss.

    Selecting on any relation-facing metric -- a probe, a CCGP, a transplant
    score -- would tune each arm toward the endpoint it is about to be judged on.
    ADR 0007 forbids it, and nothing in this loop reads a relation label except
    the supervised arms' auxiliary head, which is their whole definition.
    """
    import torch

    spec = ARMS[arm]
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device(config.device)
    model = build_model(state_dim, struct_dim, action_dim, spec, config).to(device)

    train = batch_stream["train"]
    val = batch_stream["val"]
    for part_name, part in (("train", train), ("val", val)):
        if "future" not in part:
            raise ValueError(
                f"the {part_name} part carries no future steps; build it with "
                f"future_steps={config.rollout_steps}. The loss rolls the predictor, "
                "so a one-step part cannot be trained on -- and a KeyError from "
                "inside the loop would not say so."
            )
        if part["future"].shape[1] < config.rollout_steps:
            raise ValueError(
                f"the {part_name} part carries {part['future'].shape[1]} future steps "
                f"but the loss rolls {config.rollout_steps}; build it with "
                f"future_steps={config.rollout_steps}"
            )
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_val = math.inf
    best_epoch = -1
    best_state = None
    n_train = train["state_i"].shape[0]
    steps = config.rollout_steps

    def loss_on(part: dict[str, Any], train_mode: bool) -> Any:
        if train_mode:
            out = model(part, steps=steps)
        else:
            with torch.no_grad():
                out = model(part, steps=steps)
        loss = None
        for index, prediction in enumerate(out["predictions"]):
            term = torch.nn.functional.mse_loss(prediction, part["future"][:, index])
            loss = term if loss is None else loss + term
        loss = loss / len(out["predictions"])
        if spec.uses_relation_label and "relation" in part:
            loss = loss + config.relation_weight * torch.nn.functional.cross_entropy(
                out["relation_logits"], part["relation"]
            )
        return loss

    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, config.batch_size):
            index = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = loss_on({key: value[index] for key, value in train.items()}, True)
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_on(val, False))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint selected; the training loop did not run")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "arm": arm,
        "model": model,
        "val_loss": best_val,
        "best_epoch": best_epoch,
        "n_parameters": parameter_count(model),
    }
