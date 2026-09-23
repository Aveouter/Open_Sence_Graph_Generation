"""The decoder that makes the transplant endpoint measurable.

The endpoint asks a question about the *encoder's* latents, but answering it
needs something that turns a latent back into a future: a decoder is trained on
the frozen encoder's outputs, then the encoder is held fixed and only the latent
it is fed is varied. Without this step the transplant would have to be scored
through the arm's own predictor, and the arms differ in their predictors -- so
every difference between arms would be a difference between predictors rather
than between codes.

The decoder is deliberately one small object shared by every arm. It is not part
of the comparison; it is the instrument the comparison is read through, and it is
trained identically for each arm so the readings are on one scale.

Autoregressive rather than open-loop over the horizon: the world is a dynamical
system, and a decoder that saw the true future state at every step would be
answering an easier question than the one the transplant needs answered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .batch_stream import OBJECT_DIM
from .representation import _mlp


ROLLOUT_STEPS = 4


@dataclass(frozen=True)
class DecoderConfig:
    hidden: int = 64
    depth: int = 2
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    rollout_steps: int = ROLLOUT_STEPS
    """Steps the decoder is trained to roll, and so the horizon it can be trusted at.

    One-step training is what the first version of this did, and it made the
    endpoint blind: the next state is largely determined by the current one, so
    the decoder minimised its loss without reading the code at all and reported
    the *code's* irrelevance when the truth was the *decoder's* indifference. It
    was measurable -- with a one-hot mechanism label as the code, 180 of 200
    swapped pairs produced a bit-identical prediction.

    Training on the rolled sequence removes that option, because within a few
    steps the constraint is the only thing that decides the trajectory and the
    state no longer determines it. The plan asks for the same thing of the
    predictors (section 18), for the same reason.
    """

    def __post_init__(self) -> None:
        for name in ("hidden", "depth", "epochs", "batch_size", "rollout_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")


def build_decoder(
    state_dim: int, struct_dim: int, action_dim: int, latent: int, config: DecoderConfig
) -> Any:
    import torch

    # `state_dim` is already the pair width -- both bodies -- so it is not
    # doubled. The four blocks are the scene, the objects' structural attributes,
    # the transplanted code, and the action.
    width = state_dim + struct_dim + latent + action_dim

    class Decoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = _mlp([width, *([config.hidden] * config.depth), state_dim])

        def forward(self, state: Any, structural: Any, latent_vector: Any, action: Any) -> Any:
            """The predicted *residual*, so the model does not have to relearn identity."""
            combined = torch.cat([state, structural, latent_vector, action], dim=1)
            return state + self.net(combined)

    return Decoder()


def freeze_latents(model: Any, part: dict[str, Any]) -> Any:
    """``z_ij + z_ji`` for every row, with no gradient path back to the encoder.

    The sum rather than the concatenation, and the same sum the arms' own
    predictors use: to read the transplant through a different combination than
    the one the arm was trained with would test the decoder's ability to
    interpret an unfamiliar code rather than the code's portability.
    """
    import torch

    with torch.no_grad():
        z_ij, z_ji = model.pair_latents(part)
        return z_ij + z_ji


def train_decoder(
    part: dict[str, Any],
    latents: Any,
    state_dim: int,
    struct_dim: int,
    action_dim: int,
    latent: int,
    config: DecoderConfig,
) -> dict[str, Any]:
    import torch

    if "future" not in part or "action_sequence" not in part:
        raise ValueError(
            "the decoder trains on a rolled sequence; build the part with "
            f"future_steps={config.rollout_steps}"
        )
    steps = config.rollout_steps
    if part["future"].shape[1] < steps:
        raise ValueError(
            f"the part carries {part['future'].shape[1]} future steps, need {steps}"
        )

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    device = torch.device(config.device)
    model = build_decoder(state_dim, struct_dim, action_dim, latent, config).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    state = torch.cat([part["state_i"], part["state_j"]], dim=1)
    structural = part["structural"]
    future = part["future"][:, :steps]
    actions = part["action_sequence"][:, :steps]
    n_rows = state.shape[0]

    def rollout_loss(selection: Any, train_mode: bool) -> Any:
        current = state[selection]
        block = structural[selection]
        code = latents[selection]
        loss = None
        for step in range(steps):
            if train_mode:
                nxt = model(current, block, code, actions[selection, step])
            else:
                with torch.no_grad():
                    nxt = model(current, block, code, actions[selection, step])
            term = torch.nn.functional.mse_loss(nxt, future[selection, step])
            loss = term if loss is None else loss + term
            # The predicted state is fed back: this is the quantity that has to be
            # right, not the one-step residual.
            current = nxt
        return loss / steps

    for _ in range(config.epochs):
        model.train()
        order = torch.randperm(n_rows, generator=generator)
        for start in range(0, n_rows, config.batch_size):
            index = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            rollout_loss(index, True).backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        # The selection is capped so the diagnostic is cheap on a large split; it
        # is a training readout, not an endpoint.
        sample = torch.arange(min(n_rows, 2048))
        final = float(rollout_loss(sample, False))
    return {"model": model, "final_loss": final}


def roll_from_state(
    decoder: Any,
    initial_state: list[float],
    structural: Any,
    actions: list[tuple[float, ...]],
    latent_vector: Any,
    steps: int,
) -> list[list[float]]:
    """Roll the decoder forward ``steps`` times from one initial state.

    The action is the *true* one at every step -- the schedule is exogenous and
    known, which is the same thing the arms' predictors get. The *state* is the
    predicted one, so error compounds exactly as it would in deployment, and a
    latent that fails to carry the constraint shows up as divergence rather than
    as a constant one-step offset.

    Premises are checked rather than assumed: a latent of the wrong width or an
    action block of the wrong shape would otherwise be broadcast by torch into a
    silently different computation, which is the failure class this repository
    treats as its worst.
    """
    import torch

    if len(initial_state) != 2 * OBJECT_DIM:
        raise ValueError(f"expected a {2 * OBJECT_DIM}-wide state, got {len(initial_state)}")
    if len(actions) < steps:
        raise ValueError(f"need {steps} actions, got {len(actions)}")
    if latent_vector.dim() == 1:
        # One code, one scene. The decoder's other inputs are batched, so the
        # latent is promoted rather than the call being special-cased.
        latent_vector = latent_vector.unsqueeze(0)
    current = [float(value) for value in initial_state]
    predicted: list[list[float]] = []
    for step in range(steps):
        state_tensor = torch.tensor([current], dtype=torch.float32)
        action_tensor = torch.tensor([list(actions[step])], dtype=torch.float32)
        with torch.no_grad():
            nxt = decoder(state_tensor, structural, latent_vector, action_tensor)
        if nxt.shape[-1] != 2 * OBJECT_DIM:
            raise ValueError(
                f"the decoder returned width {nxt.shape[-1]}, expected {2 * OBJECT_DIM}"
            )
        current = [float(value) for value in nxt[0]]
        predicted.append(current[:])
    return predicted
