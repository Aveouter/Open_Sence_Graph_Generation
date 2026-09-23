"""History-inferred mechanism codes and a shared train/eval residual rollout."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .protocol import Protocol

ARMS = ("P-R", "P-G", "Shuffle-R", "InitialStatic-R", "PostState-R")


@dataclass(frozen=True)
class ModelConfig:
    latent: int = 16
    hidden: int = 64
    depth: int = 2
    epochs: int = 40
    batch_size: int = 128
    learning_rate: float = 1e-3

    def __post_init__(self):
        if min(self.latent, self.hidden, self.depth, self.epochs, self.batch_size) < 1:
            raise ValueError("model dimensions and training counts must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning rate must be positive")


def mlp(inputs, outputs, config, hidden=None):
    widths = [inputs, *([hidden or config.hidden] * config.depth), outputs]
    layers = []
    for i, (left, right) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
        layers.append(nn.Linear(left, right))
        if i < len(widths) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def shuffle_frames(frames, generator=None):
    """Independent uniform permutations per episode; no permutation reaches q_M.

    All objects and incoming actions move together, preserving the frame multiset.
    This removes the fixed slot-to-time map, but physical content might still
    reveal chronology. The development chronology diagnostic gates that claim.
    """
    order = torch.rand(
        frames.shape[:2], generator=generator, device=frames.device
    ).argsort(dim=1)
    return frames.gather(1, order[:, :, None, None].expand_as(frames)), order


class Representation(nn.Module):
    def __init__(
        self, arm: str, protocol: Protocol, config: ModelConfig, global_hidden=None
    ):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown v2 arm: {arm}")
        self.arm, self.protocol, self.config = arm, protocol, config
        self.pairwise = arm != "P-G"
        self.edges = tuple(
            (i, j)
            for i in range(protocol.objects)
            for j in range(protocol.objects)
            if i != j
        )
        n, latent = protocol.objects, config.latent
        self.static = arm in ("InitialStatic-R", "PostState-R")
        history_width = (protocol.cutoff + 1) * 8 + 10
        if self.pairwise:
            self.state_encoder = mlp(14, latent, config)
            # Static arms retain capacity: fill the same slots with the named
            # snapshot and zero actions, without granting any extra observations.
            self.mechanism_encoder = mlp(history_width, latent, config)
            self.self_update = mlp(11, 4, config)
            self.interaction_update = mlp(11 + 2 * latent, 4, config)
        else:
            self.scene_encoder = mlp(
                (protocol.cutoff + 1) * n * 4 + n * 5, latent, config, global_hidden
            )
            self.scene_state = mlp(n * 9, latent, config, global_hidden)
            self.scene_update = mlp(n * 6 + 2 * latent, n * 4, config, global_hidden)

    def observed_frames(self, batch, generator=None):
        if self.static:
            key = (
                "initial_positions"
                if self.arm == "InitialStatic-R"
                else "post_positions"
            )
            positions = batch[key]
            frame = torch.cat([positions, torch.zeros_like(positions)], dim=-1)
            return frame[:, None].expand(-1, self.protocol.cutoff + 1, -1, -1)
        frames = batch["frames"]
        if self.arm == "Shuffle-R":
            frames, _ = shuffle_frames(frames, generator)
        return frames

    def infer_mechanism(self, batch, generator=None):
        frames, structural = self.observed_frames(batch, generator), batch["structural"]
        if self.pairwise:
            features = [
                torch.cat(
                    [
                        frames[:, :, i].flatten(1),
                        frames[:, :, j].flatten(1),
                        structural[:, i],
                        structural[:, j],
                    ],
                    dim=-1,
                )
                for i, j in self.edges
            ]
            return self.mechanism_encoder(torch.stack(features, dim=1))
        return self.scene_encoder(
            torch.cat([frames.flatten(1), structural.flatten(1)], dim=-1)
        )

    def snapshot_codes(self, state, structural):
        if not self.pairwise:
            return self.scene_state(
                torch.cat([state.flatten(1), structural.flatten(1)], dim=-1)
            )
        features = [
            torch.cat(
                [state[:, i, :2], state[:, j, :2], structural[:, i], structural[:, j]],
                dim=-1,
            )
            for i, j in self.edges
        ]
        return self.state_encoder(torch.stack(features, dim=1))

    def transition(self, current, structural, action, mechanism, *, block_edges=False):
        state_code = self.snapshot_codes(current, structural)
        if not self.pairwise:
            return current + self.scene_update(
                torch.cat(
                    [current.flatten(1), action.flatten(1), state_code, mechanism],
                    dim=-1,
                )
            ).view_as(current)
        # (i,j) is the directed j -> i message. No raw other-object state or
        # history reaches the update except through these explicit edge codes.
        if block_edges:
            state_code, mechanism = (
                torch.zeros_like(state_code),
                torch.zeros_like(mechanism),
            )
        own = torch.cat([current, structural, action], dim=-1)
        delta = self.self_update(own)
        messages = self.interaction_update(
            torch.stack(
                [
                    torch.cat(
                        [own[:, i], state_code[:, edge], mechanism[:, edge]], dim=-1
                    )
                    for edge, (i, _) in enumerate(self.edges)
                ],
                dim=1,
            )
        )
        incoming = torch.stack(
            [
                messages[
                    :, [e for e, (i, _) in enumerate(self.edges) if i == node]
                ].sum(1)
                for node in range(self.protocol.objects)
            ],
            dim=1,
        )
        return current + delta + incoming / (self.protocol.objects - 1)

    def rollout(
        self, state, structural, actions, mechanism, *, steps=None, block_edges=False
    ):
        count = actions.shape[1] if steps is None else steps
        if not 1 <= count <= actions.shape[1]:
            raise ValueError("requested rollout exceeds provided actions")
        current, predictions = state, []
        for step in range(count):
            current = self.transition(
                current,
                structural,
                actions[:, step],
                mechanism,
                block_edges=block_edges,
            )
            predictions.append(current)
        return torch.stack(predictions, dim=1)

    def forward(
        self, batch, *, steps=None, mechanism=None, generator=None, block_edges=False
    ):
        code = (
            self.infer_mechanism(batch, generator) if mechanism is None else mechanism
        )
        return self.rollout(
            batch["state"],
            batch["structural"],
            batch["actions"],
            code,
            steps=steps,
            block_edges=block_edges,
        )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def build_model(arm, protocol, config):
    if arm != "P-G":
        return Representation(arm, protocol, config)
    # Search counts only, never outcomes. fork_rng makes matching independent
    # of model initialization and consumes no training RNG draws.
    with torch.random.fork_rng():
        target = parameter_count(Representation("P-R", protocol, config))
        widths = range(4, 4 * config.hidden + 1)
        hidden = min(
            widths,
            key=lambda h: abs(
                parameter_count(Representation("P-G", protocol, config, h)) - target
            ),
        )
    model = Representation(arm, protocol, config, hidden)
    if abs(parameter_count(model) - target) / target > 0.01:
        raise ValueError("no global width matches active parameter count within 1%")
    return model


def select_batch(batch, indices):
    return {key: value[indices] for key, value in batch.items()}


def train_model(arm, protocol, config, batch, targets, split, seed):
    """Validation prediction loss is the sole checkpoint criterion for every arm."""
    torch.manual_seed(seed)
    model = build_model(arm, protocol, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(seed)
    train = torch.tensor(split["train"])
    val_batch, val_target = select_batch(batch, split["val"]), targets[split["val"]]
    best, state, best_epoch = float("inf"), None, None
    for epoch in range(config.epochs):
        model.train()
        order = train[torch.randperm(len(train), generator=generator)]
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(
                model(select_batch(batch, indices), generator=generator),
                targets[indices],
            )
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            # A reproducible random validation draw, never a deterministic reversal.
            validation_rng = torch.Generator().manual_seed(seed + 100000)
            loss = float(
                nn.functional.mse_loss(
                    model(val_batch, generator=validation_rng), val_target
                )
            )
        if loss < best:
            best, best_epoch = loss, epoch
            state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
    if state is None:
        raise ValueError("no finite validation prediction checkpoint")
    model.load_state_dict(state)
    model.eval()
    return model, {
        "seed": seed,
        "arm": arm,
        "validation_prediction_loss": best,
        "best_epoch": best_epoch,
        "parameters": parameter_count(model),
    }
