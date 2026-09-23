"""Rollout-mode Level-2 oracle: train one step, evaluate over the horizon.

The direct head in :mod:`oracles` learns ``(X_0, F, A) -> X_1..X_H`` from one row
per episode. Measured on the frozen dataset it overfits across base contexts --
training loss 0.49-0.66, validation 0.80-0.97, best epoch 2-11 of 200 -- so all
three arms sit near the trivial predictor and ``delta_M`` is unmeasurable at
±0.002. This module trains the same three arms on ``(X_t, F, A_t) -> X_{t+1}``
and scores them by rolling the horizon out autoregressively.

Two things change and both matter. The training set is ``horizon`` times larger,
which is context diversity rather than repetition: every step of every episode is
a different state to condition on. And the function to learn is one transition
instead of a whole trajectory.

``oracles.OracleConfig`` argues against a rollout, on the grounds that compounded
early error would mix "who knows more about ``M``" with "who accumulated error
faster". That concern is real for an absolute number and is why the loss here is
*not* comparable to the direct mode's. It does not apply to ``delta_M``: the
three arms are paired, share weights and batch order, and differ only in the
label, so any compounding affects them alike — and to the extent a better ``M``
yields a better one-step prediction, the compounding *amplifies* the difference
that is being measured rather than confounding it. What the mode gives up is the
reading of ``delta_M`` as "information per step"; what it buys is a comparison
that is measurable at all.

The comparison stays paired and the checkpoint rule is unchanged: minimum
validation prediction loss, never a relation-facing metric (ADR 0007).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from .oracles import (
    BASE_MODE,
    M_SLOT_WIDTH,
    MECHANISM_INDEX,
    MODES,
    TRUE_MODE,
    ArmReport,
    OracleConfig,
    OracleDataset,
    SeedReport,
    Split,
    build_oracle,
    group_split,
    shuffled_mechanisms,
)

HORIZON_LOSS_WEIGHT_FLOOR = 1e-6


def per_step_action_width(dataset: OracleDataset) -> int:
    """The impulse-vector width at one step: ``w_a / horizon``."""
    if dataset.horizon <= 0 or dataset.w_a % dataset.horizon != 0:
        raise ValueError(
            f"the action block width {dataset.w_a} is not divisible by the horizon "
            f"{dataset.horizon}"
        )
    return dataset.w_a // dataset.horizon


def transition_tensors(
    dataset: OracleDataset, split: Sequence[int]
) -> tuple[Any, Any, Any, Any]:
    """``(x_t, f, a_t, x_next)`` for every (episode, step) of ``split``."""
    import torch

    # `w_y` is the per-step state width (8) while `w_a` is the whole action block
    # (horizon * 4), because `output_width` multiplies the horizon in but the
    # action block is stored flat. Indexing them the same way silently reshapes
    # the dataset into nonsense, so the per-step width is derived explicitly.
    states = torch.tensor(dataset.y, dtype=torch.float32).view(
        -1, dataset.horizon, dataset.w_y
    )
    actions = torch.tensor(dataset.a, dtype=torch.float32).view(
        -1, dataset.horizon, per_step_action_width(dataset)
    )
    initial = torch.tensor(dataset.x0, dtype=torch.float32).unsqueeze(1)
    previous = torch.cat([initial, states[:, :-1, :]], dim=1)
    rows = torch.tensor(list(split), dtype=torch.long)
    features = torch.tensor(dataset.f, dtype=torch.float32)
    return previous[rows], features[rows], actions[rows], states[rows]


def state_and_action_stats(
    dataset: OracleDataset, split: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Per-dimension means and standard deviations, from the training split alone.

    Computed here rather than reused from the direct mode's standardiser because
    the two standardise different things: that one scales a whole ``H * 8``
    trajectory as one vector, this one scales a single state. Reusing either for
    the other would leave the reported losses in units that do not mean what the
    report says they mean.
    """
    train = list(split)
    count = len(train)
    if count == 0:
        raise ValueError("the training split is empty")

    def column_stats(rows: list[tuple[float, ...]], width: int) -> tuple[tuple, tuple]:
        means = tuple(sum(row[dim] for row in rows) / count for dim in range(width))
        deviations = tuple(
            max(
                (sum((row[dim] - means[dim]) ** 2 for row in rows) / count) ** 0.5,
                HORIZON_LOSS_WEIGHT_FLOOR,
            )
            for dim in range(width)
        )
        return means, deviations

    # `dataset.y` is the whole trajectory flattened, so a state dimension lives
    # at every `step * w_y + dim`, not at `dim`. Pooling across the steps is what
    # gives a state scale; taking the first `w_y` columns would scale by step 1
    # alone, which is the state the bodies are *released* in and is far quieter
    # than the rest of the trajectory. Getting this wrong multiplies every
    # standardised state by a large factor -- it showed up as a training loss in
    # the hundreds, where a trivial predictor scores about one.
    horizon = dataset.horizon

    def state_column_stats(rows: list[tuple[float, ...]], width: int) -> tuple[tuple, tuple]:
        means = []
        deviations = []
        for dim in range(width):
            values = [row[step * width + dim] for row in rows for step in range(horizon)]
            mean = sum(values) / len(values)
            means.append(mean)
            deviations.append(
                max(
                    (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
                    HORIZON_LOSS_WEIGHT_FLOOR,
                )
            )
        return tuple(means), tuple(deviations)

    state_mean, state_std = state_column_stats(
        [dataset.y[index] for index in train], dataset.w_y
    )

    # The action block is stored flat, one four-component impulse vector per step,
    # so a column is a (step, component) pair rather than a component. Pooling
    # across steps is what gives a per-component scale -- and it matters, because
    # impulses do not start at step 0, so the early columns are near-constant and
    # a per-column scale would divide them by their own noise.
    per_step = per_step_action_width(dataset)
    horizon = dataset.horizon
    action_rows = [dataset.a[index] for index in train]
    total = count * horizon

    # The filler block is standardised too, and it is not optional: it holds the
    # masses, which run to about 18 where every other block is order 1. Left raw
    # it dominates the first layer and the outputs diverge -- visible as a
    # training loss in the hundreds where a trivial predictor would score about
    # one, which is how it was found.
    filler_mean, filler_std = column_stats(
        [dataset.f[index] for index in train], dataset.w_f
    )
    action_mean_list = []
    action_std_list = []
    for component in range(per_step):
        values = [
            row[step * per_step + component] for row in action_rows for step in range(horizon)
        ]
        mean = sum(values) / total
        action_mean_list.append(mean)
        action_std_list.append(
            max((sum((value - mean) ** 2 for value in values) / total) ** 0.5, HORIZON_LOSS_WEIGHT_FLOOR)
        )
    return (
        (*state_mean, *filler_mean, *action_mean_list),
        (*state_std, *filler_std, *action_std_list),
    )


def _slot(
    indices: Any, n_episodes: int, mode: str
) -> Any:
    """The four-unit ``M`` slot, left outside the standardisation on purpose."""
    import torch

    slot = torch.zeros((n_episodes, M_SLOT_WIDTH), dtype=torch.float32)
    if mode != BASE_MODE:
        slot[torch.arange(n_episodes), indices] = 1.0
    return slot


def fit_rollout_arm(
    dataset: OracleDataset,
    split: Split,
    labels: Sequence[str],
    mode: str,
    config: OracleConfig,
    seed: int,
) -> ArmReport:
    import torch

    # Identical weights and identical batch order across the arms of a seed; the
    # only difference is the label. Same discipline as the direct mode.
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    device = torch.device(config.device)

    stats = state_and_action_stats(dataset, split.train)
    means, deviations = stats
    state_mean = torch.tensor(means[: dataset.w_y], dtype=torch.float32)
    state_std = torch.tensor(deviations[: dataset.w_y], dtype=torch.float32)
    filler_mean = torch.tensor(
        means[dataset.w_y : dataset.w_y + dataset.w_f], dtype=torch.float32
    )
    filler_std = torch.tensor(
        deviations[dataset.w_y : dataset.w_y + dataset.w_f], dtype=torch.float32
    )
    action_mean = torch.tensor(means[dataset.w_y + dataset.w_f :], dtype=torch.float32)
    action_std = torch.tensor(deviations[dataset.w_y + dataset.w_f :], dtype=torch.float32)

    train_x, train_f, train_a, train_y = transition_tensors(dataset, split.train)
    val_x, val_f, val_a, val_y = transition_tensors(dataset, split.val)

    train_x = (train_x - state_mean) / state_std
    train_a = (train_a - action_mean) / action_std
    train_f = (train_f - filler_mean) / filler_std
    train_y = (train_y - state_mean) / state_std
    val_x = (val_x - state_mean) / state_std
    val_a = (val_a - action_mean) / action_std
    val_f = (val_f - filler_mean) / filler_std
    val_y = (val_y - state_mean) / state_std

    # Predict the *change* in state, not the state. The identity map then
    # predicts "nothing moves", which is a respectable baseline for a one-step
    # physics transition, and an autoregressive rollout inherits that stability
    # instead of compounding absolute errors away from the data manifold.
    train_delta = train_y - train_x

    horizon = dataset.horizon
    model = build_oracle(
        dataset.w_y + dataset.w_f + per_step_action_width(dataset) + M_SLOT_WIDTH,
        dataset.w_y,
        config,
    )
    model = model.to(device)

    train_labels = torch.tensor(
        [MECHANISM_INDEX[labels[index]] for index in split.train], dtype=torch.long
    )
    val_labels = torch.tensor(
        [MECHANISM_INDEX[labels[index]] for index in split.val], dtype=torch.long
    )
    train_slot = _slot(train_labels, len(split.train), mode).to(device)
    val_slot = _slot(val_labels, len(split.val), mode).to(device)

    flat_train = torch.cat(
        [
            train_x.reshape(-1, dataset.w_y),
            train_f.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, dataset.w_f),
            train_a.reshape(-1, per_step_action_width(dataset)),
            train_slot.unsqueeze(1).expand(-1, horizon, -1).reshape(-1, M_SLOT_WIDTH),
        ],
        dim=1,
    ).to(device)
    flat_targets = train_delta.reshape(-1, dataset.w_y).to(device)

    scored_horizon = min(config.eval_horizon, horizon)

    def rollout_loss() -> float:
        """MSE over the rolled-out interval, in the same standardised state units."""
        model.eval()
        with torch.no_grad():
            current = val_x[:, 0, :].to(device)
            total = 0.0
            for step in range(scored_horizon):
                # The filler block is per episode, not per step: it is the same
                # 16 numbers at every step, so it is concatenated unchanged.
                inputs = torch.cat(
                    [
                        current,
                        val_f.to(device),
                        val_a[:, step, :].to(device),
                        val_slot,
                    ],
                    dim=1,
                )
                current = current + model(inputs)
                total += float(
                    torch.nn.functional.mse_loss(
                        current, val_y[:, step, :].to(device), reduction="sum"
                    )
                )
            return total / (val_x.shape[0] * scored_horizon * dataset.w_y)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    n_train = flat_train.shape[0]
    best_val = math.inf
    best_state = None
    best_epoch = -1
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(n_train, generator=generator)
        for start in range(0, n_train, config.batch_size):
            batch = order[start : start + config.batch_size]
            optimiser.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(
                model(flat_train[batch]), flat_targets[batch]
            )
            loss.backward()
            optimiser.step()
        val_loss = rollout_loss()
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("no checkpoint was selected; the training loop did not run")
    model.load_state_dict(best_state)
    model.train()
    with torch.no_grad():
        train_loss = float(
            torch.nn.functional.mse_loss(model(flat_train), flat_targets, reduction="mean")
        )
    return ArmReport(
        seed=seed,
        mode=mode,
        train_loss=train_loss,
        val_loss=best_val,
        best_epoch=best_epoch,
        epochs=config.epochs,
        n_parameters=sum(parameter.numel() for parameter in model.parameters()),
    )


def run_rollout_seed(dataset: OracleDataset, config: OracleConfig, seed: int) -> SeedReport:
    """The rollout counterpart of :func:`oracles.run_seed`, same three arms."""
    split = group_split(dataset, config.val_fraction, seed)
    if split.overlaps():
        raise ValueError(f"seed {seed}: the split leaked groups {split.overlaps()!r}")
    shuffled = shuffled_mechanisms(dataset, seed)

    arms: dict[str, ArmReport] = {}
    for mode in MODES:
        labels = dataset.mechanism if mode == TRUE_MODE else shuffled
        arms[mode] = fit_rollout_arm(
            dataset=dataset, split=split, labels=labels, mode=mode, config=config, seed=seed
        )
        report = arms[mode]
        print(
            f"[level2] seed={seed} arm={mode:8s} rollout_val={report.val_loss:.6f} "
            f"train={report.train_loss:.6f} best_epoch={report.best_epoch}"
        )
    parameter_counts = {report.n_parameters for report in arms.values()}
    if len(parameter_counts) != 1:
        raise ValueError(
            f"seed {seed}: the arms are not capacity-identical, parameter counts "
            f"{sorted(parameter_counts)!r}"
        )
    return SeedReport(
        seed=seed,
        n_train=len(split.train) * dataset.horizon,
        n_val=len(split.val) * dataset.horizon,
        val_groups=split.val_groups,
        standardiser_signature="rollout-state-action",
        arms=arms,
    )
