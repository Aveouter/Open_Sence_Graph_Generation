"""Fixtures for the RelateAnything source-behaviour audits (#155, #156).

These helpers build a *tiny, CPU-only, deterministic* ``RelSGG`` so that a
training forward pass can be executed without any released checkpoint, dataset
or GPU. The DINOv3 backbone is replaced by a stub and the scene map is passed
in through ``precomputed_features``, so no weights are downloaded or loaded.

This is deliberately **not** a reproduction harness: nothing here measures
model quality, and no number produced with it is baseline evidence. It exists
so that the exact source under audit in ``src/modules/relateanything`` (whose
model and loss files are byte-identical to ``Maelic/RelateAnything`` at
``06766fdf56752ca535fc9b971fca99ce563676d0``) can be executed and its
contracts characterised.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

# Small enough that a full forward pass costs milliseconds on CPU.
BACKBONE_DIM = 32
D_MODEL = 16
TEXT_DIM = 16
SCENE_HW = 4
BOXES = 4


class StubBackbone(nn.Module):
    """Stand-in for DINOv3.

    ``RelSGG`` only reads ``d_model`` from the backbone as long as
    ``precomputed_features`` is supplied, so this needs no weights.
    """

    def __init__(
        self,
        model_name=None,
        patch_size=16,
        pretrained=True,
        layer_offsets=None,
        d_model=BACKBONE_DIM,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        # Kept so the stub's state dict has the same shape as the real one.
        self.layer_weights = nn.Parameter(torch.zeros(3))


@contextmanager
def stub_backbone():
    """Swap ``Backbone`` for :class:`StubBackbone` for the duration of the block."""
    from src.modules.relateanything.model import relsgg

    original = relsgg.Backbone
    relsgg.Backbone = StubBackbone
    try:
        yield
    finally:
        relsgg.Backbone = original


def tiny_config(cfa_prob: float = 0.0, **overrides):
    """Released-recipe config scaled down to a CPU-testable width."""
    from src.modules.relateanything.config import RelSGGConfig

    kwargs = dict(
        d_model=D_MODEL,
        text_dim=TEXT_DIM,
        n_heads=2,
        n_self_layers=1,
        n_cross_layers=1,
        n_dep_layers=1,
        n_gnd_layers=1,
        deformable_heads=1,
        # Determinism: no dropout, no modality dropout, no object-alignment
        # term (it needs a category vocabulary we do not have).
        dropout=0.0,
        box_token_dropout=0.0,
        lambda_obj=0.0,
        geo_budget=16,
        final_budget=8,
        # CFA: prob 1.0 always mixes, alpha 1e6 pins Beta(1e6,1e6) at 0.5, so
        # the only remaining randomness is *which* partner is drawn.
        cfa_prob=cfa_prob,
        cfa_alpha=1e6,
    )
    kwargs.update(overrides)
    return RelSGGConfig(**kwargs)


def build_model(cfa_prob: float, vocab: int = 8, seed: int = 0, **overrides):
    """A deterministic RelSGG in train mode with a synthetic vocabulary.

    ``overrides`` reach :func:`tiny_config`, so the released CFA settings can be
    exercised directly (``cfa_alpha=1.0`` with ``cfa_prob=0.5``).
    """
    from src.modules.relateanything.model.relsgg import RelSGG
    from src.modules.relateanything.training.losses import PredicateOntology

    with stub_backbone():
        torch.manual_seed(seed)
        model = RelSGG(tiny_config(cfa_prob=cfa_prob, **overrides))
    model.train()

    names = [f"p{i}" for i in range(vocab)]
    weights = torch.randn(vocab, TEXT_DIM, generator=torch.Generator().manual_seed(1))
    model.vocab_head.set_vocabulary_matrix(names, weights)
    model.install_losses(
        PredicateOntology(
            names,
            pos_w=torch.eye(vocab),
            neg_lw=torch.zeros(vocab, vocab),
            sym=torch.zeros(vocab),
            inverse_mask=torch.zeros(vocab, vocab, dtype=torch.bool),
        ),
        n_neg=4,
    )
    return model


def forward(model, relations, seed: int = 7, boxes_n: int = BOXES):
    """One training forward pass; identical RNG stream for every call."""
    boxes = torch.rand(
        1, boxes_n, 4, generator=torch.Generator().manual_seed(seed)
    ).clamp(0.1, 0.9)
    scene = torch.randn(
        1,
        SCENE_HW,
        SCENE_HW,
        BACKBONE_DIM,
        generator=torch.Generator().manual_seed(seed),
    )
    torch.manual_seed(seed)
    return model(
        images=torch.zeros(1, 3, SCENE_HW * 16, SCENE_HW * 16),
        boxes=boxes,
        box_counts=torch.tensor([boxes_n]),
        targets=[
            {
                "relations": torch.as_tensor(relations, dtype=torch.long),
                "entity_labels": torch.zeros(boxes_n, dtype=torch.long),
            }
        ],
        precomputed_features=scene,
    )


def gt_pred_labels(relations, n: int = BOXES):
    """``RelatednessPairSampler._gt_grid`` predicate grid, ``[n*n]``."""
    from src.modules.relateanything.model.sampler import RelatednessPairSampler

    _, labels = RelatednessPairSampler._gt_grid(
        [{"relations": torch.as_tensor(relations, dtype=torch.long)}], 1, n, "cpu"
    )
    return labels[0]


def cfa_partner_groups(relations, n: int = BOXES):
    """Predicate id -> ordered pairs that carry it.

    ``_gt_grid`` keeps one predicate per pair (the last one scattered), and
    ``RelSGG._mix_partners`` treats slots sharing that value as partners, so
    a group is *partner-eligible* exactly when it holds more than one slot.
    """
    labels = gt_pred_labels(relations, n)
    groups: dict[int, list[tuple[int, int]]] = {}
    for slot in (labels >= 0).nonzero(as_tuple=True)[0].tolist():
        groups.setdefault(int(labels[slot]), []).append((slot // n, slot % n))
    return groups


def eligible_partners(relations, n: int = BOXES):
    """Just the partner-eligible groups (more than one slot)."""
    return {
        g: pairs
        for g, pairs in cfa_partner_groups(relations, n).items()
        if len(pairs) > 1
    }
