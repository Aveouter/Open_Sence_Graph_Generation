"""Probe architectures B1_add .. B4.

These are deliberately small MLPs over frozen inputs.  The experiment is testing
the *dataset and ontology*, not model capacity, so a failure has to be
attributable to the data rather than to an elaborate architecture -- and a
success has to be attributable to the features rather than to the probe.

``B1_add`` is the load-bearing control.  A flat pair lookup (``B1_lookup``) is
worthless on unseen pairs by construction, so if ``Δ_visual`` were measured
against it any visual gain would be overstated: a *factored* prior over
``(c_s, c_o)`` embeddings can generalise compositionally to unseen pairs on its
own, and that is the bar visual evidence actually has to clear.

Geometry reuses ``src.models.motifs._pair_geometry`` so the geometry-only gain is
defined identically to the repo's Motifs baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.motifs import _pair_geometry

NUM_OBJECT_SLOTS = 151  # ids 0..150; id 0 is unused but keeps indexing unconditional
GEOMETRY_DIM = 8
LABEL_DIM = 128

__all__ = [
    "PROBE_INPUTS",
    "ProbeMLP",
    "ProbeModel",
    "build_probe",
    "normalized_cxcywh",
    "geometry_from_boxes",
    "PROBE_NAMES",
]

#: Which feature blocks each probe consumes.  B0/B1_lookup are closed-form and
#: live in prior_baselines.py rather than here.
PROBE_INPUTS: dict[str, tuple[str, ...]] = {
    "B1_add": ("label_s", "label_o"),
    "B2": ("label_s", "label_o", "geometry"),
    "B3": ("visual_s", "visual_o", "visual_u"),
    "B4": (
        "label_s",
        "label_o",
        "geometry",
        "visual_s",
        "visual_o",
        "visual_u",
    ),
}
PROBE_NAMES = tuple(PROBE_INPUTS)


def normalized_cxcywh(
    sub_xyxy: torch.Tensor,
    obj_xyxy: torch.Tensor,
    image_size: torch.Tensor,
) -> torch.Tensor:
    """Absolute xyxy boxes -> normalized cxcywh, as the repo's models expect.

    Returns ``[2N, 4]``: subjects in rows ``0..N-1`` and objects in rows
    ``N..2N-1``, which is the box-major layout ``motifs._pair_geometry`` indexes
    into with an explicit pair list.  ``image_size`` is [N, 2] as (height,
    width).  Transposing height/width would still yield plausible-looking
    geometry, so the conversion is unit-tested against a hand-computed case.
    """
    height = image_size[:, :1].clamp(min=1.0)
    width = image_size[:, 1:2].clamp(min=1.0)

    def _one(box: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = box.unbind(-1)
        cx = (x1 + x2) / 2.0 / width[:, 0]
        cy = (y1 + y2) / 2.0 / height[:, 0]
        w = (x2 - x1) / width[:, 0]
        h = (y2 - y1) / height[:, 0]
        return torch.stack((cx, cy, w, h), dim=-1)

    return torch.cat((_one(sub_xyxy), _one(obj_xyxy)), dim=0)


def geometry_from_boxes(
    sub_xyxy: torch.Tensor,
    obj_xyxy: torch.Tensor,
    image_size: torch.Tensor,
) -> torch.Tensor:
    """``[N, 8]`` pair geometry in the repo's convention (see ``motifs._pair_geometry``)."""
    if not (sub_xyxy.shape == obj_xyxy.shape and sub_xyxy.shape[-1] == 4):
        raise ValueError(
            f"expected matching [N, 4] boxes, got {tuple(sub_xyxy.shape)} and "
            f"{tuple(obj_xyxy.shape)}"
        )
    boxes = normalized_cxcywh(sub_xyxy, obj_xyxy, image_size)
    n = sub_xyxy.shape[0]
    indices = torch.arange(n, device=boxes.device)
    geometry = _pair_geometry(boxes, torch.stack((indices, indices + n), dim=1))
    # _pair_geometry would happily emit a wider tensor if the box layout were
    # wrong, and the only symptom would be a downstream shape mismatch, so the
    # width is asserted here where the cause is obvious.
    if geometry.shape != (n, GEOMETRY_DIM):
        raise AssertionError(
            f"pair geometry has shape {tuple(geometry.shape)}, expected "
            f"{(n, GEOMETRY_DIM)}"
        )
    return geometry


class ProbeMLP(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout, twice, then a linear head."""

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden: Sequence[int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for width in hidden:
            layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, width),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = width
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ProbeModel(nn.Module):
    """Assembles the named feature blocks for one probe and classifies."""

    def __init__(
        self,
        probe: str,
        n_classes: int,
        visual_dim: int = 768,
        label_dim: int = LABEL_DIM,
        hidden: Sequence[int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if probe not in PROBE_INPUTS:
            raise ValueError(f"unknown probe {probe!r}; expected one of {PROBE_NAMES}")
        self.probe = probe
        self.inputs = PROBE_INPUTS[probe]

        self.label_embed_s: nn.Embedding | None = None
        self.label_embed_o: nn.Embedding | None = None
        if "label_s" in self.inputs or "label_o" in self.inputs:
            # One table per role: subject and object predicate statistics differ,
            # and sharing would force the probe to learn that asymmetry.
            self.label_embed_s = nn.Embedding(NUM_OBJECT_SLOTS, label_dim)
            self.label_embed_o = nn.Embedding(NUM_OBJECT_SLOTS, label_dim)

        block_dims = {
            "label_s": label_dim,
            "label_o": label_dim,
            "geometry": GEOMETRY_DIM,
            "visual_s": visual_dim,
            "visual_o": visual_dim,
            "visual_u": visual_dim,
        }
        # Per-block normalisation is load-bearing, not cosmetic. The blocks have
        # wildly different scales (visual CLS vectors have norm ~26; geometry
        # features are O(1) ratios), so normalising the *concatenation* lets the
        # 2304 visual dims set the statistics for the whole vector and the
        # semantic block barely registers. That would make B4 a visual-only model
        # with a semantic afterthought and understate what vision adds.
        self.block_norms = nn.ModuleDict(
            {name: nn.LayerNorm(block_dims[name]) for name in self.inputs}
        )
        in_dim = sum(block_dims[name] for name in self.inputs)
        self.mlp = ProbeMLP(in_dim, n_classes, hidden=hidden, dropout=dropout)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        for name in self.inputs:
            if name == "label_s":
                labels = features["labels_s"].clamp(0, NUM_OBJECT_SLOTS - 1)
                block = self.label_embed_s(labels)  # type: ignore[misc]
            elif name == "label_o":
                labels = features["labels_o"].clamp(0, NUM_OBJECT_SLOTS - 1)
                block = self.label_embed_o(labels)  # type: ignore[misc]
            else:
                block = features[name]
                if block.dim() > 2:
                    block = block.flatten(1)
                block = block.float()
            blocks.append(self.block_norms[name](block))
        return self.mlp(torch.cat(blocks, dim=-1))

    def describe(self) -> dict[str, Any]:
        return {
            "probe": self.probe,
            "inputs": list(self.inputs),
            "n_parameters": sum(p.numel() for p in self.parameters()),
        }


def build_probe(
    probe: str,
    n_classes: int,
    visual_dim: int = 768,
    label_dim: int = LABEL_DIM,
    hidden: Sequence[int] = (512, 256),
    dropout: float = 0.1,
) -> ProbeModel:
    return ProbeModel(
        probe,
        n_classes,
        visual_dim=visual_dim,
        label_dim=label_dim,
        hidden=hidden,
        dropout=dropout,
    )
