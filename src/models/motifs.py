"""
Scene-Graph-Benchmark/Kaihua style Neural Motifs for OpenSGG.

The default Motifs path in this file is aligned with the public
Scene-Graph-Benchmark.pytorch ``MotifPredictor`` checkpoint layout:

  ROI features + object label embeddings + 9D box encodings
    -> bidirectional object context LSTM
    -> object decoder / GT labels
    -> bidirectional edge context LSTM
    -> post_emb -> subject/object pair concatenation
    -> post_cat -> visual gating -> rel_compress
    -> frequency bias P(predicate | subject, object)

OpenSGG still supplies its native per-image tensors, so the implementation keeps
the OpenSGG method interface while using module names and tensor shapes that can
load the SGB/Kaihua Motifs relation-head checkpoints.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _pure_predicate_count(rel_nums: int) -> int:
    """Return actual predicate count from OpenSGG's bg-inclusive convention."""
    return rel_nums - 1 if rel_nums > 50 else rel_nums


def generate_object_pairs(num_objects: int, device: torch.device) -> torch.Tensor:
    """Generate all directed object pairs (i, j), i != j."""
    idx = torch.arange(num_objects, device=device)
    subj, obj = torch.meshgrid(idx, idx, indexing="ij")
    keep = subj != obj
    return torch.stack((subj[keep], obj[keep]), dim=-1)


class FrequencyBias(nn.Module):
    """Legacy marginal frequency bias kept for other OpenSGG baselines.

    Older local models import ``FrequencyBias`` from this module and call it as
    ``logits = freq_bias(logits)``.  Neural Motifs itself uses
    ``PairFrequencyBias`` below.
    """

    def __init__(self, num_predicates: int, eps: float = 1e-12):
        super().__init__()
        self.eps = eps
        self.register_buffer("bias", torch.zeros(num_predicates))

    def load_freq_bias(self, distribution: torch.Tensor) -> None:
        self.bias.copy_(torch.log(distribution + self.eps))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + self.bias


class ObjectEncoder(nn.Module):
    """Legacy object feature encoder shared by older local baselines."""

    def __init__(
        self,
        visual_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, hidden_dim // 2)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, visual_feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        visual = self.visual_proj(visual_feats)
        label = self.label_embed(labels.clamp(min=0, max=self.label_embed.num_embeddings - 1))
        return self.dropout(torch.cat((visual, label), dim=-1))


class PairFeatureGenerator(nn.Module):
    """Legacy pair feature generator shared by older local baselines."""

    def __init__(self, obj_feat_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(obj_feat_dim * 2 + 10, hidden_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _union_box_features(sub_box: torch.Tensor, obj_box: torch.Tensor) -> torch.Tensor:
        sx, sy, sw, sh = sub_box.unbind(-1)
        ox, oy, ow, oh = obj_box.unbind(-1)
        sx1, sy1 = sx - sw / 2, sy - sh / 2
        sx2, sy2 = sx + sw / 2, sy + sh / 2
        ox1, oy1 = ox - ow / 2, oy - oh / 2
        ox2, oy2 = ox + ow / 2, oy + oh / 2
        ux1, uy1 = torch.minimum(sx1, ox1), torch.minimum(sy1, oy1)
        ux2, uy2 = torch.maximum(sx2, ox2), torch.maximum(sy2, oy2)
        uw, uh = (ux2 - ux1).clamp(min=0), (uy2 - uy1).clamp(min=0)
        return torch.stack(((ux1 + ux2) / 2, (uy1 + uy2) / 2, uw, uh, uw * uh), dim=-1)

    def forward(
        self,
        obj_feats: torch.Tensor,
        boxes: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> torch.Tensor:
        sub_feat = obj_feats[pair_indices[:, 0]]
        obj_feat = obj_feats[pair_indices[:, 1]]
        sub_box = boxes[pair_indices[:, 0]]
        obj_box = boxes[pair_indices[:, 1]]
        geom = _pair_geometry(boxes, pair_indices)
        union = self._union_box_features(sub_box, obj_box)
        # _pair_geometry returns 8 features; extract the 5-feature legacy
        # subset [dx, dy, center_dist, area_ratio, iou] that matches the
        # original _compute_spatial_feat convention used by VCTree / CVC /
        # TransformerSGG checkpoints.
        spatial = geom[:, [0, 1, 5, 4, 6]]
        pair_feat = torch.cat((sub_feat, obj_feat, spatial, union), dim=-1)
        return self.proj(pair_feat)


def _onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(labels.clamp(min=0, max=num_classes - 1), num_classes).float()


class _SafeBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running stats for single-object images."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.training and input.size(0) <= 1:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                self.momentum,
                self.eps,
            )
        return super().forward(input)


def _pair_geometry(boxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Geometry features for cxcywh-normalized boxes."""
    sb = boxes[pairs[:, 0]]
    ob = boxes[pairs[:, 1]]

    sx, sy, sw, sh = sb.unbind(-1)
    ox, oy, ow, oh = ob.unbind(-1)

    dx = (sx - ox) / (ow + 1e-6)
    dy = (sy - oy) / (oh + 1e-6)
    log_wr = torch.log((sw + 1e-6) / (ow + 1e-6))
    log_hr = torch.log((sh + 1e-6) / (oh + 1e-6))
    area_ratio = (sw * sh) / (ow * oh + 1e-6)
    center_dist = torch.sqrt(dx.square() + dy.square())

    sx1, sy1 = sx - sw / 2, sy - sh / 2
    sx2, sy2 = sx + sw / 2, sy + sh / 2
    ox1, oy1 = ox - ow / 2, oy - oh / 2
    ox2, oy2 = ox + ow / 2, oy + oh / 2
    inter_w = (torch.minimum(sx2, ox2) - torch.maximum(sx1, ox1)).clamp(min=0)
    inter_h = (torch.minimum(sy2, oy2) - torch.maximum(sy1, oy1)).clamp(min=0)
    inter = inter_w * inter_h
    union = sw * sh + ow * oh - inter
    iou = inter / (union + 1e-6)

    ux1, uy1 = torch.minimum(sx1, ox1), torch.minimum(sy1, oy1)
    ux2, uy2 = torch.maximum(sx2, ox2), torch.maximum(sy2, oy2)
    union_area = (ux2 - ux1).clamp(min=0) * (uy2 - uy1).clamp(min=0)

    return torch.stack(
        (dx, dy, log_wr, log_hr, area_ratio, center_dist, iou, union_area),
        dim=-1,
    )


def _box_info_from_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Return official-style 8D box info from normalized cxcywh boxes."""
    if boxes.numel() == 0:
        return boxes.new_zeros(0, 8)
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((x1, y1, x2, y2, cx, cy, w, h), dim=-1)


def _box_info_from_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Return official-style 8D box info from normalized xyxy boxes."""
    if boxes.numel() == 0:
        return boxes.new_zeros(0, 8)
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack((x1, y1, x2, y2, cx, cy, w, h), dim=-1)


def _pair_box_info(boxes: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Official TDE 32D pair-box feature: subject, object, union, intersection."""
    if pairs.numel() == 0:
        return boxes.new_zeros(0, 32)
    box_info = _box_info_from_cxcywh(boxes)
    box1 = box_info[pairs[:, 0]]
    box2 = box_info[pairs[:, 1]]

    union_xyxy = box1[:, :4].clone()
    union_xyxy[:, 0] = torch.minimum(box1[:, 0], box2[:, 0])
    union_xyxy[:, 1] = torch.minimum(box1[:, 1], box2[:, 1])
    union_xyxy[:, 2] = torch.maximum(box1[:, 2], box2[:, 2])
    union_xyxy[:, 3] = torch.maximum(box1[:, 3], box2[:, 3])
    union_info = _box_info_from_xyxy(union_xyxy)

    inter_xyxy = box1[:, :4].clone()
    inter_xyxy[:, 0] = torch.maximum(box1[:, 0], box2[:, 0])
    inter_xyxy[:, 1] = torch.maximum(box1[:, 1], box2[:, 1])
    inter_xyxy[:, 2] = torch.minimum(box1[:, 2], box2[:, 2])
    inter_xyxy[:, 3] = torch.minimum(box1[:, 3], box2[:, 3])
    invalid = (inter_xyxy[:, 2] < inter_xyxy[:, 0]) | (inter_xyxy[:, 3] < inter_xyxy[:, 1])
    inter_info = _box_info_from_xyxy(inter_xyxy)
    if invalid.any():
        inter_info[invalid] = 0

    return torch.cat((box1, box2, union_info, inter_info), dim=-1)


class PairFrequencyBias(nn.Module):
    """P(predicate | subject class, object class) log-frequency bias.

    SGB/Kaihua checkpoints store this table as
    ``freq_bias.obj_baseline.weight`` with shape
    ``[num_object_classes ** 2, num_predicates]``.  The table can also be
    initialized from OpenSGG's COCO-style ``train.json`` and ``rel.json`` when
    no external checkpoint is loaded.
    """

    def __init__(
        self,
        num_objects: int,
        num_predicates: int,
        eps: float = 1e-3,
        data_root: Optional[str] = None,
        predicate_bg_index: Optional[str] = None,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.num_predicates = num_predicates
        self.eps = eps
        self.predicate_bg_index = predicate_bg_index

        prior = self._build_uniform_prior()
        if data_root:
            loaded = self._build_vg_prior(data_root)
            if loaded is not None:
                prior = loaded

        self.obj_baseline = nn.Embedding(num_objects * num_objects, num_predicates)
        self.obj_baseline.weight.data.copy_(prior.view(-1, num_predicates))
        self.obj_baseline.weight.requires_grad_(False)

    def load_freq_bias(self, distribution: torch.Tensor) -> None:
        """Load a precomputed ``[num_objects, num_objects, num_predicates]``
        log-probability tensor."""
        if distribution.shape != (
            self.num_objects,
            self.num_objects,
            self.num_predicates,
        ):
            raise ValueError(
                f"Expected shape "
                f"({self.num_objects}, {self.num_objects}, {self.num_predicates}), "
                f"got {tuple(distribution.shape)}"
            )
        self.obj_baseline.weight.data.copy_(
            distribution.reshape(-1, self.num_predicates)
        )

    def _build_uniform_prior(self) -> torch.Tensor:
        prob = torch.full(
            (self.num_objects, self.num_objects, self.num_predicates),
            1.0 / max(self.num_predicates, 1),
        )
        return torch.log(prob + self.eps)

    def _build_vg_prior(self, data_root: str) -> Optional[torch.Tensor]:
        train_path = os.path.join(data_root, "train.json")
        rel_path = os.path.join(data_root, "rel.json")
        if not (os.path.isfile(train_path) and os.path.isfile(rel_path)):
            return None

        try:
            with open(train_path, "r") as f:
                train_data = json.load(f)
            with open(rel_path, "r") as f:
                rel_data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        img_to_labels = {}
        for ann in train_data.get("annotations", []):
            img_id = str(ann.get("image_id"))
            img_to_labels.setdefault(img_id, []).append(int(ann.get("category_id", 0)))

        counts = torch.full(
            (self.num_objects, self.num_objects, self.num_predicates),
            float(self.eps),
        )
        # ── bg_matrix: all object class co-occurrences (official PENet) ──
        # Official get_VG_statistics: bg_matrix counts all overlapping pairs,
        # bg_matrix += 1 (Laplace), then fg_matrix[:,:,0] = bg_matrix.
        # Without this, the bg prior is uniform and predictions collapse.
        bg_counts = torch.zeros(
            (self.num_objects, self.num_objects), dtype=torch.float64
        )
        for img_id in rel_data.get("train", {}).keys():
            labels = img_to_labels.get(str(img_id))
            if not labels:
                continue
            labels_arr = [lbl for lbl in labels
                          if 0 <= lbl < self.num_objects]
            n = len(labels_arr)
            for i in range(n):
                c1 = labels_arr[i]
                for j in range(n):
                    if i != j:
                        bg_counts[c1, labels_arr[j]] += 1
        counts[:, :, 0] = bg_counts.to(counts.dtype) + 1.0  # Laplace smoothing

        for img_id, rels in rel_data.get("train", {}).items():
            labels = img_to_labels.get(str(img_id))
            if not labels:
                continue
            for rel in rels:
                if len(rel) < 3:
                    continue
                subj_idx, obj_idx, pred_id = int(rel[0]), int(rel[1]), int(rel[2])
                if subj_idx >= len(labels) or obj_idx >= len(labels):
                    continue
                pred_idx = self._predicate_index(pred_id)
                subj_cls, obj_cls = labels[subj_idx], labels[obj_idx]
                if (
                    0 <= subj_cls < self.num_objects
                    and 0 <= obj_cls < self.num_objects
                    and 0 <= pred_idx < self.num_predicates
                ):
                    counts[subj_cls, obj_cls, pred_idx] += 1.0

        probs = counts / counts.sum(dim=-1, keepdim=True).clamp(min=self.eps)
        return torch.log(probs + self.eps)

    def _predicate_index(self, pred_id: int) -> int:
        """Map VG predicate ids (1..50) into this table's class layout."""
        if self.predicate_bg_index == "first":
            return pred_id
        return pred_id - 1

    def index_with_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Lookup bias for ``labels`` with shape [num_pairs, 2]."""
        labels = labels.clamp(min=0, max=self.num_objects - 1)
        return self.obj_baseline(labels[:, 0] * self.num_objects + labels[:, 1])

    def index_with_probability(self, pair_prob: torch.Tensor) -> torch.Tensor:
        """Lookup frequency bias from subject/object class probabilities.

        This mirrors the official TDE ``FrequencyBias.index_with_probability``
        API. ``pair_prob`` has shape ``[num_pairs, num_objects, 2]`` where the
        last dimension contains subject and object distributions.
        """
        if pair_prob.dim() != 3 or pair_prob.size(-1) != 2:
            raise ValueError(
                "pair_prob must have shape [num_pairs, num_objects, 2], "
                f"got {tuple(pair_prob.shape)}"
            )
        subj_dists = pair_prob[:, :, 0].contiguous()
        obj_dists = pair_prob[:, :, 1].contiguous()
        return self.forward(subj_dists, obj_dists)

    def forward(self, subj_dists: torch.Tensor, obj_dists: torch.Tensor) -> torch.Tensor:
        joint = subj_dists[:, :, None] * obj_dists[:, None, :]
        return joint.reshape(joint.size(0), -1) @ self.obj_baseline.weight


def _encode_box_info(boxes: torch.Tensor) -> torch.Tensor:
    """Encode normalized cxcywh boxes in SGB's 9D box-info layout."""
    if boxes.numel() == 0:
        return boxes.new_zeros(0, 9)

    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack((w, h, cx, cy, x1, y1, x2, y2, w * h), dim=-1)


def _sort_by_center_x(boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SGB-style right-to-left sort and inverse sort for one image."""
    if boxes.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=boxes.device)
        return empty, empty
    _, perm = torch.sort(boxes[:, 0], descending=True)
    _, inv_perm = torch.sort(perm)
    return perm, inv_perm


def _run_ordered_lstm(
    rnn: nn.LSTM,
    inputs: torch.Tensor,
    boxes: torch.Tensor,
    linear: Optional[nn.Linear] = None,
) -> torch.Tensor:
    """Run an SGB-compatible LSTM over one image sorted by box center."""
    perm, inv_perm = _sort_by_center_x(boxes)
    sorted_inputs = inputs[perm].unsqueeze(1)
    outputs, _ = rnn(sorted_inputs)
    outputs = outputs.squeeze(1)
    if linear is not None:
        outputs = linear(outputs)
    return outputs[inv_perm]


class SGBDecoderRNN(nn.Module):
    """Highway object decoder with SGB checkpoint-compatible parameter names."""

    def __init__(
        self,
        num_classes: int,
        embed_dim: int,
        inputs_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.inputs_dim = inputs_dim
        self.hidden_size = hidden_dim
        self.dropout = dropout

        self.obj_embed = nn.Embedding(num_classes + 1, embed_dim)
        self.input_linearity = nn.Linear(inputs_dim + embed_dim, 6 * hidden_dim)
        self.state_linearity = nn.Linear(hidden_dim, 5 * hidden_dim)
        self.out_obj = nn.Linear(hidden_dim, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.obj_embed.weight, std=0.01)
        nn.init.xavier_uniform_(self.input_linearity.weight)
        nn.init.constant_(self.input_linearity.bias, 0.0)
        nn.init.xavier_uniform_(self.state_linearity.weight)
        nn.init.constant_(self.state_linearity.bias, 0.0)
        nn.init.xavier_uniform_(self.out_obj.weight)
        nn.init.constant_(self.out_obj.bias, 0.0)

    def _step(
        self,
        timestep_input: torch.Tensor,
        previous_state: torch.Tensor,
        previous_memory: torch.Tensor,
        dropout_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected_input = self.input_linearity(timestep_input)
        projected_state = self.state_linearity(previous_state)

        input_gate = torch.sigmoid(
            projected_input[:, 0 * self.hidden_size : 1 * self.hidden_size]
            + projected_state[:, 0 * self.hidden_size : 1 * self.hidden_size]
        )
        forget_gate = torch.sigmoid(
            projected_input[:, 1 * self.hidden_size : 2 * self.hidden_size]
            + projected_state[:, 1 * self.hidden_size : 2 * self.hidden_size]
        )
        memory_init = torch.tanh(
            projected_input[:, 2 * self.hidden_size : 3 * self.hidden_size]
            + projected_state[:, 2 * self.hidden_size : 3 * self.hidden_size]
        )
        output_gate = torch.sigmoid(
            projected_input[:, 3 * self.hidden_size : 4 * self.hidden_size]
            + projected_state[:, 3 * self.hidden_size : 4 * self.hidden_size]
        )
        memory = input_gate * memory_init + forget_gate * previous_memory
        timestep_output = output_gate * torch.tanh(memory)

        highway_gate = torch.sigmoid(
            projected_input[:, 4 * self.hidden_size : 5 * self.hidden_size]
            + projected_state[:, 4 * self.hidden_size : 5 * self.hidden_size]
        )
        highway_input_projection = projected_input[
            :, 5 * self.hidden_size : 6 * self.hidden_size
        ]
        timestep_output = (
            highway_gate * timestep_output
            + (1.0 - highway_gate) * highway_input_projection
        )

        if dropout_mask is not None and self.training:
            timestep_output = timestep_output * dropout_mask
        return timestep_output, memory

    def forward(
        self,
        inputs: torch.Tensor,
        boxes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.numel() == 0:
            return (
                inputs.new_zeros(0, self.num_classes),
                torch.empty(0, dtype=torch.long, device=inputs.device),
            )

        perm, inv_perm = _sort_by_center_x(boxes)
        sequence = inputs[perm]
        sorted_labels = labels[perm] if labels is not None else None

        previous_memory = sequence.new_zeros(1, self.hidden_size)
        previous_state = sequence.new_zeros(1, self.hidden_size)
        previous_obj_embed = self.obj_embed.weight[0].view(1, self.embed_dim)
        dropout_mask = None
        if self.dropout > 0.0 and self.training:
            dropout_mask = (
                torch.rand_like(previous_memory).gt(self.dropout).float()
                / (1.0 - self.dropout)
            )

        out_dists = []
        out_preds = []
        for idx in range(sequence.size(0)):
            timestep_input = torch.cat((sequence[idx : idx + 1], previous_obj_embed), dim=1)
            previous_state, previous_memory = self._step(
                timestep_input,
                previous_state,
                previous_memory,
                dropout_mask,
            )
            pred_dist = self.out_obj(previous_state)
            out_dists.append(pred_dist)

            if self.training and sorted_labels is not None:
                next_label = sorted_labels[idx : idx + 1].clone()
                if int(next_label.item()) == 0:
                    next_label = pred_dist[:, 1:].argmax(dim=1) + 1
            else:
                next_label = pred_dist[:, 1:].argmax(dim=1) + 1
            out_preds.append(next_label)
            self._embed_label_for_decode = next_label
            previous_obj_embed = self.obj_embed(next_label.clamp(0, self.num_classes - 1) + 1)

        obj_dists = torch.cat(out_dists, dim=0)[inv_perm]
        obj_preds = torch.cat(out_preds, dim=0)[inv_perm]
        return obj_dists, obj_preds


class SGBLSTMContext(nn.Module):
    """SGB/Kaihua Motifs context layer with checkpoint-compatible names."""

    def __init__(
        self,
        num_classes: int,
        obj_dim: int,
        hidden_dim: int,
        embed_dim: int,
        obj_lstm_layers: int,
        edge_lstm_layers: int,
        dropout: float,
        effect_analysis: bool = True,
    ):
        super().__init__()
        if obj_lstm_layers <= 0 or edge_lstm_layers <= 0:
            raise ValueError("SGB Motifs requires object and edge LSTM layers > 0.")

        self.num_classes = num_classes
        self.obj_dim = obj_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout
        self.nl_obj = obj_lstm_layers
        self.nl_edge = edge_lstm_layers
        self.effect_analysis = effect_analysis
        self.average_ratio = 0.0005

        self.obj_embed1 = nn.Embedding(num_classes, embed_dim)
        self.obj_embed2 = nn.Embedding(num_classes, embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            _SafeBatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )

        obj_input_dim = obj_dim + embed_dim + 128
        self.obj_ctx_rnn = nn.LSTM(
            input_size=obj_input_dim,
            hidden_size=hidden_dim,
            num_layers=obj_lstm_layers,
            dropout=dropout if obj_lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.decoder_rnn = SGBDecoderRNN(
            num_classes=num_classes,
            embed_dim=embed_dim,
            inputs_dim=hidden_dim + obj_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.edge_ctx_rnn = nn.LSTM(
            input_size=embed_dim + obj_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=edge_lstm_layers,
            dropout=dropout if edge_lstm_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.lin_obj_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_edge_h = nn.Linear(hidden_dim * 2, hidden_dim)

        if effect_analysis:
            self.register_buffer("untreated_dcd_feat", torch.zeros(hidden_dim + obj_input_dim))
            self.register_buffer("untreated_obj_feat", torch.zeros(obj_input_dim))
            self.register_buffer("untreated_edg_feat", torch.zeros(embed_dim + obj_dim))

        self._init_weights()

    def moving_average(self, holder: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        """Official TDE moving average update for untreated context features."""
        if input.numel() == 0:
            return holder
        with torch.no_grad():
            holder.mul_(1.0 - self.average_ratio)
            holder.add_(self.average_ratio * input.mean(0).view_as(holder))
        return holder

    def _init_weights(self) -> None:
        for embed in (self.obj_embed1, self.obj_embed2):
            nn.init.normal_(embed.weight, std=0.01)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        obj_dists: Optional[torch.Tensor] = None,
        all_average: bool = False,
        ctx_average: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if visual_feats.numel() == 0:
            empty_logits = visual_feats.new_zeros(0, self.num_classes)
            return empty_logits, labels, visual_feats.new_zeros(0, self.hidden_dim)

        labels = labels.clamp(min=0, max=self.num_classes - 1).long()
        if return_obj_preds and obj_dists is not None:
            obj_embed = F.softmax(obj_dists, dim=1) @ self.obj_embed1.weight
        else:
            obj_embed = self.obj_embed1(labels)

        obj_pre_rep = torch.cat(
            (visual_feats, obj_embed, self.pos_embed(_encode_box_info(boxes))),
            dim=-1,
        )
        if all_average and self.effect_analysis and not self.training:
            obj_pre_rep = self.untreated_obj_feat.view(1, -1).expand(
                visual_feats.size(0), -1
            )

        obj_ctx = _run_ordered_lstm(
            self.obj_ctx_rnn,
            obj_pre_rep,
            boxes,
            linear=self.lin_obj_h,
        )

        if return_obj_preds:
            decoder_input = torch.cat((obj_pre_rep, obj_ctx), dim=-1)
            if ctx_average and self.effect_analysis and not self.training:
                decoder_input = self.untreated_dcd_feat.view(1, -1).expand(
                    visual_feats.size(0), -1
                )
            if self.training and self.effect_analysis:
                self.moving_average(self.untreated_dcd_feat, decoder_input)
            obj_logits, obj_preds = self.decoder_rnn(decoder_input, boxes, labels)
        else:
            obj_preds = labels
            obj_logits = _onehot(labels, self.num_classes).to(visual_feats.dtype) * 1000.0

        obj_embed2 = self.obj_embed2(obj_preds.clamp(min=0, max=self.num_classes - 1))
        obj_rel_rep = torch.cat((obj_embed2, visual_feats, obj_ctx), dim=-1)
        if (all_average or ctx_average) and self.effect_analysis and not self.training:
            avg_edge_input = self.untreated_edg_feat.view(1, -1).expand(
                visual_feats.size(0), -1
            )
            obj_rel_rep = torch.cat((avg_edge_input, obj_ctx), dim=-1)
        if self.training and self.effect_analysis:
            self.moving_average(self.untreated_obj_feat, obj_pre_rep)
            self.moving_average(
                self.untreated_edg_feat,
                torch.cat((obj_embed2, visual_feats), dim=-1),
            )
        edge_ctx = _run_ordered_lstm(
            self.edge_ctx_rnn,
            obj_rel_rep,
            boxes,
            linear=self.lin_edge_h,
        )
        return obj_logits, obj_preds, edge_ctx


class MotifsContext(nn.Module):
    """Legacy lightweight object and edge context module."""

    def __init__(
        self,
        num_classes: int,
        visual_dim: int,
        hidden_dim: int,
        embed_dim: int,
        obj_lstm_layers: int,
        edge_lstm_layers: int,
        dropout: float,
        order: str = "leftright",
        pos_embed_dim: int = 128,
        use_pos_batchnorm: bool = True,
        obj_feat_to_edge: bool = False,
    ):
        super().__init__()
        if order not in {"leftright", "size", "random", "confidence"}:
            raise ValueError(f"Unsupported Motifs order: {order}")

        self.num_classes = num_classes
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.order = order
        self.obj_lstm_layers = obj_lstm_layers
        self.edge_lstm_layers = edge_lstm_layers
        self.obj_feat_to_edge = obj_feat_to_edge

        self.obj_embed = nn.Embedding(num_classes, embed_dim)
        pos_layers: list[nn.Module] = []
        if use_pos_batchnorm:
            pos_layers.append(_SafeBatchNorm1d(4, momentum=0.001))
        pos_layers.extend(
            [
                nn.Linear(4, pos_embed_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        )
        self.pos_embed = nn.Sequential(*pos_layers)

        obj_input_dim = visual_dim + embed_dim + pos_embed_dim
        if obj_lstm_layers > 0:
            self.obj_ctx_rnn = nn.LSTM(
                obj_input_dim,
                hidden_dim,
                num_layers=obj_lstm_layers,
                batch_first=True,
                dropout=dropout if obj_lstm_layers > 1 else 0.0,
            )
            self.obj_classifier = nn.Linear(hidden_dim, num_classes)
        else:
            if edge_lstm_layers > 0:
                self.obj_ctx_trans = nn.Linear(obj_input_dim, hidden_dim)
            self.decoder_lin = nn.Linear(obj_input_dim, num_classes)

        if edge_lstm_layers > 0:
            self.obj_embed2 = nn.Embedding(num_classes, embed_dim)
            edge_input_dim = embed_dim + hidden_dim
            if obj_feat_to_edge:
                edge_input_dim += visual_dim
            self.edge_ctx_rnn = nn.LSTM(
                edge_input_dim,
                hidden_dim,
                num_layers=edge_lstm_layers,
                batch_first=True,
                dropout=dropout if edge_lstm_layers > 1 else 0.0,
            )

        self._init_weights(obj_input_dim)

    def _init_weights(self, obj_input_dim: int) -> None:
        nn.init.normal_(self.obj_embed.weight, std=0.01)
        if hasattr(self, "obj_embed2"):
            nn.init.normal_(self.obj_embed2.weight, std=0.01)
        if hasattr(self, "obj_ctx_trans"):
            nn.init.normal_(self.obj_ctx_trans.weight, 0.0, math.sqrt(1.0 / obj_input_dim))
            nn.init.constant_(self.obj_ctx_trans.bias, 0.0)
        if hasattr(self, "decoder_lin"):
            nn.init.normal_(self.decoder_lin.weight, 0.0, math.sqrt(1.0 / obj_input_dim))
            nn.init.constant_(self.decoder_lin.bias, 0.0)
        special = {
            getattr(self, "obj_ctx_trans", None),
            getattr(self, "decoder_lin", None),
        }
        for module in self.modules():
            if isinstance(module, nn.Linear) and module not in special:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _sort_indices(
        self,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=boxes.device)
        if self.order == "leftright":
            scores = boxes[:, 0]
        elif self.order == "size":
            scores = boxes[:, 2] * boxes[:, 3]
        elif self.order == "confidence":
            scores = confidence if confidence is not None else labels.float()
        else:
            scores = torch.rand(boxes.size(0), device=boxes.device)
        return torch.argsort(scores, descending=True)

    def _run_sorted_lstm(
        self,
        rnn: nn.LSTM,
        inputs: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sort_idx = self._sort_indices(boxes, labels, confidence)
        unsort_idx = torch.argsort(sort_idx)
        sorted_inputs = inputs[sort_idx].unsqueeze(0)
        sorted_outputs, _ = rnn(sorted_inputs)
        return sorted_outputs.squeeze(0)[unsort_idx]

    def _predict_objects(
        self,
        obj_logits: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool,
    ) -> torch.Tensor:
        if not return_obj_preds:
            return labels
        if self.training:
            return labels
        probs = F.softmax(obj_logits, dim=-1)
        return probs[:, 1:].argmax(dim=-1) + 1 if probs.size(1) > 1 else probs.argmax(dim=-1)

    def forward(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        obj_dists: Optional[torch.Tensor] = None,
    ):
        if obj_dists is None:
            obj_dists = _onehot(labels, self.num_classes)
        else:
            obj_dists = obj_dists.clamp(min=0)

        obj_embed = obj_dists @ self.obj_embed.weight
        obj_pre_rep = torch.cat(
            (visual_feats, obj_embed, self.pos_embed(boxes)),
            dim=-1,
        )

        obj_conf = (
            obj_dists[:, 1:].max(dim=1)[0]
            if obj_dists.size(1) > 1
            else obj_dists.max(dim=1)[0]
        )
        if self.obj_lstm_layers > 0:
            obj_ctx = self._run_sorted_lstm(
                self.obj_ctx_rnn,
                obj_pre_rep,
                boxes,
                labels,
                confidence=obj_conf,
            )
            obj_logits = self.obj_classifier(obj_ctx)
        else:
            obj_logits = self.decoder_lin(obj_pre_rep)
            obj_ctx = self.obj_ctx_trans(obj_pre_rep) if self.edge_lstm_layers > 0 else None

        obj_preds = self._predict_objects(obj_logits, labels, return_obj_preds)

        edge_ctx = None
        if self.edge_lstm_layers > 0:
            edge_embed = self.obj_embed2(obj_preds.clamp(min=0, max=self.num_classes - 1))
            edge_inputs = [edge_embed, obj_ctx]
            if self.obj_feat_to_edge:
                edge_inputs.append(visual_feats)
            edge_input = torch.cat(edge_inputs, dim=-1)
            obj_dists_for_edge = obj_logits.detach()
            if not return_obj_preds:
                obj_dists_for_edge = _onehot(labels, self.num_classes).to(obj_logits.dtype)
            edge_conf = F.softmax(obj_dists_for_edge, dim=-1).gather(
                1,
                obj_preds.clamp(min=0, max=self.num_classes - 1).view(-1, 1),
            ).squeeze(1)
            edge_ctx = self._run_sorted_lstm(
                self.edge_ctx_rnn,
                edge_input,
                boxes,
                obj_preds,
                confidence=edge_conf,
            )

        return obj_logits, obj_preds, edge_ctx


class MotifsModel(nn.Module):
    """SGB/Kaihua MotifPredictor relation head for PredCLS/SGCLS."""

    EXTERNAL_PREFIXES = (
        "module.roi_heads.relation.predictor.",
        "roi_heads.relation.predictor.",
    )

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        visual_dim: int = 2048,
        hidden_dim: int = 512,
        pooling_dim: int = 4096,
        motif_feat_dim: Optional[int] = None,
        embed_dim: int = 200,
        obj_lstm_layers: int = 1,
        edge_lstm_layers: int = 1,
        use_freq_bias: bool = True,
        freq_bias_eps: float = 1e-3,
        dropout: float = 0.2,
        order: str = "leftright",
        data_root: Optional[str] = None,
        use_tanh: bool = False,
        use_vision: bool = True,
        include_bg_predicate: bool = True,
        predicate_bg_index: str = "first",
        pos_embed_dim: int = 128,
        use_pos_batchnorm: bool = True,
        obj_feat_to_edge: bool = True,
        effect_analysis: bool = True,
    ):
        super().__init__()
        if order != "leftright":
            raise ValueError("SGB/Kaihua Motifs currently uses leftright object ordering.")
        if pos_embed_dim != 128 or not use_pos_batchnorm:
            raise ValueError("SGB/Kaihua Motifs checkpoint layout requires 128D box embeddings with BatchNorm.")
        if not obj_feat_to_edge:
            raise ValueError("SGB/Kaihua Motifs passes ROI features into edge context.")

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.obj_feat_dim = motif_feat_dim or pooling_dim
        self.pooling_dim = pooling_dim
        self.use_freq_bias = use_freq_bias
        self.use_tanh = use_tanh
        self.use_vision = use_vision
        self.predicate_bg_index = predicate_bg_index if include_bg_predicate else None

        if visual_dim == self.obj_feat_dim:
            self.input_visual_proj = nn.Identity()
        else:
            self.input_visual_proj = nn.Linear(visual_dim, self.obj_feat_dim)
            nn.init.xavier_uniform_(self.input_visual_proj.weight)
            nn.init.constant_(self.input_visual_proj.bias, 0.0)

        self.context_layer = SGBLSTMContext(
            num_classes=num_classes,
            obj_dim=self.obj_feat_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            obj_lstm_layers=obj_lstm_layers,
            edge_lstm_layers=edge_lstm_layers,
            dropout=dropout,
            effect_analysis=effect_analysis,
        )

        self.post_emb = nn.Linear(hidden_dim, hidden_dim * 2)
        self.post_cat = nn.Linear(hidden_dim * 2, pooling_dim)
        self.rel_compress = nn.Linear(pooling_dim, num_predicates, bias=True)
        nn.init.normal_(self.post_emb.weight, 0.0, 10.0 * math.sqrt(1.0 / hidden_dim))
        nn.init.constant_(self.post_emb.bias, 0.0)
        nn.init.xavier_uniform_(self.post_cat.weight)
        nn.init.constant_(self.post_cat.bias, 0.0)
        nn.init.xavier_uniform_(self.rel_compress.weight)
        nn.init.constant_(self.rel_compress.bias, 0.0)

        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = PairFrequencyBias(
                num_objects=num_classes,
                num_predicates=num_predicates,
                eps=freq_bias_eps,
                data_root=data_root,
                predicate_bg_index=self.predicate_bg_index,
            )

    def _generate_pairs(self, num_objects: int, device: torch.device) -> torch.Tensor:
        return generate_object_pairs(num_objects, device)

    def _fallback_union_features(
        self,
        motif_visual_feats: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            motif_visual_feats[pairs[:, 0]]
            + motif_visual_feats[pairs[:, 1]]
        )

    def _project_union_features(self, union_feats: torch.Tensor) -> torch.Tensor:
        if union_feats.size(-1) == self.pooling_dim:
            return union_feats
        if union_feats.size(-1) == self.visual_dim and self.visual_dim != self.obj_feat_dim:
            return self.input_visual_proj(union_feats)
        if union_feats.size(-1) == self.obj_feat_dim and self.obj_feat_dim == self.pooling_dim:
            return union_feats
        raise ValueError(
            f"Union features dim {union_feats.size(-1)} cannot be used with "
            f"pooling_dim={self.pooling_dim}, visual_dim={self.visual_dim}, "
            f"obj_feat_dim={self.obj_feat_dim}."
        )

    def extract_object_context(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-object edge-context features for downstream pair generators.

        CVC/VCTree-style methods that need object-level context features
        (instead of the full relation head) can call this to get
        ``[N, hidden_dim]`` features usable with :class:`PairFeatureGenerator`.
        """
        motif_visual_feats = self.input_visual_proj(visual_feats)
        _, _, edge_ctx = self.context_layer(
            motif_visual_feats, boxes, labels, return_obj_preds=False,
        )
        return edge_ctx

    def remap_external_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Map full SGB/Kaihua checkpoints to this relation-head module."""
        model_state = self.state_dict()
        remapped: dict[str, torch.Tensor] = {}
        for name, tensor in state_dict.items():
            clean = name[7:] if name.startswith("module.") else name
            candidates = [clean]
            for prefix in self.EXTERNAL_PREFIXES:
                if name.startswith(prefix):
                    candidates.append(name[len(prefix):])
                if clean.startswith(prefix.removeprefix("module.")):
                    candidates.append(clean[len(prefix.removeprefix("module.")):])

            for candidate in candidates:
                if candidate in model_state:
                    remapped[candidate] = tensor
                    break
        if remapped:
            return remapped
        return {
            (name[7:] if name.startswith("module.") else name): tensor
            for name, tensor in state_dict.items()
        }

    def forward(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        obj_dists: Optional[torch.Tensor] = None,
        union_feats: Optional[torch.Tensor] = None,
    ):
        num_objects = visual_feats.size(0)
        device = visual_feats.device
        pairs = self._generate_pairs(num_objects, device)
        motif_visual_feats = self.input_visual_proj(visual_feats)

        if num_objects == 0 or pairs.numel() == 0:
            empty_logits = motif_visual_feats.new_zeros(0, self.num_predicates)
            return {
                "rel_logits": empty_logits,
                "obj_logits": None,
                "obj_labels": labels,
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "predicate_bg_index": self.predicate_bg_index,
                "relation_softmax_scope": "all",
            }

        obj_logits, obj_preds, edge_ctx = self.context_layer(
            motif_visual_feats,
            boxes,
            labels,
            return_obj_preds=return_obj_preds,
            obj_dists=obj_dists,
        )

        edge_rep = self.post_emb(edge_ctx).view(num_objects, 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous()
        tail_rep = edge_rep[:, 1].contiguous()
        prod_rep = torch.cat(
            (head_rep[pairs[:, 0]], tail_rep[pairs[:, 1]]),
            dim=-1,
        )
        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            edge_visual_feats = (
                self._project_union_features(union_feats)
                if union_feats is not None
                else self._fallback_union_features(motif_visual_feats, pairs)
            )
            prod_rep = prod_rep * edge_visual_feats

        if self.use_tanh:
            prod_rep = torch.tanh(prod_rep)

        rel_logits = self.rel_compress(prod_rep)

        if self.freq_bias is not None:
            pair_labels = torch.stack((obj_preds[pairs[:, 0]], obj_preds[pairs[:, 1]]), dim=-1)
            rel_logits = rel_logits + self.freq_bias.index_with_labels(pair_labels.long())

        return {
            "rel_logits": rel_logits,
            "obj_logits": obj_logits,
            "obj_labels": obj_preds,
            "pair_indices": pairs,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
            "predicate_bg_index": self.predicate_bg_index,
            "relation_softmax_scope": "all",
        }


class TDEModel(MotifsModel):
    """Official-style CausalAnalysisPredictor for Motifs-SUM TDE.

    The model keeps OpenSGG's per-image tensor interface while mirroring the
    official TDE predictor structure: visual, context, and frequency branches,
    untreated moving-average buffers, and effect modes ``none/TDE/NIE/TE``.
    """

    VALID_EFFECTS = {"none", "TDE", "NIE", "TE"}
    VALID_FUSIONS = {"sum", "gate"}

    def __init__(
        self,
        *args,
        effect_type: str = "TDE",
        fusion_type: str = "sum",
        spatial_for_vision: bool = True,
        separate_spatial: bool = False,
        average_ratio: float = 0.0005,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        effect_type = effect_type if effect_type == "none" else effect_type.upper()
        fusion_type = fusion_type.lower()
        if effect_type not in self.VALID_EFFECTS:
            raise ValueError(f"Unsupported TDE effect_type: {effect_type}")
        if fusion_type not in self.VALID_FUSIONS:
            raise ValueError(f"Unsupported TDE fusion_type: {fusion_type}")
        if separate_spatial:
            raise ValueError("TDE separate_spatial is not supported by OpenSGG union features yet.")

        self.effect_type = effect_type
        self.fusion_type = fusion_type
        self.spatial_for_vision = spatial_for_vision
        self.separate_spatial = separate_spatial
        self.average_ratio = average_ratio
        if self.effect_type != "none" and not self.context_layer.effect_analysis:
            raise ValueError(
                "TDE effect analysis requires motifs_effect_analysis=True; "
                f"got effect_type={self.effect_type!r} with effect_analysis disabled."
            )

        self.post_cat = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.pooling_dim),
            nn.ReLU(inplace=True),
        )
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_predicates)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_predicates)
        if self.fusion_type == "gate":
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_predicates)

        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(
                nn.Linear(32, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, self.pooling_dim),
                nn.ReLU(inplace=True),
            )

        self._init_causal_weights()
        self.register_buffer("untreated_spt", torch.zeros(32))

    def _init_causal_weights(self) -> None:
        nn.init.xavier_uniform_(self.post_cat[0].weight)
        nn.init.constant_(self.post_cat[0].bias, 0.0)
        nn.init.xavier_uniform_(self.ctx_compress.weight)
        nn.init.constant_(self.ctx_compress.bias, 0.0)
        nn.init.xavier_uniform_(self.vis_compress.weight)
        nn.init.constant_(self.vis_compress.bias, 0.0)
        if hasattr(self, "ctx_gate_fc"):
            nn.init.xavier_uniform_(self.ctx_gate_fc.weight)
            nn.init.constant_(self.ctx_gate_fc.bias, 0.0)
        if hasattr(self, "spt_emb"):
            nn.init.xavier_uniform_(self.spt_emb[0].weight)
            nn.init.constant_(self.spt_emb[0].bias, 0.0)
            nn.init.xavier_uniform_(self.spt_emb[2].weight)
            nn.init.constant_(self.spt_emb[2].bias, 0.0)

    def moving_average(self, holder: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        if input.numel() == 0:
            return holder
        with torch.no_grad():
            holder.mul_(1.0 - self.average_ratio)
            holder.add_(self.average_ratio * input.mean(0).view_as(holder))
        return holder

    def _pair_feature_generate(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        obj_dists: Optional[torch.Tensor] = None,
        union_feats: Optional[torch.Tensor] = None,
        ctx_average: bool = False,
    ) -> dict[str, torch.Tensor]:
        num_objects = visual_feats.size(0)
        pairs = self._generate_pairs(num_objects, visual_feats.device)
        motif_visual_feats = self.input_visual_proj(visual_feats)
        obj_logits, obj_preds, edge_ctx = self.context_layer(
            motif_visual_feats,
            boxes,
            labels,
            return_obj_preds=return_obj_preds,
            obj_dists=obj_dists,
            ctx_average=ctx_average,
        )
        obj_prob = F.softmax(obj_logits, dim=-1)

        edge_rep = self.post_emb(edge_ctx).view(num_objects, 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous()
        tail_rep = edge_rep[:, 1].contiguous()
        ctx_rep = torch.cat(
            (head_rep[pairs[:, 0]], tail_rep[pairs[:, 1]]),
            dim=-1,
        )
        post_ctx_rep = self.post_cat(ctx_rep)

        pair_pred = torch.stack((obj_preds[pairs[:, 0]], obj_preds[pairs[:, 1]]), dim=-1)
        pair_obj_probs = torch.stack(
            (obj_prob[pairs[:, 0]], obj_prob[pairs[:, 1]]),
            dim=-1,
        )
        pair_bbox = _pair_box_info(boxes, pairs)
        if self.use_vision:
            visual_rep = (
                self._project_union_features(union_feats)
                if union_feats is not None
                else self._fallback_union_features(motif_visual_feats, pairs)
            )
        else:
            visual_rep = post_ctx_rep.new_zeros(post_ctx_rep.shape)

        return {
            "pairs": pairs,
            "obj_logits": obj_logits,
            "obj_preds": obj_preds,
            "post_ctx_rep": post_ctx_rep,
            "pair_pred": pair_pred,
            "pair_obj_probs": pair_obj_probs,
            "pair_bbox": pair_bbox,
            "visual_rep": visual_rep,
        }

    def _branch_logits(
        self,
        vis_rep: torch.Tensor,
        ctx_rep: torch.Tensor,
        frq_rep: torch.Tensor,
        use_label_dist: bool = True,
    ) -> dict[str, torch.Tensor]:
        if self.freq_bias is None:
            frq_dists = vis_rep.new_zeros(vis_rep.size(0), self.num_predicates)
        elif use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())
        return {
            "vis": self.vis_compress(vis_rep),
            "ctx": self.ctx_compress(ctx_rep),
            "frq": frq_dists,
        }

    def calculate_logits(
        self,
        vis_rep: torch.Tensor,
        ctx_rep: torch.Tensor,
        frq_rep: torch.Tensor,
        use_label_dist: bool = True,
    ) -> torch.Tensor:
        branches = self._branch_logits(vis_rep, ctx_rep, frq_rep, use_label_dist)
        if self.fusion_type == "sum":
            return branches["vis"] + branches["ctx"] + branches["frq"]
        if self.fusion_type == "gate":
            ctx_gate = self.ctx_gate_fc(ctx_rep)
            return branches["ctx"] * torch.sigmoid(branches["vis"] + branches["frq"] + ctx_gate)
        raise ValueError(f"Unsupported TDE fusion_type: {self.fusion_type}")

    def _target_predicate_index(self, pred_id: int, logits_dim: int) -> int:
        if self.predicate_bg_index == "first" and logits_dim == self.num_predicates:
            return pred_id
        return pred_id - 1

    def _gt_for_pairs(
        self,
        pairs: torch.Tensor,
        rel_annotations: Optional[torch.Tensor],
        logits_dim: int,
    ) -> torch.Tensor:
        gt = torch.full((pairs.size(0),), -1, dtype=torch.long, device=pairs.device)
        if rel_annotations is None or rel_annotations.numel() == 0:
            return gt
        pair_to_index = {
            (int(pair[0]), int(pair[1])): idx
            for idx, pair in enumerate(pairs.detach().cpu().tolist())
        }
        for ann in rel_annotations.detach().cpu().tolist():
            if len(ann) < 3:
                continue
            pair_idx = pair_to_index.get((int(ann[0]), int(ann[1])))
            pred_idx = self._target_predicate_index(int(ann[2]), logits_dim)
            if pair_idx is not None and 0 <= pred_idx < logits_dim:
                gt[pair_idx] = pred_idx
        return gt

    def _auxiliary_losses(
        self,
        branches: dict[str, torch.Tensor],
        pairs: torch.Tensor,
        rel_annotations: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.training:
            return {}
        gt = self._gt_for_pairs(pairs, rel_annotations, branches["ctx"].size(-1))
        valid = gt >= 0
        if not valid.any():
            return {}

        losses = {
            "auxiliary_ctx": F.cross_entropy(branches["ctx"][valid], gt[valid]),
        }
        if self.fusion_type != "gate":
            losses["auxiliary_vis"] = F.cross_entropy(branches["vis"][valid], gt[valid])
            losses["auxiliary_frq"] = F.cross_entropy(branches["frq"][valid], gt[valid])
        return losses

    def _empty_output(
        self,
        motif_visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        pairs: torch.Tensor,
    ) -> dict:
        return {
            "rel_logits": motif_visual_feats.new_zeros(0, self.num_predicates),
            "obj_logits": None,
            "obj_labels": labels,
            "pair_indices": pairs,
            "sub_boxes": boxes.new_zeros(0, 4),
            "obj_boxes": boxes.new_zeros(0, 4),
            "predicate_bg_index": self.predicate_bg_index,
            "relation_softmax_scope": "all",
            "add_losses": {},
        }

    def forward(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        apply_tde: bool = True,
        obj_dists: Optional[torch.Tensor] = None,
        union_feats: Optional[torch.Tensor] = None,
        rel_annotations: Optional[torch.Tensor] = None,
    ) -> dict:
        num_objects = visual_feats.size(0)
        pairs = self._generate_pairs(num_objects, visual_feats.device)
        motif_visual_feats = self.input_visual_proj(visual_feats)
        if num_objects == 0 or pairs.numel() == 0:
            return self._empty_output(motif_visual_feats, boxes, labels, pairs)

        features = self._pair_feature_generate(
            visual_feats,
            boxes,
            labels,
            return_obj_preds=return_obj_preds,
            obj_dists=obj_dists,
            union_feats=union_feats,
        )
        post_ctx_rep = features["post_ctx_rep"]
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(features["pair_bbox"])

        factual_branches = self._branch_logits(
            features["visual_rep"],
            post_ctx_rep,
            features["pair_pred"],
            use_label_dist=False,
        )
        rel_logits = self.calculate_logits(
            features["visual_rep"],
            post_ctx_rep,
            features["pair_pred"],
            use_label_dist=False,
        )
        add_losses = self._auxiliary_losses(
            factual_branches,
            features["pairs"],
            rel_annotations,
        )

        if self.training and self.context_layer.effect_analysis:
            if self.spatial_for_vision:
                self.moving_average(self.untreated_spt, features["pair_bbox"])
        elif (
            apply_tde
            and self.effect_type != "none"
            and self.context_layer.effect_analysis
        ):
            with torch.no_grad():
                avg_features = self._pair_feature_generate(
                    visual_feats,
                    boxes,
                    labels,
                    return_obj_preds=return_obj_preds,
                    obj_dists=obj_dists,
                    union_feats=union_feats,
                    ctx_average=True,
                )
                avg_ctx_rep = avg_features["post_ctx_rep"]
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.view(1, -1))
                    avg_ctx_rep = avg_ctx_rep * avg_spt_rep
                avg_frq_rep = avg_features["pair_obj_probs"]

            if self.effect_type == "TDE":
                rel_logits = self.calculate_logits(
                    features["visual_rep"], post_ctx_rep, features["pair_obj_probs"]
                ) - self.calculate_logits(
                    features["visual_rep"], avg_ctx_rep, features["pair_obj_probs"]
                )
            elif self.effect_type == "NIE":
                rel_logits = self.calculate_logits(
                    features["visual_rep"], avg_ctx_rep, features["pair_obj_probs"]
                ) - self.calculate_logits(
                    features["visual_rep"], avg_ctx_rep, avg_frq_rep
                )
            elif self.effect_type == "TE":
                rel_logits = self.calculate_logits(
                    features["visual_rep"], post_ctx_rep, features["pair_obj_probs"]
                ) - self.calculate_logits(
                    features["visual_rep"], avg_ctx_rep, avg_frq_rep
                )
            elif self.effect_type != "none":
                raise ValueError(f"Unsupported TDE effect_type: {self.effect_type}")

        return {
            "rel_logits": rel_logits,
            "obj_logits": features["obj_logits"],
            "obj_labels": features["obj_preds"],
            "pair_indices": features["pairs"],
            "sub_boxes": boxes[features["pairs"][:, 0]],
            "obj_boxes": boxes[features["pairs"][:, 1]],
            "predicate_bg_index": self.predicate_bg_index,
            "relation_softmax_scope": "all",
            "add_losses": add_losses,
        }


def build_motifs(args) -> MotifsModel:
    rel_nums = getattr(args, "rel_nums", 51)
    include_bg = getattr(args, "motifs_include_bg_predicate", rel_nums > 50)
    default_num_predicates = rel_nums if include_bg else _pure_predicate_count(rel_nums)
    return MotifsModel(
        num_classes=getattr(args, "entity_nums", 151),
        num_predicates=getattr(args, "motifs_num_predicates", default_num_predicates),
        visual_dim=getattr(args, "visual_dim", 2048),
        hidden_dim=getattr(args, "hidden_dim", 512),
        pooling_dim=getattr(args, "pooling_dim", 4096),
        motif_feat_dim=getattr(args, "motifs_obj_feat_dim", getattr(args, "pooling_dim", 4096)),
        embed_dim=getattr(args, "embed_dim", 200),
        obj_lstm_layers=getattr(args, "obj_lstm_layers", 1),
        edge_lstm_layers=getattr(args, "edge_lstm_layers", 1),
        use_freq_bias=getattr(args, "use_freq_bias", True),
        freq_bias_eps=getattr(args, "freq_bias_eps", 1e-3),
        dropout=getattr(args, "dropout", 0.2),
        order=getattr(args, "motifs_order", "leftright"),
        data_root=getattr(args, "data_root", None),
        use_tanh=getattr(args, "use_tanh", False),
        use_vision=getattr(args, "use_vision", True),
        include_bg_predicate=include_bg,
        predicate_bg_index=getattr(args, "motifs_predicate_bg_index", "first"),
        pos_embed_dim=getattr(args, "motifs_pos_embed_dim", 128),
        use_pos_batchnorm=getattr(args, "motifs_pos_batchnorm", True),
        obj_feat_to_edge=getattr(args, "motifs_obj_feat_to_edge", True),
        effect_analysis=getattr(args, "motifs_effect_analysis", True),
    )


def build_tde(args) -> TDEModel:
    rel_nums = getattr(args, "rel_nums", 51)
    include_bg = getattr(args, "motifs_include_bg_predicate", rel_nums > 50)
    default_num_predicates = rel_nums if include_bg else _pure_predicate_count(rel_nums)
    effect_type = getattr(args, "tde_effect_type", "TDE")
    fusion_type = getattr(args, "tde_fusion_type", getattr(args, "tde_fusion", "sum"))
    if fusion_type in {"subtract", "softmax_subtract"}:
        # Backward compatibility for the old local wrapper config.
        effect_type = "TDE"
        fusion_type = "sum"
    return TDEModel(
        num_classes=getattr(args, "entity_nums", 151),
        num_predicates=getattr(args, "motifs_num_predicates", default_num_predicates),
        visual_dim=getattr(args, "visual_dim", 2048),
        hidden_dim=getattr(args, "hidden_dim", 512),
        pooling_dim=getattr(args, "pooling_dim", 4096),
        motif_feat_dim=getattr(args, "motifs_obj_feat_dim", getattr(args, "pooling_dim", 4096)),
        embed_dim=getattr(args, "embed_dim", 200),
        obj_lstm_layers=getattr(args, "obj_lstm_layers", 1),
        edge_lstm_layers=getattr(args, "edge_lstm_layers", 1),
        use_freq_bias=getattr(args, "use_freq_bias", True),
        freq_bias_eps=getattr(args, "freq_bias_eps", 1e-3),
        dropout=getattr(args, "dropout", 0.2),
        order=getattr(args, "motifs_order", "leftright"),
        data_root=getattr(args, "data_root", None),
        use_tanh=getattr(args, "use_tanh", False),
        use_vision=getattr(args, "use_vision", True),
        effect_type=effect_type,
        fusion_type=fusion_type,
        spatial_for_vision=getattr(args, "tde_spatial_for_vision", True),
        separate_spatial=getattr(args, "tde_separate_spatial", False),
        average_ratio=getattr(args, "tde_average_ratio", 0.0005),
        include_bg_predicate=include_bg,
        predicate_bg_index=getattr(args, "motifs_predicate_bg_index", "first"),
        pos_embed_dim=getattr(args, "motifs_pos_embed_dim", 128),
        use_pos_batchnorm=getattr(args, "motifs_pos_batchnorm", True),
        obj_feat_to_edge=getattr(args, "motifs_obj_feat_to_edge", True),
        effect_analysis=getattr(args, "motifs_effect_analysis", True),
    )
