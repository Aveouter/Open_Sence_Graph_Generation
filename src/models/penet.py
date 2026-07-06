"""
PE-NET: Prototype-based Embedding Network for Scene Graph Generation.

Reference: "Prototype-based Embedding Network for Scene Graph Generation"
(Zheng et al., CVPR 2023)

This implementation is strictly aligned with the official VL-Group/PENET:
  https://github.com/VL-Group/PENET

Key dimensions (from official config ``e2e_relation_X_101_32_8_FPN_1x.yaml``):
  mlp_dim          = 2048  (hardcoded in official PrototypeEmbeddingNetwork)
  context_hidden_dim = 512   (CONTEXT_HIDDEN_DIM — used for refinement head only)
  embed_dim        = 300   (GloVe 300d)
  pooling_dim      = 4096  (CONTEXT_POOLING_DIM — union feature input dim)
  obj_dim          = 2048  (backbone ROI feature output)
  dropout          = 0.2

Official architecture (PrototypeEmbeddingNetwork.forward):
  Features come from RelationHead.forward which:
    1. Extracts roi_features via box_feature_extractor (per-object, 2048-dim)
    2. Extracts union_features via union_feature_extractor (per-pair, 4096-dim)
    3. Samples rel_pair_idxs and rel_labels (512 per image, 25% pos)
    4. Calls predictor(proposals, rel_pair_idxs, rel_labels, ...)
  The predictor then:
    a. Refines object labels (refine_obj_labels)
    b. Splits roi_features → sub_rep / obj_rep (post_emb, view)
    c. Loops per-image: gated fusion + fusion_func
    d. Union feature gating: r = F(s,o) - gp · h(xu)
    e. Projection head + L2 norm + cosine similarity
    f. Prototype regularization losses (training only)

Integration with OpenSGG:
  - Motifs_Method.forward() calls the model per-image
  - Feature maps are passed through for union feature computation
  - rel_annotations are matched to pairs for prototype losses
"""

from __future__ import annotations

import os
import zipfile
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .motifs import FrequencyBias, generate_object_pairs


# =========================================================================
# MLP — exact match of official class
# =========================================================================


class MLP(nn.Module):
    """Matches official ``MLP`` in roi_relation_predictors.py exactly."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# =========================================================================
# Fusion function — exact match
# =========================================================================


def fusion_func(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """F(s, o) = ReLU(s + o) - (s - o)^2  (official)."""
    return F.relu(x + y) - (x - y) ** 2


# =========================================================================
# make_fc — matches official maskrcnn_benchmark make_fc
# =========================================================================


def _make_fc(dim_in: int, hidden_dim: int) -> nn.Linear:
    """Create Linear with kaiming_uniform_ init matching official ``make_fc``."""
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc


# =========================================================================
# nms_overlaps — exact match of official utils_relation.nms_overlaps
# =========================================================================


def _nms_overlaps(boxes: torch.Tensor) -> torch.Tensor:
    """Get per-class IoU overlaps matching official ``nms_overlaps`` exactly.

    Args:
        boxes: [N, nc, 4] in (x1, y1, x2, y2) format.
    Returns:
        [N, N, nc] IoU matrix.
    """
    N, nc, _ = boxes.shape
    max_xy = torch.min(
        boxes[:, None, :, 2:].expand(N, N, nc, 2),
        boxes[None, :, :, 2:].expand(N, N, nc, 2),
    )
    min_xy = torch.max(
        boxes[:, None, :, :2].expand(N, N, nc, 2),
        boxes[None, :, :, :2].expand(N, N, nc, 2),
    )
    inter = torch.clamp(max_xy - min_xy + 1.0, min=0)
    inters = inter[:, :, :, 0] * inter[:, :, :, 1]
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
        boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0
    )
    areas = areas_flat.view(N, nc)
    union = -inters + areas[None] + areas[:, None]
    return inters / union


# =========================================================================
# GloVe loading — exact match of official obj_edge_vectors / rel_vectors
# =========================================================================


def _load_word_vectors(root: str, wv_type: str, dim: int | str):
    """Load word vectors matching official ``load_word_vectors``.

    Tries .pt (cached PyTorch), then .txt, then .zip download.
    """
    if isinstance(dim, int):
        dim = str(dim) + "d"
    fname = os.path.join(root, wv_type + "." + dim)

    # 1) Try cached .pt file (official does this first)
    if os.path.isfile(fname + ".pt"):
        try:
            return torch.load(fname + ".pt", map_location=torch.device("cpu"))
        except Exception:
            pass

    # 2) Try .txt file
    if os.path.isfile(fname + ".txt"):
        with open(fname + ".txt", "rb") as f:
            lines = [line for line in f]
    elif os.path.isfile(fname + ".zip"):
        with zipfile.ZipFile(fname + ".zip") as zf:
            # Find the .txt inside the zip
            txt_names = [n for n in zf.namelist() if n.endswith(".txt")]
            if txt_names:
                with zf.open(txt_names[0]) as f:
                    lines = [line for line in f]
            else:
                return None
    else:
        return None

    # Parse text format
    wv_dict: dict = {}
    # Infer dim from first line if needed
    int_dim = int(dim.replace("d", ""))
    for line in lines:
        parts = line.rstrip().split(b" ")
        if len(parts) < 2:
            continue
        word = parts[0]
        try:
            word = word.decode("utf-8")
        except (UnicodeDecodeError, AttributeError):
            pass
        vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float32)
        if vec.numel() != int_dim:
            continue
        wv_dict[word] = vec

    if not wv_dict:
        return None

    wv_arr = torch.stack(list(wv_dict.values()))
    return wv_dict, wv_arr, wv_arr.size(0)


def _obj_edge_vectors(
    names: list[str], wv_dir: str, wv_dim: int, wv_type: str = "glove.6B"
) -> torch.Tensor:
    """Load word vectors matching official ``obj_edge_vectors`` exactly.

    Key behaviours matching official:
      - Tries exact token match first
      - Falls back to LONGEST word in multi-word token (not averaging!)
      - Random init for any token that can't be found
    """
    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    result = _load_word_vectors(wv_dir, wv_type, wv_dim)
    if result is None:
        return vectors
    wv_dict, wv_arr, _ = result

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Official fallback: LONGEST word (by character count)
            lw_token = sorted(token.split(" "), key=lambda x: len(x), reverse=True)[0]
            wv_index = wv_dict.get(lw_token)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]

    return vectors


def _rel_vectors(
    names: list[str], wv_dir: str, wv_dim: int, wv_type: str = "glove.6B"
) -> torch.Tensor:
    """Same as _obj_edge_vectors but for predicates (index 0 = bg → random init)."""
    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    result = _load_word_vectors(wv_dir, wv_type, wv_dim)
    if result is None:
        return vectors
    wv_dict, wv_arr, _ = result

    for i, token in enumerate(names):
        if i == 0:
            continue  # keep random init for __no_relation__
        wv_index = wv_dict.get(token)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            lw_token = sorted(token.split(" "), key=lambda x: len(x), reverse=True)[0]
            wv_index = wv_dict.get(lw_token)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]

    return vectors


# =========================================================================
# Default VG class names
# =========================================================================

_VG_PREDICATE_NAMES = [
    "__no_relation__",
    "above",
    "across",
    "against",
    "along",
    "and",
    "at",
    "attached to",
    "behind",
    "belonging to",
    "between",
    "carrying",
    "covered in",
    "covering",
    "eating",
    "flying in",
    "for",
    "from",
    "growing on",
    "hanging from",
    "has",
    "holding",
    "in",
    "in front of",
    "laying on",
    "looking at",
    "lying on",
    "made of",
    "mounted on",
    "near",
    "of",
    "on",
    "on back of",
    "over",
    "painted on",
    "parked on",
    "part of",
    "playing",
    "riding",
    "says",
    "sitting on",
    "skiing on",
    "standing on",
    "surfing on",
    "to",
    "under",
    "using",
    "walking in",
    "walking on",
    "watching",
    "wearing",
    "wears",
    "with",
]

_VG_OBJECT_NAMES = [
    "__background__",
    "airplane",
    "animal",
    "arm",
    "bag",
    "banana",
    "basket",
    "beach",
    "bear",
    "bed",
    "bench",
    "bike",
    "bird",
    "board",
    "boat",
    "book",
    "boot",
    "bottle",
    "bowl",
    "box",
    "boy",
    "branch",
    "building",
    "bus",
    "cabinet",
    "cap",
    "car",
    "cat",
    "chair",
    "child",
    "clock",
    "coat",
    "counter",
    "cow",
    "cup",
    "curtain",
    "desk",
    "dog",
    "door",
    "drawer",
    "ear",
    "elephant",
    "engine",
    "eye",
    "face",
    "fence",
    "finger",
    "flag",
    "flower",
    "food",
    "fork",
    "fruit",
    "glasses",
    "guy",
    "hair",
    "hand",
    "handle",
    "hat",
    "head",
    "helmet",
    "hill",
    "horse",
    "house",
    "jacket",
    "jeans",
    "kid",
    "kite",
    "lady",
    "lamp",
    "laptop",
    "leaf",
    "leg",
    "letter",
    "light",
    "logo",
    "man",
    "men",
    "mirror",
    "motorcycle",
    "mountain",
    "mouth",
    "neck",
    "nose",
    "number",
    "pant",
    "paper",
    "paw",
    "people",
    "person",
    "phone",
    "pillow",
    "plane",
    "plant",
    "plate",
    "player",
    "pole",
    "post",
    "pot",
    "racket",
    "railing",
    "rock",
    "roof",
    "room",
    "screen",
    "seat",
    "shelf",
    "shirt",
    "shoe",
    "short",
    "sidewalk",
    "sign",
    "sink",
    "skateboard",
    "ski",
    "skier",
    "sneaker",
    "snow",
    "sock",
    "stand",
    "street",
    "surfboard",
    "table",
    "tail",
    "tie",
    "tile",
    "tire",
    "toilet",
    "towel",
    "tower",
    "track",
    "train",
    "tree",
    "truck",
    "trunk",
    "umbrella",
    "vase",
    "vehicle",
    "wave",
    "wheel",
    "window",
    "windshield",
    "wing",
    "wire",
    "woman",
    "zebra",
]


# =========================================================================
# encode_box_info — exact match of official (normalized-by-image-size)
# =========================================================================


def _encode_box_info(boxes: torch.Tensor) -> torch.Tensor:
    """Encode box info matching official ``encode_box_info`` EXACTLY.

    Official signature takes ``proposals`` (list of BoxList) and encodes:
      [w/wid, h/hei, x/wid, y/hei, x1/wid, y1/hei, x2/wid, y2/hei, w*h/(wid*hei)]

    OpenSGG boxes are (cx, cy, w, h) normalized to [0, 1], so:
      w/wid = w, h/hei = h, x/wid = cx, y/hei = cy, etc.
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    area = w * h
    # Official order: [w, h, x, y, x1, y1, x2, y2, area]
    return torch.stack([w, h, cx, cy, x1, y1, x2, y2, area], dim=-1)


# =========================================================================
# PE-NET Model
# =========================================================================


class PENetContext(nn.Module):
    """Prototype-based Embedding Network — strict match with official PENet.

    Two separate hidden dimensions:
      - ``mlp_dim = 2048`` — used for gating, projection, fusion
      - ``context_hidden_dim = 512`` — used ONLY for refinement head
        (lin_obj_cyx, out_obj), matching CONTEXT_HIDDEN_DIM in the official config.
    """

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        visual_dim: int = 4096,  # official obj_dim = MLP_HEAD_DIM = 4096
        hidden_dim: int = 2048,  # mlp_dim in official
        context_hidden_dim: int = 512,  # CONTEXT_HIDDEN_DIM in official
        embed_dim: int = 300,
        pooling_dim: int = 4096,  # CONTEXT_POOLING_DIM
        glove_dir: Optional[str] = None,
        obj_class_names: Optional[list[str]] = None,
        pred_class_names: Optional[list[str]] = None,
        use_freq_bias: bool = False,
        freq_bias_eps: float = 1e-12,
        dropout: float = 0.2,
        nms_thresh: float = 0.5,
        roi_output_size: int = 7,
        train_pairs_per_image: int = 512,
        train_positive_fraction: float = 0.25,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.mlp_dim = hidden_dim  # 2048
        self.context_hidden_dim = context_hidden_dim  # 512
        self.embed_dim = embed_dim
        self.pooling_dim = pooling_dim
        self.use_freq_bias = use_freq_bias
        self.nms_thresh = nms_thresh
        self.roi_output_size = roi_output_size
        self.train_pairs_per_image = train_pairs_per_image
        self.train_positive_fraction = train_positive_fraction

        # ===== 1. post_emb =====
        self.post_emb = nn.Linear(visual_dim, self.mlp_dim * 2)

        # ===== 2. GloVe embeddings =====
        self.obj_embed = nn.Embedding(num_classes, embed_dim)
        self.rel_embed = nn.Embedding(num_predicates, embed_dim)

        if glove_dir is not None and os.path.isdir(glove_dir):
            obj_names = obj_class_names or _VG_OBJECT_NAMES[:num_classes]
            pred_names = pred_class_names or _VG_PREDICATE_NAMES[:num_predicates]
            obj_vecs = _obj_edge_vectors(obj_names, glove_dir, embed_dim)
            pred_vecs = _rel_vectors(pred_names, glove_dir, embed_dim)
            with torch.no_grad():
                self.obj_embed.weight.copy_(obj_vecs, non_blocking=True)
                self.rel_embed.weight.copy_(pred_vecs, non_blocking=True)

        # ===== 3. Semantic projections =====
        self.W_sub = MLP(embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_obj = MLP(embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)
        self.W_pred = MLP(embed_dim, self.mlp_dim // 2, self.mlp_dim, 2)

        # ===== 4. Gating =====
        self.gate_sub = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_obj = nn.Linear(self.mlp_dim * 2, self.mlp_dim)
        self.gate_pred = nn.Linear(self.mlp_dim * 2, self.mlp_dim)

        # ===== 5. vis2sem =====
        self.vis2sem = nn.Sequential(
            nn.Linear(self.mlp_dim, self.mlp_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.mlp_dim * 2, self.mlp_dim),
        )

        # ===== 6. project_head =====
        self.project_head = MLP(self.mlp_dim, self.mlp_dim, self.mlp_dim * 2, 2)

        # ===== 7. Residual + LayerNorm =====
        self.linear_sub = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_obj = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_pred = nn.Linear(self.mlp_dim, self.mlp_dim)
        self.linear_rel_rep = nn.Linear(self.mlp_dim, self.mlp_dim)

        self.norm_sub = nn.LayerNorm(self.mlp_dim)
        self.norm_obj = nn.LayerNorm(self.mlp_dim)
        self.norm_rel_rep = nn.LayerNorm(self.mlp_dim)

        self.dropout_sub = nn.Dropout(dropout)
        self.dropout_obj = nn.Dropout(dropout)
        self.dropout_rel_rep = nn.Dropout(dropout)
        self.dropout_rel = nn.Dropout(dropout)
        self.dropout_pred = nn.Dropout(dropout)

        # ===== 8. down_samp (union feature projection) =====
        self.down_samp = MLP(pooling_dim, self.mlp_dim, self.mlp_dim, 2)

        # ===== 9. Union feature fallback =====
        self.union_fallback = nn.Sequential(
            nn.Linear(self.mlp_dim * 2, self.mlp_dim),
            nn.ReLU(inplace=True),
        )

        # ===== 10. logit_scale =====
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ===== 11. Object refinement (uses context_hidden_dim = 512) =====
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )

        # obj_embed1: separate embedding for refinement
        self.obj_embed1 = nn.Embedding(num_classes, embed_dim)
        if glove_dir is not None and os.path.isdir(glove_dir):
            obj_names2 = obj_class_names or _VG_OBJECT_NAMES[:num_classes]
            obj_vecs2 = _obj_edge_vectors(obj_names2, glove_dir, embed_dim)
            with torch.no_grad():
                self.obj_embed1.weight.copy_(obj_vecs2, non_blocking=True)

        # out_obj: make_fc(CONTEXT_HIDDEN_DIM=512, num_classes=151)
        self.out_obj = _make_fc(context_hidden_dim, num_classes)

        # lin_obj_cyx: make_fc(obj_dim + embed_dim + 128, CONTEXT_HIDDEN_DIM=512)
        # obj_dim=2048, embed_dim=300, pos_embed_out=128 → total = 2476
        self.lin_obj_cyx = _make_fc(visual_dim + embed_dim + 128, context_hidden_dim)

        # ===== 12. Frequency bias (official does not use) =====
        self.freq_bias: Optional[FrequencyBias] = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        self._init_weights()

    def _init_weights(self):
        """kaiming_uniform_ init — matches official make_fc convention."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ── SGDet per-class NMS — matches official nms_per_cls exactly ──

    def nms_per_cls(
        self,
        obj_dists: torch.Tensor,
        boxes_per_cls: torch.Tensor,
        num_objs: int,
    ) -> torch.Tensor:
        """Per-class NMS matching official ``nms_per_cls`` exactly."""
        is_overlap = (
            _nms_overlaps(boxes_per_cls).detach().cpu().numpy() >= self.nms_thresh
        )
        out_dists_sampled = F.softmax(obj_dists, -1).detach().cpu().numpy()
        out_dists_sampled[:, 0] = -1  # set bg to -1

        out_label = obj_dists.new(num_objs).fill_(0)
        for i in range(num_objs):
            box_ind, cls_ind = np.unravel_index(
                out_dists_sampled.argmax(), out_dists_sampled.shape
            )
            out_label[int(box_ind)] = int(cls_ind)
            out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
            out_dists_sampled[box_ind] = -1.0

        return out_label.long()

    # ── Object label refinement — matches official refine_obj_labels ──

    def _refine_obj_labels(
        self,
        roi_features: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        obj_dists: Optional[torch.Tensor] = None,
        use_gt_label: bool = True,
        boxes_per_cls: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine object labels matching official ``refine_obj_labels``.

        Official flow:
          pos_embed = self.pos_embed(encode_box_info(proposals))
          if USE_GT_OBJECT_LABEL:
              obj_labels → obj_embed1(obj_labels)
          else:
              obj_logits → softmax → @ obj_embed1.weight
          obj_pre_rep = lin_obj_cyx(cat([roi_features, obj_embed, pos_embed], -1))
          if predcls: obj_dists = to_onehot(obj_preds)
          elif sgdet + !training: nms_per_cls
          else: max prediction
        """
        pos_feat = self.pos_embed(_encode_box_info(boxes))  # [N, 128]

        if use_gt_label:
            obj_labels = labels.long()
            obj_embed_feat = self.obj_embed1(obj_labels)  # [N, embed_dim]
        elif obj_dists is not None:
            obj_embed_feat = F.softmax(obj_dists, dim=1) @ self.obj_embed1.weight
        else:
            obj_embed_feat = self.obj_embed1.weight.mean(0, keepdim=True).expand(
                roi_features.size(0), -1
            )

        combined = torch.cat([roi_features, obj_embed_feat, pos_feat], dim=-1)
        obj_pre_rep = self.lin_obj_cyx(combined)  # [N, 512] → [N, 151] via out_obj

        if use_gt_label:
            # predcls — GT labels directly
            obj_preds = obj_labels
            obj_dists_out = torch.zeros(
                roi_features.size(0),
                self.num_classes,
                device=roi_features.device,
            )
            obj_dists_out.scatter_(1, obj_labels.unsqueeze(1), 1.0)
        else:
            obj_dists_out = self.out_obj(obj_pre_rep)
            if boxes_per_cls is not None:
                # sgdet mode with per-class NMS
                obj_preds = self.nms_per_cls(
                    obj_dists_out, boxes_per_cls, roi_features.size(0)
                ).long()
            else:
                # sgcls — max prediction (skip bg at index 0)
                obj_preds = (obj_dists_out[:, 1:].max(1)[1] + 1).long()

        return obj_dists_out, obj_preds

    # ── Union feature resolution ──

    # ── Prototype regularization losses — exact match of official ──

    def _proto_losses(
        self,
        predicate_proto_norm: torch.Tensor,
        predicate_proto: torch.Tensor,
        rel_rep: torch.Tensor,
        rel_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Three prototype losses matching official exactly.

        Args:
            predicate_proto_norm: [K, 2*mlp_dim] L2-normalized prototypes.
            predicate_proto: [K, 2*mlp_dim] unnormalized prototypes.
            rel_rep: [P, 2*mlp_dim] relation representations.
            rel_labels: [P] GT predicate indices (0..K-1), 0 = bg.
        """
        add_losses: dict[str, torch.Tensor] = {}
        device = predicate_proto.device
        K = predicate_proto.size(0)
        P = rel_labels.size(0)

        # ---- L_{2,1}: ||C_norm @ C_norm.T||_{2,1} / K^2 ----
        simil_mat = predicate_proto_norm @ predicate_proto_norm.t()
        l21 = torch.norm(torch.norm(simil_mat, p=2, dim=1), p=1) / (K * K)
        add_losses["l21_loss"] = l21

        # ---- dist_loss2: max(0, -d- + gamma2) ----
        gamma2 = 7.0
        proto_a = predicate_proto.unsqueeze(dim=1).expand(-1, K, -1)
        proto_b = predicate_proto.detach().unsqueeze(dim=0).expand(K, -1, -1)
        proto_dis_mat = (proto_a - proto_b).norm(dim=2) ** 2
        sorted_proto_dis, _ = torch.sort(proto_dis_mat, dim=1)
        topK_proto_dis = sorted_proto_dis[:, :2].sum(dim=1) / 1
        dist_loss2 = torch.max(
            torch.zeros(K, device=device),
            -topK_proto_dis + gamma2,
        ).mean()
        add_losses["dist_loss2"] = dist_loss2

        # ---- loss_dis: max(0, g+ - g- + gamma1) ----
        gamma1 = 1.0
        rel_expand = rel_rep.unsqueeze(dim=1).expand(-1, K, -1)
        proto_expand = predicate_proto.unsqueeze(dim=0).expand(P, -1, -1)
        distance_set = (rel_expand - proto_expand).norm(dim=2) ** 2

        mask_neg = torch.ones(P, K, device=device)
        mask_neg[torch.arange(P), rel_labels] = 0
        distance_set_neg = distance_set * mask_neg
        distance_set_pos = distance_set[torch.arange(P), rel_labels]

        sorted_distance_set_neg, _ = torch.sort(distance_set_neg, dim=1)
        topK_neg = sorted_distance_set_neg[:, :11].sum(dim=1) / 10  # k1 = 10

        loss_dis = torch.max(
            torch.zeros(P, device=device),
            distance_set_pos - topK_neg + gamma1,
        ).mean()
        add_losses["loss_dis"] = loss_dis

        return add_losses

    # ── Pair sampling (matches official gtbox_relsample for PredCls) ──

    def _sample_pairs(
        self,
        pairs: torch.Tensor,
        rel_annotations: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a balanced set of fg/bg pairs for training.

        Matches official ``gtbox_relsample`` behaviour:
          - num_pos_per_img = batch_size_per_image * positive_fraction
          - Sample up to num_pos fg pairs, fill rest with bg pairs
          - Total pairs = min(train_pairs_per_image, N*(N-1))
        """
        P = pairs.size(0)
        max_pairs = min(self.train_pairs_per_image, P)
        num_pos_per_img = int(max_pairs * self.train_positive_fraction)

        # Build a lookup of annotated pairs
        ann_set = set()
        ann_labels = {}
        for ann in rel_annotations:
            s, o, p_vg = int(ann[0]), int(ann[1]), int(ann[2])
            ann_set.add((s, o))
            ann_labels[(s, o)] = p_vg

        # Find fg pairs (annotated) and bg pairs (not annotated)
        fg_mask = torch.zeros(P, dtype=torch.bool, device=device)
        fg_labels = torch.zeros(P, dtype=torch.long, device=device)
        for p_idx in range(P):
            key = (int(pairs[p_idx, 0]), int(pairs[p_idx, 1]))
            if key in ann_set:
                fg_mask[p_idx] = True
                fg_labels[p_idx] = ann_labels[key]

        fg_idxs = fg_mask.nonzero(as_tuple=True)[0]
        bg_idxs = (~fg_mask).nonzero(as_tuple=True)[0]

        # Sample fg
        if fg_idxs.numel() > num_pos_per_img:
            perm = torch.randperm(fg_idxs.numel(), device=device)[:num_pos_per_img]
            fg_idxs = fg_idxs[perm]
        num_fg = min(fg_idxs.numel(), num_pos_per_img)

        # Sample bg
        num_bg = max_pairs - num_fg
        if bg_idxs.numel() > num_bg:
            perm = torch.randperm(bg_idxs.numel(), device=device)[:num_bg]
            bg_idxs = bg_idxs[perm]

        sampled_idxs = torch.cat([fg_idxs, bg_idxs], dim=0)
        sampled_labels = torch.cat(
            [
                fg_labels[fg_idxs],
                torch.zeros(bg_idxs.numel(), dtype=torch.long, device=device),
            ],
            dim=0,
        )

        return sampled_idxs, sampled_labels

    # ── Forward ──

    def forward(
        self,
        visual_feats: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        return_obj_preds: bool = False,
        union_feats: Optional[torch.Tensor] = None,
        rel_annotations: Optional[torch.Tensor] = None,
        obj_dists: Optional[torch.Tensor] = None,
        boxes_per_cls: Optional[torch.Tensor] = None,
        precomputed_union_feats: Optional[torch.Tensor] = None,
        _compute_union_fn: Optional = None,
        _fpn_features: Optional[tuple] = None,
        _image_size: Optional[torch.Tensor] = None,
    ):
        """Forward pass — strict match with official PrototypeEmbeddingNetwork.

        Union features are resolved in this priority:
          1. ``union_feats`` / ``precomputed_union_feats`` — pre-extracted (legacy)
          2. ``_compute_union_fn(fpn_features, boxes, pairs, img_size)`` — official
             PENetUnionFeatureExtractor with FPNPooler + rect_conv
          3. Learned fallback projection from sub+obj features
        """
        device = visual_feats.device
        N = visual_feats.size(0)
        is_training = self.training
        use_gt = is_training or not return_obj_preds

        # Resolve image_size (new name _image_size, for backward compat)
        img_size = _image_size

        # ── 1. Object label refinement ──
        entity_dists, entity_preds = self._refine_obj_labels(
            visual_feats,
            boxes,
            labels,
            obj_dists=obj_dists,
            use_gt_label=use_gt,
            boxes_per_cls=boxes_per_cls if not use_gt else None,
        )

        # ── 2. Split visual features into sub / obj ──
        entity_rep = self.post_emb(visual_feats)  # [N, mlp_dim*2]
        entity_rep = entity_rep.view(N, 2, self.mlp_dim)
        sub_rep = entity_rep[:, 1].contiguous().view(-1, self.mlp_dim)  # xs
        obj_rep = entity_rep[:, 0].contiguous().view(-1, self.mlp_dim)  # xo

        # ── 3. Word embeddings ──
        entity_embeds = self.obj_embed(entity_preds)

        # ── 4. Generate all directed pairs ──
        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": entity_preds,
                "obj_logits": entity_dists if return_obj_preds else None,
                "predicate_bg_index": "first",
                "add_losses": {},
            }

        # ── 5. Determine which pairs to use ──
        if is_training and rel_annotations is not None:
            use_idxs, rel_labels_for_loss = self._sample_pairs(
                pairs,
                rel_annotations,
                device,
            )
            P = use_idxs.size(0)
            s_idx = pairs[use_idxs, 0]
            o_idx = pairs[use_idxs, 1]
        else:
            P = pairs.size(0)
            s_idx = pairs[:, 0]
            o_idx = pairs[:, 1]
            rel_labels_for_loss = None

        # ── 6. Semantic prototypes ──
        s_embed = self.W_sub(entity_embeds[s_idx])  # Ws @ ts
        o_embed = self.W_obj(entity_embeds[o_idx])  # Wo @ to

        # ── 7. Visual → semantic ──
        sem_sub = self.vis2sem(sub_rep[s_idx])  # h(xs)
        sem_obj = self.vis2sem(obj_rep[o_idx])  # h(xo)

        # ── 8. Gated fusion ──
        gate_sem_sub = torch.sigmoid(
            self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1))
        )
        gate_sem_obj = torch.sigmoid(
            self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1))
        )
        sub = s_embed + sem_sub * gate_sem_sub
        obj = o_embed + sem_obj * gate_sem_obj

        # ── 9. Residual + LN ──
        sub = self.norm_sub(self.dropout_sub(torch.relu(self.linear_sub(sub))) + sub)
        obj = self.norm_obj(self.dropout_obj(torch.relu(self.linear_obj(obj))) + obj)

        # ── 10. Fusion ──
        fusion_so = fusion_func(sub, obj)  # [P, mlp_dim]

        # ── 11. Union feature gating ──
        # Priority: 1) precomputed  2) FPNPooler extractor  3) fallback
        resolved_union = None
        if union_feats is not None and union_feats.size(0) == P:
            resolved_union = union_feats
        elif (
            precomputed_union_feats is not None and precomputed_union_feats.size(0) == P
        ):
            resolved_union = precomputed_union_feats
        elif (
            _compute_union_fn is not None
            and _fpn_features is not None
            and img_size is not None
        ):
            resolved_union = _compute_union_fn(
                _fpn_features,
                boxes,
                (
                    pairs[use_idxs]
                    if is_training and rel_annotations is not None
                    else pairs
                ),
                img_size,
            )

        if resolved_union is not None:
            sem_pred = self.vis2sem(self.down_samp(resolved_union))
        else:
            fallback_inp = torch.cat([sub_rep[s_idx], obj_rep[o_idx]], dim=-1)
            sem_pred = self.vis2sem(self.union_fallback(fallback_inp))

        gate_sem_pred = torch.sigmoid(
            self.gate_pred(torch.cat([fusion_so, sem_pred], dim=-1))
        )
        rel_rep = fusion_so - sem_pred * gate_sem_pred  # r = F(s,o) - gp · h(xu)

        # ── 12. Predicate semantic prototypes ──
        predicate_proto = self.W_pred(self.rel_embed.weight)  # [K, mlp_dim]

        # ── 13. Residual + LN ──
        rel_rep = self.norm_rel_rep(
            self.dropout_rel_rep(torch.relu(self.linear_rel_rep(rel_rep))) + rel_rep
        )

        # ── 14. Projection head ──
        rel_rep_proj = self.project_head(
            self.dropout_rel(torch.relu(rel_rep))
        )  # [P, mlp_dim*2]
        predicate_proto_proj = self.project_head(
            self.dropout_pred(torch.relu(predicate_proto))
        )  # [K, mlp_dim*2]

        # ── 15. L2 normalize ──
        rel_rep_norm = rel_rep_proj / rel_rep_proj.norm(dim=1, keepdim=True)
        predicate_proto_norm = predicate_proto_proj / predicate_proto_proj.norm(
            dim=1, keepdim=True
        )

        # ── 16. Cosine similarity ──
        rel_dists = (
            rel_rep_norm @ predicate_proto_norm.t() * self.logit_scale.exp()
        )  # [P, K]

        # ── 17. Frequency bias ──
        if self.freq_bias is not None:
            rel_dists = self.freq_bias(rel_dists)

        # ── 18. Prototype losses (training only) ──
        add_losses: dict[str, torch.Tensor] = {}
        if is_training and rel_labels_for_loss is not None:
            add_losses = self._proto_losses(
                predicate_proto_norm,
                predicate_proto_proj,
                rel_rep_proj,
                rel_labels_for_loss,
            )

        # Build output pairs
        if is_training and rel_annotations is not None:
            out_pairs = pairs[use_idxs]
        else:
            out_pairs = pairs

        return {
            "rel_logits": rel_dists,
            "pair_indices": out_pairs,
            "obj_labels": entity_preds,
            "obj_logits": entity_dists if return_obj_preds else None,
            "sub_boxes": boxes[out_pairs[:, 0]],
            "obj_boxes": boxes[out_pairs[:, 1]],
            "predicate_bg_index": "first",
            "add_losses": add_losses,
        }


# =========================================================================
# Builder
# =========================================================================


def build_penet(args) -> PENetContext:
    """Build PE-NET from config args, matching official hyperparameters."""
    model = PENetContext(
        num_classes=getattr(args, "entity_nums", 151),
        num_predicates=getattr(args, "rel_nums", 51),
        visual_dim=getattr(args, "visual_dim", 4096),  # official obj_dim
        hidden_dim=getattr(args, "hidden_dim", 2048),
        context_hidden_dim=getattr(args, "penet_context_hidden_dim", 512),
        embed_dim=getattr(args, "penet_embed_dim", 300),
        pooling_dim=getattr(args, "penet_pooling_dim", 4096),
        glove_dir=getattr(args, "glove_dir", None),
        obj_class_names=getattr(args, "obj_class_names", None),
        pred_class_names=getattr(args, "pred_class_names", None),
        use_freq_bias=getattr(args, "use_freq_bias", False),
        freq_bias_eps=getattr(args, "freq_bias_eps", 1e-12),
        dropout=getattr(args, "dropout", 0.2),
        nms_thresh=getattr(args, "penet_nms_thresh", 0.5),
        roi_output_size=getattr(args, "roi_output_size", 7),
        train_pairs_per_image=getattr(args, "penet_train_pairs", 512),
        train_positive_fraction=getattr(args, "penet_pos_frac", 0.25),
    )
    return model
