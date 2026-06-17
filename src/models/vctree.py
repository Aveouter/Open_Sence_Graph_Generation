"""
VCTree Model for Scene Graph Generation.

Reference: "Learning to Compose Dynamic Tree Structures for Visual Contexts"
(Tang et al., CVPR 2019)

Architecture:
  1. Object Encoder: Same as Motifs (visual + label embedding)
  2. Tree Construction: Build a hierarchical tree over objects using
     pairwise affinity scores (learned + spatial)
  3. TreeLSTM Context: Bidirectional TreeLSTM encodes context along tree edges
  4. Pair Feature Generation + Predicate Classification (same as Motifs)

VCTree replaces the BiLSTM context with a dynamic tree structure that
better captures hierarchical relationships between objects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


# ──────────────────────────────────────────────
# TreeLSTM Implementation
# ──────────────────────────────────────────────

class BinaryTreeLSTMCell(nn.Module):
    """Binary TreeLSTM cell (Child-Sum variant).

    Takes left and right child hidden states and produces parent state.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Gates: i, f_left, f_right, o, u
        self.iou_x = nn.Linear(input_dim, 3 * hidden_dim)  # input, output, update
        self.iou_h = nn.Linear(2 * hidden_dim, 3 * hidden_dim)  # from both children
        self.f_x = nn.Linear(input_dim, 2 * hidden_dim)  # forget gates for each child
        self.f_h = nn.Linear(2 * hidden_dim, 2 * hidden_dim)

    def forward(self, x: torch.Tensor,
                h_left: torch.Tensor, c_left: torch.Tensor,
                h_right: torch.Tensor, c_right: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single TreeLSTM cell forward.

        Args:
            x: [input_dim] input feature.
            h_left, c_left: left child state.
            h_right, c_right: right child state.

        Returns:
            (h, c) parent state.
        """
        h_children = torch.cat([h_left, h_right], dim=-1)

        # Input, output, update gates
        iou = self.iou_x(x) + self.iou_h(h_children)
        i, o, u = torch.chunk(iou, 3, dim=-1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # Forget gates
        f = self.f_x(x) + self.f_h(h_children)
        f_left, f_right = torch.chunk(f, 2, dim=-1)
        f_left = torch.sigmoid(f_left)
        f_right = torch.sigmoid(f_right)

        # Cell state update
        c = i * u + f_left * c_left + f_right * c_right
        h = o * torch.tanh(c)

        return self.dropout(h), c


class BidirectionalTreeLSTM(nn.Module):
    """Bidirectional TreeLSTM encoder.

    Processes the tree bottom-up, then top-down to produce bidirectional
    context for each node.
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Bottom-up cell
        self.bottom_up_cell = BinaryTreeLSTMCell(input_dim, hidden_dim, dropout)
        # Top-down cell
        self.top_down_cell = BinaryTreeLSTMCell(hidden_dim, hidden_dim, dropout)

        # Projection
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, features: torch.Tensor,
                tree: List[Tuple[int, int, int]]) -> torch.Tensor:
        """
        Args:
            features: [N, input_dim] node features (leaf nodes only).
            tree: List of (parent, left_child, right_child) indices.
                  Internal nodes have parent >= N; leaf children = -1.

        Returns:
            [N, hidden_dim] bidirectional context for original N nodes.
        """
        N = features.size(0)
        device = features.device
        h_dim = self.hidden_dim

        if len(tree) == 0:
            return self.proj(torch.cat([
                torch.zeros(N, h_dim, device=device),
                torch.zeros(N, h_dim, device=device),
            ], dim=-1))

        # Determine total nodes (original + internal)
        max_idx = max(max(p, l if l >= 0 else 0, r if r >= 0 else 0)
                      for p, l, r in tree) + 1
        total_nodes = max(N, max_idx)

        # Initialize state tensors with capacity for all nodes
        h_bottom = torch.zeros(total_nodes, h_dim, device=device)
        c_bottom = torch.zeros(total_nodes, h_dim, device=device)

        # Bottom-up pass
        for parent, left, right in tree:
            # Feature: from original features if leaf/root, else zero
            if parent < N:
                x = features[parent]
            else:
                x = torch.zeros(features.size(1), device=device)

            if left >= 0:
                hl, cl = h_bottom[left], c_bottom[left]
            else:
                hl = torch.zeros(h_dim, device=device)
                cl = torch.zeros(h_dim, device=device)

            if right >= 0:
                hr, cr = h_bottom[right], c_bottom[right]
            else:
                hr = torch.zeros(h_dim, device=device)
                cr = torch.zeros(h_dim, device=device)

            h_bottom[parent], c_bottom[parent] = self.bottom_up_cell(x, hl, cl, hr, cr)

        # Top-down pass (reverse of bottom-up)
        h_top = torch.zeros(total_nodes, h_dim, device=device)
        c_top = torch.zeros(total_nodes, h_dim, device=device)

        for parent, left, right in reversed(tree):
            if left >= 0:
                x_left = h_bottom[left]
                hp = h_top[parent] if h_top[parent].abs().sum() > 0 else h_bottom[parent]
                cp = c_top[parent] if c_top[parent].abs().sum() > 0 else c_bottom[parent]
                h_top[left], c_top[left] = self.top_down_cell(
                    x_left,
                    torch.zeros(h_dim, device=device), torch.zeros(h_dim, device=device),
                    hp, cp
                )

            if right >= 0:
                x_right = h_bottom[right]
                hp_r = h_top[parent] if h_top[parent].abs().sum() > 0 else h_bottom[parent]
                cp_r = c_top[parent] if c_top[parent].abs().sum() > 0 else c_bottom[parent]
                h_top[right], c_top[right] = self.top_down_cell(
                    x_right,
                    torch.zeros(h_dim, device=device), torch.zeros(h_dim, device=device),
                    hp_r, cp_r
                )

        # Return context only for original N nodes
        context = torch.cat([h_bottom[:N], h_top[:N]], dim=-1)
        return self.proj(context)


# ──────────────────────────────────────────────
# Tree Construction
# ──────────────────────────────────────────────

class TreeConstructor(nn.Module):
    """Build a binary tree over objects using an affinity-based scoring.

    Uses a hybrid score combining:
      - Visual affinity (learned pairwise MLP)
      - Spatial affinity (IoU-based)
      - Semantic affinity (label co-occurrence)

    The tree is built greedily: iteratively merge the pair with highest
    affinity into a parent node.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.affinity_net = nn.Sequential(
            nn.Linear(feat_dim * 2 + 5, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def _spatial_affinity(self, box_i: torch.Tensor,
                           box_j: torch.Tensor) -> torch.Tensor:
        """Compute spatial affinity (IoU + relative position)."""
        ix1, iy1 = box_i[0] - box_i[2] / 2, box_i[1] - box_i[3] / 2
        ix2, iy2 = box_i[0] + box_i[2] / 2, box_i[1] + box_i[3] / 2
        jx1, jy1 = box_j[0] - box_j[2] / 2, box_j[1] - box_j[3] / 2
        jx2, jy2 = box_j[0] + box_j[2] / 2, box_j[1] + box_j[3] / 2

        inter_w = max(0, min(ix2, jx2) - max(ix1, jx1))
        inter_h = max(0, min(iy2, jy2) - max(iy1, jy1))
        inter = inter_w * inter_h
        area_i = box_i[2] * box_i[3]
        area_j = box_j[2] * box_j[3]
        iou = inter / (area_i + area_j - inter + 1e-6)

        # Also include size ratio similarity
        size_ratio = min(area_i, area_j) / (max(area_i, area_j) + 1e-6)

        return (iou + size_ratio) / 2

    def forward(self, feats: torch.Tensor, boxes: torch.Tensor
                ) -> List[Tuple[int, int, int]]:
        """Build tree from object features and boxes.

        Args:
            feats: [N, D] object features.
            boxes: [N, 4] object boxes (cx, cy, w, h).

        Returns:
            List of (parent, left, right) triples defining the tree.
            Leaf nodes have children = -1.
        """
        N = feats.size(0)
        device = feats.device

        if N <= 1:
            return [(0, -1, -1)]

        tree = []
        active = set(range(N))
        next_node_id = N

        # Expandable node features and boxes
        all_feats = list(feats)
        all_boxes = list(boxes)

        while len(active) > 1:
            active_list = sorted(active)
            best_score = -float('inf')
            best_pair = None

            # Find best pair to merge
            for idx_i, i in enumerate(active_list):
                for j in active_list[idx_i + 1:]:
                    # Pairwise affinity
                    feat_cat = torch.cat([all_feats[i], all_feats[j],
                                          torch.tensor([
                                              (all_boxes[i][0] - all_boxes[j][0]).item(),
                                              (all_boxes[i][1] - all_boxes[j][1]).item(),
                                              self._spatial_affinity(all_boxes[i], all_boxes[j]).item(),
                                              all_boxes[i][2].item() * all_boxes[i][3].item(),
                                              all_boxes[j][2].item() * all_boxes[j][3].item(),
                                          ], device=device)], dim=-1)
                    score = self.affinity_net(feat_cat.unsqueeze(0)).squeeze()

                    if score > best_score:
                        best_score = score
                        best_pair = (i, j)

            if best_pair is None:
                break

            i, j = best_pair
            parent = next_node_id
            next_node_id += 1

            # Merge: parent feature = average of children
            parent_feat = (all_feats[i] + all_feats[j]) / 2
            parent_box = torch.stack([
                (all_boxes[i][0] * all_boxes[i][2] * all_boxes[i][3] +
                 all_boxes[j][0] * all_boxes[j][2] * all_boxes[j][3]) /
                (all_boxes[i][2] * all_boxes[i][3] + all_boxes[j][2] * all_boxes[j][3] + 1e-6),
                (all_boxes[i][1] * all_boxes[i][2] * all_boxes[i][3] +
                 all_boxes[j][1] * all_boxes[j][2] * all_boxes[j][3]) /
                (all_boxes[i][2] * all_boxes[i][3] + all_boxes[j][2] * all_boxes[j][3] + 1e-6),
                all_boxes[i][2] + all_boxes[j][2],
                all_boxes[i][3] + all_boxes[j][3],
            ])

            all_feats.append(parent_feat)
            all_boxes.append(parent_box)

            tree.append((parent, i, j))
            active.discard(i)
            active.discard(j)
            active.add(parent)

        # Ensure all original nodes are covered (even if they weren't merged)
        if len(tree) == 0 and N > 0:
            # Build a proper binary chain tree so every node gets context
            tree = []
            next_id = N
            current = 0
            for i in range(1, N):
                tree.append((next_id, current, i))
                current = next_id
                next_id += 1

        return tree


# ──────────────────────────────────────────────
# VCTree Model
# ──────────────────────────────────────────────

class VCTreeModel(nn.Module):
    """VCTree model for scene graph generation.

    Args:
        num_classes: Number of object classes (including bg).
        num_predicates: Number of predicate classes.
        visual_dim: Dimension of input visual features.
        hidden_dim: Hidden dimension.
        use_freq_bias: Whether to use frequency bias.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.use_freq_bias = use_freq_bias

        # Object encoding (same as Motifs)
        from src.models.motifs import ObjectEncoder, FrequencyBias, PairFeatureGenerator
        self.obj_encoder = ObjectEncoder(visual_dim, num_classes, hidden_dim, dropout)

        # TreeLSTM context (replaces BiLSTM)
        self.tree_constructor = TreeConstructor(hidden_dim, hidden_dim // 2)
        self.tree_lstm = BidirectionalTreeLSTM(hidden_dim, hidden_dim, dropout)

        self.obj_post = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Pair features and classification
        self.pair_gen = PairFeatureGenerator(hidden_dim, hidden_dim)
        self.pred_classifier = nn.Linear(hidden_dim, num_predicates)

        # Frequency bias
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        # Object classifier
        self.obj_classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def _generate_pairs(self, N: int, device: torch.device) -> torch.Tensor:
        from src.models.motifs import generate_object_pairs
        return generate_object_pairs(N, device)

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        N = visual_feats.size(0)
        device = visual_feats.device

        # Object encoding
        obj_feats = self.obj_encoder(visual_feats, labels)

        # Build tree and apply TreeLSTM
        tree = self.tree_constructor(obj_feats, boxes)
        obj_context = self.tree_lstm(obj_feats, tree)
        obj_context = self.obj_post(obj_context)

        # Object classification
        obj_logits = None
        if return_obj_preds:
            obj_logits = self.obj_classifier(obj_context)

        # Generate pairs
        pairs = self._generate_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "obj_logits": obj_logits,
                "obj_labels": obj_logits.argmax(-1) if obj_logits is not None else labels,
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
            }

        # Pair features
        pair_feats = self.pair_gen(obj_context, boxes, pairs)
        rel_logits = self.pred_classifier(pair_feats)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        return {
            "rel_logits": rel_logits,
            "obj_logits": obj_logits,
            "obj_labels": obj_logits.argmax(-1) if obj_logits is not None else labels,
            "pair_indices": pairs,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
        }


def build_vctree(args) -> VCTreeModel:
    """Build VCTree model from config args."""
    model = VCTreeModel(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
