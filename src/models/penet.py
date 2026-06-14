"""
PE-NET: Prototype-based Embedding Network for Scene Graph Generation.

Reference: "Prototype-based Embedding Network for Scene Graph Generation"
(Zheng et al., CVPR 2023)

Architecture:
  1. Object Encoder: visual projection + GloVe word embeddings
  2. Semantic-Visual Fusion: gated combination of word embeddings and visual features
     s = Ws*ts + gs · h(xs)   (subject: semantic prototype + gated visual)
     o = Wo*to + go · h(xo)   (object: semantic prototype + gated visual)
  3. Prototype-based Relation: fusion(s, o) → predicate prediction
  4. Object Refinement (optional): refine object labels from context

Key innovation: GloVe word embeddings as semantic prototypes,
gated fusion of visual and semantic features for compositional generalization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .motifs import FrequencyBias, generate_object_pairs


# ──────────────────────────────────────────────
# MLP helper
# ──────────────────────────────────────────────

class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(num_layers):
            next_dim = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(current_dim, next_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            current_dim = next_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# PE-NET Context
# ──────────────────────────────────────────────

class PENetContext(nn.Module):
    """Prototype-based Embedding Network for SGG.

    Supports three modes via config:
      - textual_features_only: use only GloVe embeddings (zero visual)
      - visual_features_only: use only visual features (zero semantic)
      - full: gated fusion of both (default)

    Args:
        num_classes: Number of object classes.
        num_predicates: Number of predicate classes.
        visual_dim: Input visual feature dimension.
        hidden_dim: Internal hidden dimension (mlp_dim in paper).
        embed_dim: GloVe word embedding dimension.
        use_freq_bias: Whether to use frequency bias.
        textual_only: Use only text features.
        visual_only: Use only visual features.
        dropout: Dropout probability.
    """

    def __init__(self, num_classes: int = 151, num_predicates: int = 51,
                 visual_dim: int = 2048, hidden_dim: int = 512,
                 embed_dim: int = 200,
                 use_freq_bias: bool = True, freq_bias_eps: float = 1e-12,
                 textual_only: bool = False, visual_only: bool = False,
                 dropout: float = 0.1):
        super().__init__()

        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.use_freq_bias = use_freq_bias
        self.textual_only = textual_only
        self.visual_only = visual_only

        # Visual feature projection: [visual_dim] → [hidden_dim*2] split to sub/obj
        self.post_emb = nn.Linear(visual_dim, hidden_dim * 2)

        # Semantic projections for subject/object
        self.W_sub = MLP(embed_dim, hidden_dim // 2, hidden_dim, 2)
        self.W_obj = MLP(embed_dim, hidden_dim // 2, hidden_dim, 2)

        # Gating: concat(semantic, visual) → gate value
        self.gate_sub = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gate_obj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Visual-to-semantic projection
        self.vis2sem = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Residual + normalization (for convergence)
        self.linear_sub = nn.Linear(hidden_dim, hidden_dim)
        self.linear_obj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_sub = nn.LayerNorm(hidden_dim)
        self.norm_obj = nn.LayerNorm(hidden_dim)
        self.dropout_sub = nn.Dropout(dropout)
        self.dropout_obj = nn.Dropout(dropout)

        # Object label embeddings (learnable, initialized from GloVe later)
        self.obj_embed = nn.Embedding(num_classes, embed_dim)

        # Position embedding (for object refinement)
        self.pos_embed = nn.Sequential(
            nn.Linear(9, 32),
            nn.BatchNorm1d(32, momentum=0.001),
            nn.Linear(32, 128),
            nn.ReLU(inplace=True),
        )

        # Object refinement head
        self.obj_refine = not textual_only
        if self.obj_refine:
            self.out_obj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.lin_obj_cyx = nn.Sequential(
                nn.Linear(visual_dim + embed_dim + 128, hidden_dim),
                nn.ReLU(inplace=True),
            )

        # Fusion function (element-wise)
        self.fusion_dim_reduce = nn.Linear(hidden_dim * 2, hidden_dim)

        # Predicate classifier
        self.pred_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_predicates),
        )

        # Frequency bias
        self.freq_bias = None
        if use_freq_bias:
            self.freq_bias = FrequencyBias(num_predicates, freq_bias_eps)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _encode_box_info(self, boxes: torch.Tensor) -> torch.Tensor:
        """9-dim box geometry feature."""
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        area = w * h
        return torch.stack([cx, cy, w, h, x1, y1, x2, y2, area], dim=-1)

    def _refine_obj_labels(self, roi_features: torch.Tensor,
                           boxes: torch.Tensor) -> torch.Tensor:
        """Refine object labels using visual + semantic + spatial features."""
        obj_embeds = self.obj_embed.weight  # [C, embed_dim]
        # Use average embedding as initial guess
        avg_embed = obj_embeds.mean(0, keepdim=True).expand(roi_features.size(0), -1)
        pos_emb = self.pos_embed(self._encode_box_info(boxes))
        combined = torch.cat([roi_features, avg_embed, pos_emb], dim=-1)
        feats = self.lin_obj_cyx(combined)
        obj_logits = self.out_obj(feats)
        return obj_logits

    def _fusion(self, sub: torch.Tensor, obj: torch.Tensor) -> torch.Tensor:
        """Fuse subject and object features: F(s, o).

        Uses element-wise product as the fusion function,
        which captures interaction between subject and object prototypes.
        """
        fused = torch.cat([sub, obj], dim=-1)
        fused = self.fusion_dim_reduce(fused)
        return F.relu(fused)

    def forward(self, visual_feats: torch.Tensor, boxes: torch.Tensor,
                labels: torch.Tensor, return_obj_preds: bool = False):
        """
        Args:
            visual_feats: [N, visual_dim] ROI features.
            boxes: [N, 4] (cx, cy, w, h) normalized.
            labels: [N] object class labels.
            return_obj_preds: If True, also return object predictions.

        Returns:
            dict with rel_logits, pair_indices, etc.
        """
        N = visual_feats.size(0)
        device = visual_feats.device

        # Refine object labels (for SGCLS)
        obj_logits = None
        if return_obj_preds and self.obj_refine:
            obj_logits = self._refine_obj_labels(visual_feats, boxes)
            entity_preds = obj_logits.argmax(-1)
        else:
            entity_preds = labels

        # Split visual features into subject/object representations
        entity_rep = self.post_emb(visual_feats)  # [N, H*2]
        entity_rep = entity_rep.view(N, 2, self.hidden_dim)
        sub_rep = entity_rep[:, 1]  # [N, H] — subject representation
        obj_rep = entity_rep[:, 0]  # [N, H] — object representation

        # Get word embeddings for predicted/GT entity labels
        if not self.visual_only:
            entity_embeds = self.obj_embed(entity_preds)  # [N, embed_dim]
        else:
            entity_embeds = torch.zeros(N, self.embed_dim, device=device)

        # Generate all pairs
        pairs = generate_object_pairs(N, device)
        if pairs.numel() == 0:
            return {
                "rel_logits": visual_feats.new_zeros(0, self.num_predicates),
                "pair_indices": pairs,
                "sub_boxes": boxes.new_zeros(0, 4),
                "obj_boxes": boxes.new_zeros(0, 4),
                "obj_labels": entity_preds,
            }

        P = pairs.size(0)
        s_idx = pairs[:, 0]
        o_idx = pairs[:, 1]

        # Semantic prototypes
        s_embed = self.W_sub(entity_embeds[s_idx])  # [P, H]
        o_embed = self.W_obj(entity_embeds[o_idx])  # [P, H]

        # Visual to semantic
        sem_sub = self.vis2sem(sub_rep[s_idx])  # [P, H]
        sem_obj = self.vis2sem(obj_rep[o_idx])  # [P, H]

        if self.textual_only:
            sub = s_embed
            obj = o_embed
        elif self.visual_only:
            sub = sem_sub
            obj = sem_obj
        else:
            # Gated fusion
            gate_sem_sub = torch.sigmoid(
                self.gate_sub(torch.cat([s_embed, sem_sub], dim=-1)))
            gate_sem_obj = torch.sigmoid(
                self.gate_obj(torch.cat([o_embed, sem_obj], dim=-1)))
            sub = s_embed + sem_sub * gate_sem_sub
            obj = o_embed + sem_obj * gate_sem_obj

        # Residual + normalization for convergence
        sub = self.norm_sub(self.dropout_sub(F.relu(self.linear_sub(sub))) + sub)
        obj = self.norm_obj(self.dropout_obj(F.relu(self.linear_obj(obj))) + obj)

        # Fusion F(s, o)
        fusion_so = self._fusion(sub, obj)  # [P, H]

        # Predicate classification
        rel_logits = self.pred_classifier(fusion_so)

        if self.freq_bias is not None:
            rel_logits = self.freq_bias(rel_logits)

        return {
            "rel_logits": rel_logits,
            "pair_indices": pairs,
            "obj_labels": entity_preds,
            "obj_logits": obj_logits,
            "sub_boxes": boxes[pairs[:, 0]],
            "obj_boxes": boxes[pairs[:, 1]],
        }


# ──────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────

def build_penet(args) -> PENetContext:
    """Build PE-NET model from config args."""
    model = PENetContext(
        num_classes=getattr(args, 'entity_nums', 151),
        num_predicates=getattr(args, 'rel_nums', 51),
        visual_dim=getattr(args, 'visual_dim', 2048),
        hidden_dim=getattr(args, 'hidden_dim', 512),
        embed_dim=getattr(args, 'penet_embed_dim', 200),
        use_freq_bias=getattr(args, 'use_freq_bias', True),
        freq_bias_eps=getattr(args, 'freq_bias_eps', 1e-12),
        textual_only=getattr(args, 'penet_textual_only', False),
        visual_only=getattr(args, 'penet_visual_only', False),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
