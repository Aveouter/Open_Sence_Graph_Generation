"""
CVC: Compositionally Verified Concept Relation Head.

Reference: LCompo-SGG — Long-tail Compositional Scene Graph Generation.

Architecture:
  1. Visual Concept Branch:   z_v   = MLP_v(visual_pair_feat)           → R^h
  2. Composition Shortcut:    z_b   = MLP_b([e_s, e_o])                → R^h
  3. Composition Corrector:   delta = MLP_c([z_v, e_s, e_o])           → R^h
  4. Composition Discriminator: c_pred = MLP_d(GRL(z_v))               → comp_id

Training Loss:
  L = CE(pred_logits, y)                           (main classification)
    + α · KL(fc_v(z_v) || stopgrad(fc_b(z_b)))      (visual-bias alignment)
    + β · DiversityLoss(z_v)                         (prototype diversity)
    + γ · AdvLoss(comp_disc(z_v), comp_label)        (adversarial comp removal)

Inference:
  Standard:   logits = fc_v(z_v) - λ · fc_b(z_b)
  Novel comp: z_corrected = z_v + Corrector(z_v, Δe_s, Δe_o)
              logits = fc_v(z_corrected) - λ · fc_b(z_b)
"""
import math
import json
from pathlib import Path
from typing import Dict, Set, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Gradient Reversal Layer (GRL)
# ═══════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal for adversarial training.

    Forward:  identity
    Backward: -lambda * gradient
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL module wrapper."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


# ═══════════════════════════════════════════════════════════════
# CVC Model Components
# ═══════════════════════════════════════════════════════════════

class MLPBlock(nn.Module):
    """Two-layer MLP with ReLU and dropout."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VisualBranch(nn.Module):
    """Visual concept branch: z_v = MLP(visual_pair_feat).

    Extracts predicate-relevant visual features from the pair representation.
    """

    def __init__(self, visual_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = MLPBlock(visual_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, visual_feat: torch.Tensor) -> torch.Tensor:
        """Args: visual_feat [P, visual_dim] → [P, hidden_dim]"""
        return self.norm(self.mlp(visual_feat))


class ShortcutBranch(nn.Module):
    """Composition shortcut branch: z_b = MLP([e_s, e_o]).

    Learns the marginal P(pred | subject_class, object_class) from
    category embeddings alone, capturing the frequency prior.
    """

    def __init__(self, class_embed_dim: int, hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.mlp = MLPBlock(class_embed_dim * 2, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, e_s: torch.Tensor, e_o: torch.Tensor) -> torch.Tensor:
        """
        Args:
            e_s: [P, class_embed_dim] subject class embeddings.
            e_o: [P, class_embed_dim] object class embeddings.

        Returns:
            [P, hidden_dim] composition shortcut features.
        """
        feat = torch.cat([e_s, e_o], dim=-1)
        return self.norm(self.mlp(feat))


class CompositionCorrector(nn.Module):
    """Composition corrector: delta = MLP([z_v, Δe_s, Δe_o]).

    For novel (S,O) compositions, computes a corrective offset based on
    the difference between the current class embeddings and the nearest
    seen composition's class embeddings.
    """

    def __init__(self, hidden_dim: int, class_embed_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        in_dim = hidden_dim + class_embed_dim * 2
        self.mlp = MLPBlock(in_dim, hidden_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z_v: torch.Tensor, delta_e_s: torch.Tensor,
                delta_e_o: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_v: [P, hidden_dim] visual features.
            delta_e_s: [P, class_embed_dim] embedding offset for subject.
            delta_e_o: [P, class_embed_dim] embedding offset for object.

        Returns:
            [P, hidden_dim] corrective offset, gated by confidence.
        """
        feat = torch.cat([z_v, delta_e_s, delta_e_o], dim=-1)
        delta = self.norm(self.mlp(feat))
        gate = self.gate(feat)  # [P, 1]
        return gate * delta


class CompositionDiscriminator(nn.Module):
    """Adversarial composition discriminator.

    Tries to predict the composition ID (hash of (s_class, o_class))
    from visual features. The GRL on input reverses gradients, so the
    visual branch is trained to REMOVE composition-identifiable information.
    """

    def __init__(self, hidden_dim: int, num_compositions: int,
                 dropout: float = 0.1):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_=0.1)
        self.mlp = MLPBlock(hidden_dim, hidden_dim // 2, num_compositions, dropout)

    def forward(self, z_v: torch.Tensor) -> torch.Tensor:
        """Args: z_v [P, hidden_dim] → [P, num_compositions] comp logits."""
        return self.mlp(self.grl(z_v))


# ═══════════════════════════════════════════════════════════════
# CVC Main Module
# ═══════════════════════════════════════════════════════════════

class CVCModule(nn.Module):
    """CVC: Compositionally Verified Concept relation prediction head.

    Args:
        visual_dim: Dimension of input pair visual features.
        class_embed_dim: Dimension of class embeddings (subject/object).
        num_predicates: Number of predicate classes (including bg).
        hidden_dim: Hidden dimension for all branches.
        num_compositions: Number of unique (S,O) compositions in training data
                          (for adversarial discriminator).
        bias_lambda: λ for inference-time bias subtraction.
        dropout: Dropout probability.
    """

    def __init__(self, visual_dim: int, class_embed_dim: int,
                 num_predicates: int = 51, hidden_dim: int = 512,
                 num_compositions: int = 10068,
                 bias_lambda: float = 0.5,
                 dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_predicates = num_predicates
        self.bias_lambda = bias_lambda

        # Class embeddings (shared subject/object embedding space)
        self.class_embed = nn.Embedding(151, class_embed_dim)  # 150 classes + bg

        # Branches
        self.visual_branch = VisualBranch(visual_dim, hidden_dim, dropout)
        self.shortcut_branch = ShortcutBranch(class_embed_dim, hidden_dim, dropout)
        self.corrector = CompositionCorrector(hidden_dim, class_embed_dim, dropout)

        # Predicate classifiers
        self.fc_visual = nn.Linear(hidden_dim, num_predicates)
        self.fc_bias = nn.Linear(hidden_dim, num_predicates)

        # Composition discriminator (adversarial)
        self.has_discriminator = num_compositions > 0
        if self.has_discriminator:
            self.comp_disc = CompositionDiscriminator(hidden_dim, num_compositions, dropout)

        # Seen composition database (loaded at inference)
        self.register_buffer('_seen_db_keys', torch.zeros(0, hidden_dim))
        self.register_buffer('_seen_db_values', torch.zeros(0, class_embed_dim * 2))
        self._seen_db_loaded = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, visual_feat: torch.Tensor,
                subj_class: torch.Tensor, obj_class: torch.Tensor,
                return_all: bool = False):
        """
        Args:
            visual_feat: [P, visual_dim] pair visual features.
            subj_class: [P] subject class labels.
            obj_class: [P] object class labels.
            return_all: If True, return intermediate features for loss computation.

        Returns:
            dict with:
              - pred_logits: [P, num_predicates] final predicate logits.
              - bias_logits: [P, num_predicates] shortcut branch logits.
              - z_v: [P, hidden_dim] visual features (if return_all).
              - z_b: [P, hidden_dim] shortcut features (if return_all).
              - comp_logits: [P, num_compositions] discriminator output (if has_disc).
        """
        # Class embeddings
        e_s = self.class_embed(subj_class)  # [P, class_embed_dim]
        e_o = self.class_embed(obj_class)   # [P, class_embed_dim]

        # Visual branch
        z_v = self.visual_branch(visual_feat)  # [P, hidden_dim]

        # Shortcut branch
        z_b = self.shortcut_branch(e_s, e_o)   # [P, hidden_dim]

        # Predicate logits
        pred_logits = self.fc_visual(z_v) - self.bias_lambda * self.fc_bias(z_b)

        out = {
            "pred_logits": pred_logits,
            "bias_logits": self.fc_bias(z_b),
        }

        if return_all:
            out["z_v"] = z_v
            out["z_b"] = z_b
            if self.has_discriminator:
                out["comp_logits"] = self.comp_disc(z_v)

        return out

    # ------------------------------------------------------------------
    # Inference with composition correction
    # ------------------------------------------------------------------

    def forward_corrected(self, visual_feat: torch.Tensor,
                           subj_class: torch.Tensor, obj_class: torch.Tensor,
                           is_novel: torch.Tensor):
        """Forward pass with composition correction for novel pairs.

        Args:
            visual_feat: [P, visual_dim].
            subj_class, obj_class: [P].
            is_novel: [P] bool tensor, True for unseen compositions.

        Returns:
            Same as forward().
        """
        e_s = self.class_embed(subj_class)
        e_o = self.class_embed(obj_class)
        z_v = self.visual_branch(visual_feat)
        z_b = self.shortcut_branch(e_s, e_o)

        # For novel compositions, find nearest seen and apply correction
        if is_novel.any() and self._seen_db_loaded and self._seen_db_keys.size(0) > 0:
            z_v_novel = z_v[is_novel]
            e_s_novel = e_s[is_novel]
            e_o_novel = e_o[is_novel]

            # Find nearest seen composition by cosine similarity in z_v space
            z_v_norm = F.normalize(z_v_novel, dim=-1)
            db_norm = F.normalize(self._seen_db_keys, dim=-1)

            # Top-1 nearest neighbor
            sim = z_v_norm @ db_norm.T  # [P_novel, N_db]
            _, nn_idx = sim.max(dim=-1)

            e_seen = self._seen_db_values[nn_idx]  # [P_novel, 2*class_embed_dim]
            e_s_seen = e_seen[:, :e_s.size(-1)]
            e_o_seen = e_seen[:, e_s.size(-1):]

            delta_e_s = e_s_novel - e_s_seen
            delta_e_o = e_o_novel - e_o_seen

            delta = self.corrector(z_v_novel, delta_e_s, delta_e_o)
            z_v[is_novel] = z_v_novel + delta

        pred_logits = self.fc_visual(z_v) - self.bias_lambda * self.fc_bias(z_b)

        return {
            "pred_logits": pred_logits,
            "bias_logits": self.fc_bias(z_b),
        }

    # ------------------------------------------------------------------
    # Seen composition database
    # ------------------------------------------------------------------

    def load_seen_compositions(self, feats: torch.Tensor,
                                subj_class: torch.Tensor,
                                obj_class: torch.Tensor):
        """Load the seen composition database for inference-time correction.

        Args:
            feats: [N_seen, hidden_dim] visual features of seen compositions.
            subj_class: [N_seen] subject class labels.
            obj_class: [N_seen] object class labels.
        """
        e_s = self.class_embed(subj_class)  # [N_seen, class_embed_dim]
        e_o = self.class_embed(obj_class)   # [N_seen, class_embed_dim]

        self._seen_db_keys = F.normalize(feats, dim=-1)
        self._seen_db_values = torch.cat([e_s, e_o], dim=-1)
        self._seen_db_loaded = True


# ═══════════════════════════════════════════════════════════════
# CVC Loss Function
# ═══════════════════════════════════════════════════════════════

class CVCLoss(nn.Module):
    """Composite loss for CVC training.

    L = L_pred + α · L_align + β · L_diverse + γ · L_adv

    Where:
      L_pred:    Cross-entropy on predicate classification.
      L_align:   KL divergence between visual and shortcut predictions
                 (encourages visual branch to NOT duplicate shortcut).
      L_diverse: Negative entropy of visual prototypes (encourages diversity).
      L_adv:     Adversarial loss for composition discrimination.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.05,
                 gamma: float = 0.1, temperature: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

    def forward(self, outputs: dict, targets: list) -> dict:
        """
        Args:
            outputs: CVC forward output.
              - pred_logits: [P_total, num_pred] batched OR list of [P_i, num_pred]
              - bias_logits: same format as pred_logits
              - z_v, z_b: same format
              - pair_indices: list of [P_i, 2] tensors (one per image)
            targets: list of per-image target dicts.

        Returns:
            dict of loss components.
        """
        pred_logits_all = outputs["pred_logits"]
        bias_logits_all = outputs.get("bias_logits")
        z_v_all = outputs.get("z_v")
        pair_indices_list = outputs.get("pair_indices")

        # Normalize to list-of-tensors format
        if isinstance(pred_logits_all, torch.Tensor):
            # Batched format: slice by pair_indices
            pred_list = []
            bias_list = []
            z_v_list = []
            offset = 0
            for pi in pair_indices_list:
                P = pi.size(0) if pi is not None and pi.numel() > 0 else 0
                if P > 0:
                    pred_list.append(pred_logits_all[offset:offset + P])
                    if bias_logits_all is not None:
                        bias_list.append(bias_logits_all[offset:offset + P])
                    if z_v_all is not None:
                        z_v_list.append(z_v_all[offset:offset + P])
                else:
                    pred_list.append(None)
                    bias_list.append(None)
                    z_v_list.append(None)
                offset += P
        else:
            pred_list = pred_logits_all
            bias_list = bias_logits_all if isinstance(bias_logits_all, (list, tuple)) else [bias_logits_all] * len(targets)
            z_v_list = z_v_all if isinstance(z_v_all, (list, tuple)) else ([z_v_all] * len(targets) if z_v_all is not None else None)

        if bias_list is None:
            bias_list = [None] * len(targets)
        if z_v_list is None:
            z_v_list = [None] * len(targets)

        device = pred_list[0].device if pred_list[0] is not None else torch.device('cpu')
        total_pred_loss = torch.tensor(0.0, device=device)
        total_align_loss = torch.tensor(0.0, device=device)
        total_diverse_loss = torch.tensor(0.0, device=device)
        total_pairs = 0

        for i, t in enumerate(targets):
            if i >= len(pred_list) or i >= len(pair_indices_list):
                break

            pred_logits = pred_list[i]
            bias_logits = bias_list[i] if i < len(bias_list) else None
            z_v = z_v_list[i] if i < len(z_v_list) else None
            pi = pair_indices_list[i]

            if pred_logits is None or pi is None:
                continue

            # Squeeze batch dim if present
            while pred_logits.dim() > 2:
                pred_logits = pred_logits.squeeze(0)
            while pi.dim() > 2:
                pi = pi.squeeze(0)

            P = pred_logits.size(0)
            if P == 0:
                continue

            rel_anns = t.get("rel_annotations")
            if rel_anns is None or rel_anns.numel() == 0:
                continue

            # Build GT labels
            gt_preds = torch.full((P,), -1, dtype=torch.long, device=device)
            pair_to_pred = {}
            for ann in rel_anns:
                s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                pair_to_pred[(s, o)] = p

            for p_idx in range(P):
                s = int(pi[p_idx, 0])
                o = int(pi[p_idx, 1])
                if (s, o) in pair_to_pred:
                    gt_preds[p_idx] = pair_to_pred[(s, o)]

            valid = gt_preds >= 0
            if not valid.any():
                continue

            total_pred_loss += self.ce_loss(pred_logits[valid], gt_preds[valid])
            total_pairs += valid.sum()

            # Alignment loss (KL)
            if bias_logits is not None:
                while bias_logits.dim() > 2:
                    bias_logits = bias_logits.squeeze(0)
                p_v = F.log_softmax(pred_logits[valid] / self.temperature, dim=-1)
                p_b = F.softmax(bias_logits[valid].detach() / self.temperature, dim=-1)
                total_align_loss += F.kl_div(p_v, p_b, reduction='sum')

            # Diversity loss
            if z_v is not None and valid.sum() > 1:
                z_valid = z_v[valid]
                z_norm = F.normalize(z_valid, dim=-1)
                sim = z_norm @ z_norm.T
                mask = ~torch.eye(z_valid.size(0), dtype=torch.bool, device=device)
                total_diverse_loss += sim[mask].clamp(min=0).mean()

        # Compute totals
        n_pairs = max(total_pairs, 1)
        loss_dict = {"loss_predicate": total_pred_loss / n_pairs}
        total_loss = loss_dict["loss_predicate"]

        if total_align_loss > 0 and self.alpha > 0:
            loss_dict["loss_align"] = self.alpha * total_align_loss / n_pairs
            total_loss = total_loss + loss_dict["loss_align"]

        if total_diverse_loss > 0 and self.beta > 0:
            loss_dict["loss_diverse"] = self.beta * total_diverse_loss
            total_loss = total_loss + loss_dict["loss_diverse"]

        loss_dict["loss_total"] = total_loss
        return loss_dict


# ═══════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════

def build_cvc(args) -> CVCModule:
    """Build CVC module from config args.

    Note: visual_dim should match the pair feature dimension from the backbone
    (typically hidden_dim, since PairFeatureGenerator projects to hidden_dim).
    """
    hdim = getattr(args, 'hidden_dim', 512)
    model = CVCModule(
        visual_dim=getattr(args, 'cvc_visual_dim', hdim),
        class_embed_dim=getattr(args, 'class_embed_dim', 256),
        num_predicates=getattr(args, 'rel_nums', 51),
        hidden_dim=hdim,
        num_compositions=getattr(args, 'cvc_num_compositions', 10068),
        bias_lambda=getattr(args, 'cvc_bias_lambda', 0.5),
        dropout=getattr(args, 'dropout', 0.1),
    )
    return model
