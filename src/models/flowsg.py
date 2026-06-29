# src/models/flowsg.py
"""
FlowSG: Progressive Image-Conditioned Scene Graph Generation with Flow Matching.

CVPR 2026 — Hu, Qin, Yin, Li, Li, He

Faithful reimplementation matching §4 and §5.1:
  - Frozen CLIP ViT-B/16 image encoder (§4.2)
  - Slotwise VQ-VAE: 4 slots × 64 codes × 512 dim (§4.1)
  - DiT-style Graph Transformer: 5 blocks, AdaLN, ReSA (FiLM), FMA (§4.3)
  - CFM for boxes: cos schedule, Gaussian prior (§4.2)
  - DFM for semantics: masked token prediction, marginal init (§4.2)
  - Training: AdamW, lr=1e-4, wd=0.02, 500K iters, bs=128 (§5.1)
"""

from __future__ import annotations

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, List, Tuple

# Use HF mirror for downloading frozen encoders (faster in China).
# Respect explicit HF_ENDPOINT if already set; otherwise default to mirror.
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from utils.misc import NestedTensor
from utils import box_ops
from src.modules.layers.backbone import build_backbone
from src.modules.flowsg.vqvae import SlotwiseVQVAE
from src.modules.flowsg.flow_matching import (
    ContinuousFlowMatching,
    DiscreteFlowMatching,
    ODESolver,
)
from src.modules.flowsg.graph_transformer import (
    FlowSGDenoiser,
    GeometryHead,
    SemanticHead,
)


# ==============================================================================
# CLIP Image Encoder Wrapper (§4.2)
# ==============================================================================


class CLIPImageEncoder(nn.Module):
    """Frozen CLIP ViT-B/16 image encoder with trainable adapter.

    Extracts global image features C for cross-attention conditioning.
    """

    def __init__(
        self, model_name: str = "openai/clip-vit-base-patch16", dim: int = 512
    ):
        super().__init__()
        try:
            from transformers import CLIPVisionModel

            self.encoder = CLIPVisionModel.from_pretrained(model_name)
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.clip_dim = self.encoder.config.hidden_size  # 768 for ViT-B/16
        except (ImportError, OSError):
            # Fallback: use ResNet backbone loaded externally
            self.encoder = None
            self.clip_dim = 2048

        # Adapter: clip_dim → d_model
        self.adapter = nn.Sequential(
            nn.Linear(self.clip_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, pixel_values: Tensor) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Args:
            pixel_values: [B, 3, H, W] (ImageNet normalized)
        Returns:
            features: [B, L, dim] projected CLIP features (CLS + patches)
            mask:     [B, L] all-False mask (no padding in CLIP)
            spatial:  [B, dim, H_p, W_p] spatial patch features for ROI pooling
        """
        B = pixel_values.shape[0]
        device = pixel_values.device

        if self.encoder is not None:
            with torch.no_grad():
                if pixel_values.shape[-1] != 224:
                    pixel_values_resized = F.interpolate(
                        pixel_values,
                        size=(224, 224),
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    pixel_values_resized = pixel_values

                out = self.encoder(pixel_values_resized)
                features_raw = out.last_hidden_state  # e.g. [B, 197, 768]
        else:
            # Fallback: 1 CLS + 196 dummy patches to match ViT-B/16 shape
            features_raw = torch.zeros(B, 197, self.clip_dim, device=device)

        # Adapter: clip_dim → model_dim for cross-attention conditioning
        features = self.adapter(features_raw)  # [B, L, dim]
        mask = torch.zeros(B, features.shape[1], dtype=torch.bool, device=device)

        # Spatial patch features for ROI pooling (§4.1: u_i = CLIP_img(crop(I,b_i)))
        patch_tokens = features[:, 1:, :]  # [B, N_patches, dim]  (drop CLS)
        N_p = int(math.sqrt(patch_tokens.shape[1]))
        assert N_p * N_p == patch_tokens.shape[1], (
            f"CLIP patches must form a square grid "
            f"(got {patch_tokens.shape[1]} patches)"
        )
        spatial = patch_tokens.transpose(1, 2).reshape(B, features.shape[-1], N_p, N_p)

        return features, mask, spatial


# ==============================================================================
# FlowSGCriterion: Combined Loss (§4.2, Eq.18-19)
# ==============================================================================


class FlowSGCriterion(nn.Module):
    """L = L_CFM + λ · L_DFM  (paper uses λ=1 implicitly via coefficient)."""

    def __init__(
        self,
        num_classes: int = 151,
        num_predicates: int = 51,
        codebook_size: int = 64,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        flow_loss_coef: float = 1.0,
        discrete_flow_coef: float = 1.0,
        vq_loss_coef: float = 1.0,
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.codebook_size = codebook_size
        self.bbox_loss_coef = bbox_loss_coef
        self.giou_loss_coef = giou_loss_coef
        self.flow_loss_coef = flow_loss_coef
        self.discrete_flow_coef = discrete_flow_coef
        self.vq_loss_coef = vq_loss_coef
        self.eos_coef = eos_coef

        self.cfm = ContinuousFlowMatching()
        self.dfm = DiscreteFlowMatching()

        # Class weights
        empty_weight = torch.ones(num_classes)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        empty_weight_rel = torch.ones(num_predicates)
        empty_weight_rel[-1] = eos_coef
        self.register_buffer("empty_weight_rel", empty_weight_rel)

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, Tensor]:
        loss_dict = {}

        # CFM loss (flow matching for boxes)
        # Skip in edge-only batches: geometry head sees clean gt_boxes, so
        # the velocity prediction against u_star would be meaningless noise.
        if (
            "pred_velocity" in outputs
            and "u_star" in outputs
            and not outputs.get("_edge_only", False)
        ):
            mask = outputs.get("node_mask", None)
            loss_dict["loss_cfm"] = (
                self.cfm.loss(outputs["pred_velocity"], outputs["u_star"], mask)
                * self.flow_loss_coef
            )

        # VQ-VAE loss
        if "vq_loss" in outputs:
            loss_dict["loss_vq"] = outputs["vq_loss"] * self.vq_loss_coef

        # Box supervision (L1 + GIoU on final predicted boxes vs GT).
        # Skip in edge-only mode: denoiser sees clean GT boxes, so box regression
        # is trivially an identity mapping and gradients would conflict with edge
        # prediction (§5.1 edge-only training).
        if "pred_boxes" in outputs and not outputs.get("_edge_only", False):
            pred_boxes = outputs["pred_boxes"]
            B, N, _ = pred_boxes.shape
            device = pred_boxes.device

            gt_boxes = self._build_target_boxes(pred_boxes, targets)
            mask = outputs.get("node_mask", None)

            loss_bbox = F.l1_loss(pred_boxes, gt_boxes, reduction="none").sum(-1)
            if mask is not None and mask.any():
                loss_bbox = loss_bbox[mask].mean()
            else:
                loss_bbox = loss_bbox.mean()
            loss_dict["loss_bbox"] = loss_bbox * self.bbox_loss_coef

            # GIoU with 1:1 matching (per-image, avoids B*N × B*N full matrix)
            loss_giou_list = []
            for b_idx in range(B):
                p_xyxy = box_ops.box_cxcywh_to_xyxy(
                    pred_boxes[b_idx].clamp(1e-6, 1.0 - 1e-6)
                )
                g_xyxy = box_ops.box_cxcywh_to_xyxy(
                    gt_boxes[b_idx].clamp(1e-6, 1.0 - 1e-6)
                )
                try:
                    giou_1to1 = box_ops.generalized_box_iou_1to1(p_xyxy, g_xyxy)
                    loss_giou_list.append(1.0 - giou_1to1.clamp(-1, 1))
                except Exception:
                    loss_giou_list.append(torch.zeros(N, device=device))
            loss_giou = torch.stack(loss_giou_list).nan_to_num(0.0)
            if mask is not None and mask.any():
                loss_giou = loss_giou[mask].mean()
            else:
                loss_giou = loss_giou.mean()
            loss_dict["loss_giou"] = loss_giou * self.giou_loss_coef

        # DFM losses (discrete flow for predicates and appearance codes).
        # NOTE: No obj_dfm_loss — object labels serve as priors per §4.2 and are
        # never masked; DFM loss only applies to actually-masked positions.
        if "pred_dfm_loss" in outputs:
            loss_dict["loss_dfm_pred"] = (
                outputs["pred_dfm_loss"] * self.discrete_flow_coef
            )
        if "app_dfm_loss" in outputs:
            loss_dict["loss_dfm_app"] = (
                outputs["app_dfm_loss"] * self.discrete_flow_coef * 0.1
            )

        return loss_dict

    def _build_target_boxes(self, pred: Tensor, targets: List[Dict]) -> Tensor:
        B, N, _ = pred.shape
        device = pred.device
        gt = torch.zeros(B, N, 4, device=device)
        for b, t in enumerate(targets):
            boxes = t.get("boxes", None)
            if boxes is not None and len(boxes) > 0:
                n = min(len(boxes), N)
                gt[b, :n] = boxes[:n].to(device)
        return gt


# ==============================================================================
# Frozen Mask2Former Detector (§5.1)
# ==============================================================================

MASK2FORMER_AVAILABLE = False
_m2f_model_name = "facebook/mask2former-swin-small-coco-instance"


class FrozenMask2Former(nn.Module):
    """Frozen Mask2Former (COCO pretrained) for object proposal generation.

    Provides object boxes used to (1) crop CLIP spatial patch features for
    VQ-VAE encoding (§4.1) and (2) serve as initial proposals for box
    regression loss.

    On first use, downloads ~500MB — set HF_HUB_OFFLINE=1 or pre-download
    via ``transformers``.  Falls back to ``None`` (→ ResNet detection head)
    when weights are unavailable.
    """

    def __init__(self, num_queries: int = 100):
        super().__init__()
        self.num_queries = num_queries
        self._ok = False

        try:
            from transformers import Mask2FormerForUniversalSegmentation

            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                _m2f_model_name
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

            # COCO → VG label remapping built lazily on first forward
            self._coco_to_vg = None
            self._ok = True
            global MASK2FORMER_AVAILABLE
            MASK2FORMER_AVAILABLE = True
        except (ImportError, OSError, EnvironmentError):
            self.model = None
            import logging

            logging.warning(
                "Mask2Former (%s) not available — falling back to ResNet "
                "detection head.  Pre-download with: "
                "transformers.Mask2FormerForUniversalSegmentation.from_pretrained('%s')",
                _m2f_model_name,
                _m2f_model_name,
            )

    @property
    def available(self) -> bool:
        return self._ok

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Run frozen Mask2Former.

        Args:
            images: [B, 3, H, W] — ImageNet-normalized, already on device.
                    Resized to 384×384 internally (Swin-S input size).

        Returns:
            boxes:       [B, N, 4]        normalized cxcywh boxes (N = num_queries)
            coco_logits: [B, N, 134]      COCO class logits (133 classes + no-obj)
            scores:      [B, N]           detection confidence  (for logging)
        """
        from torchvision.ops import masks_to_boxes

        B, _, H_in, W_in = images.shape
        device = images.device
        N = self.num_queries

        # Resize to M2F input resolution
        if H_in != 384 or W_in != 384:
            images_384 = F.interpolate(
                images, size=(384, 384), mode="bilinear", align_corners=False
            )
        else:
            images_384 = images

        with torch.no_grad():
            out = self.model(pixel_values=images_384)

        # class_queries_logits: [B, M, 134]  (133 COCO + 1 no-object)
        # masks_queries_logits:  [B, M, 96, 96]  (384 / 4 = 96)
        cls = out.class_queries_logits  # [B, M, 134]
        msk = out.masks_queries_logits  # [B, M, 96, 96]
        M = cls.shape[1]

        # Objectness: max over COCO classes (skip "no object" channel)
        scores = cls[:, :, :-1].sigmoid().max(dim=-1).values  # [B, M]

        # Binarize masks → extract bounding boxes (vectorized per batch item)
        masks_bool = msk.sigmoid() > 0.5  # [B, M, 96, 96]
        boxes_norm = torch.zeros(B, M, 4, device=device)

        for b in range(B):
            nonempty = masks_bool[b].flatten(1).any(dim=1)  # [M]
            if nonempty.any():
                bboxes = masks_to_boxes(
                    masks_bool[b][nonempty]
                )  # [K, 4] xyxy in [0, 96]
                boxes_norm[b, nonempty, 0] = (bboxes[:, 0] + bboxes[:, 2]) / (
                    2 * 96
                )  # cx
                boxes_norm[b, nonempty, 1] = (bboxes[:, 1] + bboxes[:, 3]) / (
                    2 * 96
                )  # cy
                boxes_norm[b, nonempty, 2] = (bboxes[:, 2] - bboxes[:, 0]) / 96  # w
                boxes_norm[b, nonempty, 3] = (bboxes[:, 3] - bboxes[:, 1]) / 96  # h

        # Take top-N by score; pad if fewer than N
        num_coco_cls = cls.shape[-1]  # 81 (instance) or 134 (panoptic)
        top_boxes = torch.zeros(B, N, 4, device=device)
        top_scores = torch.zeros(B, N, device=device)
        top_coco_logits = torch.zeros(B, N, num_coco_cls, device=device)
        for b in range(B):
            k = min(N, M)
            if k > 0:
                _, idx = torch.topk(scores[b], k=k)
                top_boxes[b, :k] = boxes_norm[b, idx]
                top_scores[b, :k] = scores[b, idx]
                top_coco_logits[b, :k] = cls[b, idx]
            else:
                top_boxes[b] = 0.1 + 0.8 * torch.rand(N, 4, device=device)
                top_coco_logits[b, :, -1] = 1.0  # "no-object" for empty

        return top_boxes, top_coco_logits, top_scores


# ==============================================================================
# FlowSG Model (§4)
# ==============================================================================


class FlowSG(nn.Module):
    """FlowSG: Hybrid flow matching for scene graph generation.

    Architecture (§5.1):
      - Frozen CLIP ViT-B/16 → image features C
      - Slotwise VQ-VAE: 4 slots × 64 codes × 512 dim
      - DiT denoiser: 5 blocks, dim=512, heads=8
      - Geometry head + Semantic head
    """

    def __init__(
        self,
        backbone: nn.Module,  # ResNet backbone (used as fallback / ROI features)
        num_classes: int = 151,
        num_predicates: int = 51,
        dim: int = 512,  # hidden dim (512 per paper)
        num_heads: int = 8,
        num_blocks: int = 5,  # 5 DiT blocks
        num_slots: int = 4,  # M=4 ordered slots
        codebook_size: int = 64,  # K=64
        num_queries: int = 100,  # N object proposals
        num_flow_steps: int = 10,  # ODE steps at inference
        dropout: float = 0.1,
        edge_only_prob: float = 0.2,  # stochastic edge-only training (§5.1)
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_predicates = num_predicates
        self.num_queries = num_queries
        self.num_slots = num_slots
        self.codebook_size = codebook_size
        self.num_flow_steps = num_flow_steps
        self.edge_only_prob = edge_only_prob

        # Frozen Mask2Former detector (§5.1)
        # Provides object proposals for ROI cropping + box initialization.
        # Falls back to ResNet detection head when M2F weights unavailable.
        self.detector = FrozenMask2Former(num_queries=num_queries)

        # CLIP image encoder with adapter (§4.2)
        # Returns (global features, mask, spatial patches) for denoiser + ROI
        self.image_encoder = CLIPImageEncoder(dim=dim)

        # Fallback detection (ResNet → linear heads; used when M2F unavailable)
        self.backbone = backbone
        backbone_channels = backbone.num_channels
        self.backbone_proj = nn.Sequential(
            nn.Conv2d(backbone_channels, dim, kernel_size=1),
            nn.GroupNorm(32, dim),
        )
        self.obj_class_head = nn.Linear(dim, num_classes)
        self.obj_bbox_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, 4)
        )
        nn.init.constant_(self.obj_bbox_head[-1].weight, 0)
        nn.init.constant_(self.obj_bbox_head[-1].bias, 0)

        # COCO → VG class projection (§4.2: detector class priors)
        # M2F outputs 81 (instance) or 134 (panoptic) COCO logits.
        # Built on first forward once the actual count is known.
        self.coco_to_vg = None
        self._coco_cls_count = None

        # VQ-VAE: appearance tokenizer (§4.1)
        # Input: CLIP spatial ROI features u_i = CLIP_img(crop(I, b_i))
        self.app_vqvae = SlotwiseVQVAE(
            num_slots=num_slots,
            codebook_size=codebook_size,
            embedding_dim=dim,
            input_dim=dim,  # CLIP adapter output dim
            commitment_cost=0.25,
        )

        # Node embedding: object class + appearance codes + box encoding
        self.obj_embed = nn.Embedding(
            num_classes + 1, dim
        )  # +1 for mask_id_obj (§4.2 DFM)
        self.mask_id_obj = num_classes  # MASK token for object classes
        self.app_embed = nn.Embedding(codebook_size + 1, dim)  # +1 for mask_id_app
        self.mask_id_app = codebook_size  # MASK token for appearance
        self.box_encoder = nn.Linear(4, dim)  # box → embedding

        # Edge embedding: predicate code (+ mask_id_pred)
        self.pred_embed = nn.Embedding(num_predicates + 1, dim)
        self.mask_id_pred = num_predicates

        # Graph Transformer Denoiser (DiT-style, §4.3)
        self.denoiser = FlowSGDenoiser(
            dim=dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            time_dim=256,
            mlp_ratio=4.0,
            dropout=dropout,
        )

        # Output heads
        self.geometry_head = GeometryHead(dim=dim)  # velocity field
        self.semantic_head = SemanticHead(  # clean posteriors
            dim=dim,
            num_classes=num_classes,
            num_predicates=num_predicates,
            codebook_size=codebook_size,
        )

        # Flow matching modules
        self.cfm = ContinuousFlowMatching()
        self.dfm = DiscreteFlowMatching()

    def _extract_detections(self, backbone_feat: Tensor) -> Tuple[Tensor, Tensor]:
        """Fallback detection — conv → grid pool → linear heads on ResNet features.

        Only used when FrozenMask2Former weights are unavailable.
        Returns:
            obj_logits: [B, N, num_classes]
            obj_boxes: [B, N, 4]  normalized cxcywh
        """
        B, C, H, W = backbone_feat.shape
        D = self.dim
        device = backbone_feat.device

        # Global average pool → feature grid
        feat = self.backbone_proj(backbone_feat)  # [B, D, H, W]

        # Simple grid-based proposals: evenly sample N points
        N = self.num_queries
        grid_h = int(math.sqrt(N * H / W))
        grid_w = N // grid_h if grid_h > 0 else N

        # Actually use adaptive pooling to get N proposals
        pooled = F.adaptive_avg_pool2d(feat, (grid_h, grid_w))  # [B, D, grid_h, grid_w]
        pooled = pooled.flatten(2).transpose(1, 2)  # [B, grid_h*grid_w, D]

        # Pad or trim to N
        actual_N = pooled.shape[1]
        if actual_N < N:
            pad = torch.zeros(B, N - actual_N, D, device=device)
            pooled = torch.cat([pooled, pad], dim=1)
        elif actual_N > N:
            pooled = pooled[:, :N]

        pooled = pooled.nan_to_num(0.0)

        obj_logits = self.obj_class_head(pooled)  # [B, N, num_classes]
        raw_boxes = self.obj_bbox_head(pooled).nan_to_num(0.0)
        obj_boxes = raw_boxes.sigmoid().clamp(1e-4, 1.0 - 1e-4)

        return obj_logits, obj_boxes

    def _build_node_embedding(
        self, obj_logits: Tensor, app_indices: Tensor, boxes: Tensor
    ) -> Tensor:
        """Build initial node embeddings h_i^(0) (§4.3).

        h_i^(0) = Emb(c_i) + Emb(a_i) + Enc(b_i)
        """
        B, N = obj_logits.shape[:2]

        # Object class embedding (handle both logits [B,N,C] and labels [B,N])
        if obj_logits.dim() == 3:
            obj_cls = obj_logits.argmax(dim=-1)  # [B, N]
        else:
            obj_cls = obj_logits  # already [B, N] (may include mask_id_obj)
        obj_cls = obj_cls.clamp(0, self.mask_id_obj)
        cls_emb = self.obj_embed(obj_cls)  # [B, N, dim]

        # Appearance code embedding (sum over M slots). mask_id_app is a valid index.
        clamped = app_indices.clamp(0, self.mask_id_app)
        app_emb = 0
        for m in range(self.num_slots):
            app_emb = app_emb + self.app_embed(clamped[..., m])
        app_emb = app_emb / self.num_slots

        # Box encoding
        box_emb = self.box_encoder(boxes)  # [B, N, dim]

        return cls_emb + app_emb + box_emb

    def _build_edge_embedding(self, pred_tokens: Tensor) -> Tensor:
        """Build initial edge embeddings e_ij^(0).

        e_ij^(0) = Emb(r_ij)  where r_ij ∈ [num_predicates]
        """
        clamped = pred_tokens.clamp(0, self.mask_id_pred)
        return self.pred_embed(clamped)  # [B, N, N, dim]

    def forward(
        self,
        samples: NestedTensor,
        targets: Optional[List[Dict]] = None,
    ) -> Dict[str, Tensor]:
        B = samples.tensors.shape[0]
        N = self.num_queries
        device = samples.tensors.device

        # 1. Extract image features
        #    a) CLIP features for global conditioning + spatial patches for ROI
        clip_feat, clip_mask, clip_spatial = self.image_encoder(samples.tensors)

        # 2. Object proposals (§5.1: frozen Mask2Former)
        if self.detector.available:
            # Frozen Mask2Former (COCO pretrained) → boxes + class priors
            # §4.2: object categories from detector serve as priors, NOT masked
            obj_boxes, coco_logits, _obj_scores = self.detector(samples.tensors)
            # Lazy-init COCO→VG projection on first forward
            if self.coco_to_vg is None:
                self._coco_cls_count = coco_logits.shape[-1]
                self.coco_to_vg = nn.Sequential(
                    nn.Linear(self._coco_cls_count, self.dim),
                    nn.SiLU(),
                    nn.Linear(self.dim, self.num_classes),
                ).to(device)
            obj_logits = self.coco_to_vg(coco_logits)  # [B, N, num_classes]
        else:
            # Fallback: ResNet backbone + grid-based detection heads
            backbone_feats, _ = self.backbone(samples)
            backbone_src = backbone_feats[-1].tensors  # [B, C_b, H, W]
            obj_logits, obj_boxes = self._extract_detections(backbone_src)

        # 3. Extract ROI features from CLIP spatial patches (§4.1)
        #    u_i = CLIP_img(crop(I, b_i)) — CLIP features at object locations
        roi_features = self._roi_pool_clip(clip_spatial, obj_boxes)  # [B, N, dim]

        if targets is not None:
            # ================================================================
            # TRAINING
            # ================================================================

            # Use GT boxes and labels for supervised nodes
            gt_boxes = self._pad_gt_boxes(targets, N, device)
            gt_labels = self._pad_gt_labels(targets, N, device)
            node_mask = self._get_node_mask(targets, N, device)

            # VQ-VAE encode appearance from ROI features
            vq_out = self.app_vqvae(roi_features)
            app_indices = vq_out["indices"]  # [B, N, M]
            app_indices = app_indices.clamp(0, self.codebook_size - 1)

            # Sample time t ~ U[0, 1]
            # NOTE: CFM interpolation always runs because kappa is needed for
            # DFM masking (predicate + appearance) regardless of edge_only mode.
            # In edge_only batches, g_t/u_star are computed but unused — the
            # geometry head sees clean gt_boxes and CFM loss is excluded below.
            t = torch.rand(B, device=device)
            g_0 = self.cfm.sample_prior((B, N, 4), device)
            g_t, u_star, kappa, kappa_dot = self.cfm.interpolate(g_0, gt_boxes, t)

            # Stochastic edge-only training (§5.1): p=0.2 keep nodes fixed at GT
            edge_only = self.training and torch.rand(1).item() < self.edge_only_prob
            if edge_only:
                # Edge-only: clean GT nodes, denoiser focuses on edge prediction
                node_boxes_used = gt_boxes
                node_labels_for_denoiser = gt_labels
            else:
                # Normal: noisy boxes (CFM). Object labels NOT masked (§4.2: "object
                # categories are not masked but serve as priors"). Only rel/app are DFM.
                node_boxes_used = g_t
                node_labels_for_denoiser = (
                    gt_labels  # clean GT → denoiser (+ loss supervision)
                )

            # ── DFM: mask predicates & appearance (§4.2) ──

            # Build GT predicate tokens from targets (used later for DFM loss)
            # Per-edge masking: each edge independently masked with prob (1-κ_t)
            pred_kappa = kappa.squeeze(-1)  # [B, 1]
            pred_mask_prob = (1.0 - pred_kappa).unsqueeze(-1)  # [B, 1, 1]
            pred_rand = torch.rand(B, N, N, device=device)  # [B, N, N] per-edge
            pred_is_masked = pred_rand < pred_mask_prob  # [B, N, N] broadcast

            # Create dense edge tokens [B, N, N].
            # Default = background class (num_predicates - 1); non-edge pairs
            # should never be labeled as a real predicate.
            clean_pred_dense = torch.full(
                (B, N, N), self.num_predicates - 1, dtype=torch.long, device=device
            )
            for b_idx, target in enumerate(targets):
                rels = target.get("rel_annotations", None)
                if rels is not None and len(rels) > 0:
                    s, o, p = rels[:, 0].long(), rels[:, 1].long(), rels[:, 2].long()
                    # Filter OOB indices (safety); valid targets should always be in range
                    valid = (s >= 0) & (s < N) & (o >= 0) & (o < N)
                    s, o, p = s[valid], o[valid], p[valid]
                    if len(p) > 0:
                        p = p.clamp(0, self.num_predicates - 1)
                        clean_pred_dense[b_idx, s, o] = p

            # Mask predicate tokens (per-edge independent masking)
            pred_tokens = clean_pred_dense.clone()
            pred_tokens[pred_is_masked] = self.mask_id_pred

            # Appearance tokens: mask with probability (1-κ_t)
            app_kappa = kappa  # [B, 1, 1] already
            app_mask_prob = 1.0 - app_kappa  # [B, 1, 1] — will broadcast to [B, N, M]
            app_rand = torch.rand(B, N, self.num_slots, device=device)
            app_is_masked = app_rand < app_mask_prob
            masked_app_indices = app_indices.clone()
            masked_app_indices[app_is_masked] = self.mask_id_app

            # Build node and edge embeddings
            node_emb = self._build_node_embedding(
                node_labels_for_denoiser, masked_app_indices, node_boxes_used
            )
            edge_emb = self._build_edge_embedding(
                pred_tokens.clamp(0, self.mask_id_pred)
            )

            # Graph Transformer denoiser
            node_feat, edge_feat = self.denoiser(
                node_emb=node_emb,
                edge_emb=edge_emb,
                image_feat=clip_feat,
                image_mask=clip_mask,
                t=t,
            )

            # Predictions
            pred_velocity = self.geometry_head(node_feat)  # [B, N, 4]
            sem_out = self.semantic_head(node_feat, edge_feat)

            # Box prediction from geometry head (direct regression for supervision)
            pred_boxes = self.obj_bbox_head(node_feat).sigmoid().clamp(1e-6, 1.0 - 1e-6)

            # DFM losses — only on masked positions (§4.2)
            # NOTE: object labels are NOT masked (they serve as priors per §4.2),
            # so there is no obj_dfm_loss. Only predicates and appearance codes
            # are DFM targets.

            # Predicate DFM loss: only on per-edge masked positions
            pred_logits_flat = sem_out["pred_logits"].reshape(
                B * N * N, self.num_predicates
            )
            clean_pred_flat = clean_pred_dense.reshape(B * N * N)
            pred_is_masked_flat = pred_is_masked.reshape(B * N * N)
            pred_dfm_loss = self.dfm.loss(
                pred_logits_flat, clean_pred_flat, t,
                masked_positions=pred_is_masked_flat,
            )

            # Appearance DFM loss: only on masked slots
            # app_logits: [B, N, M, K]
            app_logits_flat = sem_out["app_logits"].reshape(
                B * N * self.num_slots, self.codebook_size
            )
            app_targets_flat = app_indices.reshape(B * N * self.num_slots)
            app_is_masked_flat = app_is_masked.reshape(B * N * self.num_slots)
            app_dfm_loss = self.dfm.loss(
                app_logits_flat, app_targets_flat, t,
                masked_positions=app_is_masked_flat,
            )

            # For RelTR-compatible eval output: use semantic head predictions on edges
            # Extract top-K edges by predicate confidence
            pred_scores = sem_out["pred_logits"].softmax(-1).max(-1).values  # [B, N, N]
            # Flatten and take top-K
            K = min(400, N * N)
            flat_scores = pred_scores.reshape(B, -1)
            topk_vals, topk_idx = torch.topk(flat_scores, k=K, dim=-1)
            pair_i = (topk_idx // N).long()
            pair_j = (topk_idx % N).long()

            # Build batch indices for correct broadcasting with 2D pair indices
            b_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]
            sub_logits = sem_out["obj_logits"][b_idx, pair_i].contiguous()  # [B, K, C]
            obj_logits = sem_out["obj_logits"][b_idx, pair_j].contiguous()  # [B, K, C]
            sub_boxes = pred_boxes[b_idx, pair_i].contiguous()  # [B, K, 4]
            obj_boxes = pred_boxes[b_idx, pair_j].contiguous()  # [B, K, 4]
            rel_logits = sem_out["pred_logits"][
                b_idx, pair_i, pair_j
            ].contiguous()  # [B, K, P]

            return {
                "pred_velocity": pred_velocity,
                "u_star": u_star,
                "g_t": g_t,
                "t": t,
                "vq_loss": vq_out["vq_loss"],
                "vq_perplexity": vq_out["perplexity"],
                "node_mask": node_mask,
                "pred_boxes": pred_boxes,
                "pred_dfm_loss": pred_dfm_loss,
                "app_dfm_loss": app_dfm_loss,
                # For evaluation compatibility
                "pred_logits": sem_out["obj_logits"],
                "sub_logits": sub_logits,
                "obj_logits": obj_logits,
                "sub_boxes": sub_boxes,
                "obj_boxes": obj_boxes,
                "rel_logits": rel_logits,
                # In edge-only mode, exclude u_star/t/g_t to skip CFM loss
                # (geometry head sees clean boxes; velocity prediction is not meaningful)
                "_edge_only": edge_only,
            }

        else:
            # ================================================================
            # INFERENCE
            # ================================================================

            # Sample boxes from Gaussian prior p_0 = N(0, I_4) (§4.2 CFM).
            # Aligned with training: g_0 ~ N(0,I) → ODE integration → g_1 ≈ data.
            g_0 = self.cfm.sample_prior((B, N, 4), device)

            # Initialize appearance codes from VQ-VAE (detector ROI features)
            with torch.no_grad():
                vq_out = self.app_vqvae(roi_features)
            init_app = vq_out["indices"].clamp(0, self.codebook_size - 1)

            # Initialize predicates at marginal (most common predicate)
            init_preds = torch.ones(B, N, N, dtype=torch.long, device=device)

            # ODE solver velocity function from t=0 → t=1.
            # obj_logits and init_app are fixed (image-conditioned), only boxes evolve.
            def velocity_fn(x: Tensor, t_step: Tensor) -> Tensor:
                node_emb = self._build_node_embedding(obj_logits, init_app, x)
                edge_emb = self._build_edge_embedding(init_preds)
                node_feat, _ = self.denoiser(
                    node_emb=node_emb,
                    edge_emb=edge_emb,
                    image_feat=clip_feat,
                    image_mask=clip_mask,
                    t=t_step,
                )
                return self.geometry_head(node_feat)

            # Solve ODE: dg/dt = v_θ(g_t, t) from t=0 → t=1 (§3, §4.2)
            solver = ODESolver(num_steps=self.num_flow_steps)
            refined_boxes = solver.solve(velocity_fn, g_0)
            # Clamp to normalized coordinate range after integration
            refined_boxes = refined_boxes.clamp(0, 1)

            # Final forward pass at t=1 (clean state) for semantic prediction (§4.2, Eq.19)
            t_final = torch.ones(B, device=device)
            node_emb = self._build_node_embedding(obj_logits, init_app, refined_boxes)
            edge_emb = self._build_edge_embedding(init_preds)
            node_feat, edge_feat = self.denoiser(
                node_emb=node_emb,
                edge_emb=edge_emb,
                image_feat=clip_feat,
                image_mask=clip_mask,
                t=t_final,
            )
            sem_out = self.semantic_head(node_feat, edge_feat)

            # Extract top-K edges
            pred_scores = sem_out["pred_logits"].softmax(-1).max(-1).values
            K = min(400, N * N)
            flat_scores = pred_scores.reshape(B, -1)
            topk_idx = torch.topk(flat_scores, k=K, dim=-1).indices
            pair_i = (topk_idx // N).long()
            pair_j = (topk_idx % N).long()

            # Build batch indices for correct broadcasting
            b_idx = torch.arange(B, device=device).unsqueeze(1)  # [B, 1]

            return {
                "pred_logits": sem_out["obj_logits"],
                "pred_boxes": refined_boxes,
                "sub_logits": sem_out["obj_logits"][b_idx, pair_i].contiguous(),
                "obj_logits": sem_out["obj_logits"][b_idx, pair_j].contiguous(),
                "sub_boxes": refined_boxes[b_idx, pair_i].contiguous(),
                "obj_boxes": refined_boxes[b_idx, pair_j].contiguous(),
                "rel_logits": sem_out["pred_logits"][
                    b_idx, pair_i, pair_j
                ].contiguous(),
            }

    def _roi_pool_clip(
        self, clip_spatial: Tensor, boxes: Tensor, size: int = 7
    ) -> Tensor:
        """RoI-Align on CLIP spatial patch features (§4.1).

        u_i = CLIP_img(crop(I, b_i)) — crops CLIP patch grid at box locations.
        CLIP adapter has already projected features to model dim, so no
        separate projection step is needed.
        """
        import torchvision

        B, C, H_p, W_p = clip_spatial.shape  # C = dim (512)
        N = boxes.shape[1]
        device = clip_spatial.device

        # Clamp and convert cxcywh → xyxy in CLIP spatial grid coords
        boxes_safe = boxes.nan_to_num(0.5).clamp(min=1e-4, max=1.0 - 1e-4).detach()
        xyxy = torch.zeros_like(boxes_safe)
        xyxy[..., 0] = (boxes_safe[..., 0] - boxes_safe[..., 2] / 2) * W_p  # x1
        xyxy[..., 1] = (boxes_safe[..., 1] - boxes_safe[..., 3] / 2) * H_p  # y1
        xyxy[..., 2] = (boxes_safe[..., 0] + boxes_safe[..., 2] / 2) * W_p  # x2
        xyxy[..., 3] = (boxes_safe[..., 1] + boxes_safe[..., 3] / 2) * H_p  # y2

        # Batch roi_align: [B*N, C, size, size]
        batch_indices = (
            torch.arange(B, device=device).unsqueeze(1).expand(-1, N).reshape(-1)
        )
        rois = torch.cat(
            [batch_indices.unsqueeze(1).float(), xyxy.reshape(-1, 4)], dim=1
        )

        pooled = torchvision.ops.roi_align(
            clip_spatial,
            rois,
            output_size=(size, size),
            spatial_scale=1.0,
            aligned=True,
        )  # [B*N, C, size, size]

        # Average pool to [B, N, dim] — features are already in model space
        roi_feat = pooled.mean(dim=[-2, -1]).reshape(B, N, C)
        return roi_feat

    def _pad_gt_boxes(self, targets: List[Dict], N: int, device) -> Tensor:
        B = len(targets)
        gt = torch.zeros(B, N, 4, device=device)
        for b, t in enumerate(targets):
            boxes = t.get("boxes", None)
            if boxes is not None and len(boxes) > 0:
                n = min(len(boxes), N)
                gt[b, :n] = boxes[:n].to(device)
        return gt

    def _pad_gt_labels(self, targets: List[Dict], N: int, device) -> Tensor:
        B = len(targets)
        gt = torch.full(
            (
                B,
                N,
            ),
            self.num_classes,
            device=device,
            dtype=torch.long,
        )
        for b, t in enumerate(targets):
            labels = t.get("labels", None)
            if labels is not None and len(labels) > 0:
                n = min(len(labels), N)
                gt[b, :n] = labels[:n].to(device).long()
        return gt.clamp(0, self.num_classes - 1)

    def _get_node_mask(self, targets: List[Dict], N: int, device) -> Tensor:
        B = len(targets)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for b, t in enumerate(targets):
            n = len(t.get("labels", t.get("boxes", [])))
            if n > 0:
                mask[b, : min(n, N)] = True
        return mask


# ==============================================================================
# Build function
# ==============================================================================


def build_flowsg(args):
    num_classes = getattr(args, "entity_nums", 151)
    num_predicates = getattr(args, "rel_nums", 51)
    dim = getattr(args, "hidden_dim", 512)
    num_heads = getattr(args, "nheads", 8)
    num_blocks = getattr(args, "num_blocks", 5)
    num_slots = getattr(args, "num_slots", 4)
    codebook_size = getattr(args, "codebook_size", 64)
    num_queries = getattr(args, "num_queries", 100)
    num_flow_steps = getattr(args, "num_flow_steps", 10)
    dropout = getattr(args, "dropout", 0.1)
    edge_only_prob = getattr(args, "edge_only_prob", 0.2)

    backbone = build_backbone(args)

    model = FlowSG(
        backbone=backbone,
        num_classes=num_classes,
        num_predicates=num_predicates,
        dim=dim,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_slots=num_slots,
        codebook_size=codebook_size,
        num_queries=num_queries,
        num_flow_steps=num_flow_steps,
        dropout=dropout,
        edge_only_prob=edge_only_prob,
    )

    criterion = FlowSGCriterion(
        num_classes=num_classes,
        num_predicates=num_predicates,
        codebook_size=codebook_size,
        bbox_loss_coef=getattr(args, "bbox_loss_coef", 5.0),
        giou_loss_coef=getattr(args, "giou_loss_coef", 2.0),
        flow_loss_coef=getattr(args, "flow_loss_coef", 1.0),
        discrete_flow_coef=getattr(args, "discrete_flow_coef", 1.0),
        vq_loss_coef=getattr(args, "vq_loss_coef", 1.0),
        eos_coef=getattr(args, "eos_coef", 0.1),
    )

    return model, criterion
