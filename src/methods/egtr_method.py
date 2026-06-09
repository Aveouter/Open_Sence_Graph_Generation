# src/methods/egtr_method.py
"""
EGTR method wrapper — integrates the EGTR model into the OpenSGG framework.

EGTR: Extracting Graph from Transformer for Scene Graph Generation (CVPR 2024)
Uses Deformable DETR backbone + lightweight relation extraction head.
Depends on HuggingFace transformers (DeformableDetrConfig, DeformableDetrFeatureExtractor).
"""

import torch
import pickle
import torch.nn as nn
from .base_method import Base_method

from utils.misc import NestedTensor, nested_tensor_from_tensor_list


def build_egtr(args):
    """
    Build EGTR model from config and pretrained checkpoint.

    Args:
        args: Namespace or SimpleNamespace with:
            - egtr_pretrained_path: path to pretrained DeformableDETR checkpoint
            - architecture: HuggingFace model name (default: 'SenseTime/deformable-detr')
            - entity_nums: number of object classes (including bg, e.g. 151)
            - rel_nums: number of relation classes (including bg, e.g. 51)
            - num_queries: number of object queries (default: 200)
            - use_freq_bias: whether to use frequency bias
            - logit_adjustment: whether to apply logit adjustment at inference
            - logit_adj_tau: tau for logit adjustment
            - freq_bias_eps: epsilon for frequency bias
            - use_log_softmax: whether to use log_softmax for freq bias
            - rel_loss_coef: relation loss coefficient
            - connectivity_loss_coef: connectivity loss coefficient
            - bbox_loss_coef: bbox loss coefficient
            - giou_loss_coef: giou loss coefficient
            - ce_loss_coef: cross-entropy loss coefficient
            - eos_coef: eos coefficient for no-object class
            - focal_alpha: focal loss alpha
            - rel_sample_negatives: number of negative samples per positive relation
            - rel_sample_nonmatching: number of non-matching samples
            - smoothing: label smoothing for matcher
            - aux_loss: whether to use auxiliary losses
            - dec_layers: number of decoder layers
            - device: device string
    """
    from src.modules.egtr.deformable_detr import DeformableDetrConfig
    from src.modules.egtr.deformable_detr import (
        DeformableDetrFeatureExtractor,
        DeformableDetrFeatureExtractorWithAugmentorNoCrop,
    )
    from src.modules.egtr.egtr import DetrForSceneGraphGeneration

    device = torch.device(getattr(args, 'device', 'cuda'))
    architecture = getattr(args, 'architecture', 'SenseTime/deformable-detr')

    # Try to load config from pretrained, fall back to creating locally
    pretrained_path = getattr(args, 'egtr_pretrained_path', None)
    try:
        if pretrained_path is not None:
            config = DeformableDetrConfig.from_pretrained(pretrained_path)
        else:
            config = DeformableDetrConfig.from_pretrained(architecture)
    except (OSError, ValueError, EnvironmentError, pickle.UnpicklingError):
        # Offline mode: create config from scratch with defaults
        config = DeformableDetrConfig(
            backbone='resnet50',
            d_model=256,
            encoder_layers=6,
            decoder_layers=6,
            encoder_attention_heads=8,
            decoder_attention_heads=8,
            encoder_ffn_dim=1024,
            decoder_ffn_dim=1024,
            num_feature_levels=4,
            encoder_n_points=4,
            decoder_n_points=4,
            num_queries=200,
        )

    # Override config with args
    entity_n = getattr(args, 'entity_nums', 150)  # EGTR official: 150 obj classes
    rel_n = getattr(args, 'rel_nums', 50)          # EGTR official: 50 pred classes
    print(f'[Info] build_egtr: entity_nums={entity_n}, rel_nums={rel_n}')
    config.num_labels = entity_n     # num object classes
    config.num_rel_labels = rel_n     # num relation classes
    config.num_queries = getattr(args, 'num_queries', 200)
    config.use_freq_bias = getattr(args, 'use_freq_bias', True)
    config.logit_adjustment = getattr(args, 'logit_adjustment', False)
    config.logit_adj_tau = getattr(args, 'logit_adj_tau', 0.3)
    config.freq_bias_eps = getattr(args, 'freq_bias_eps', 1e-12)
    config.use_log_softmax = getattr(args, 'use_log_softmax', False)
    config.rel_loss_coefficient = getattr(args, 'rel_loss_coef', 15.0)
    config.connectivity_loss_coefficient = getattr(args, 'connectivity_loss_coef', 30.0)
    config.bbox_loss_coefficient = getattr(args, 'bbox_loss_coef', 5.0)
    config.giou_loss_coefficient = getattr(args, 'giou_loss_coef', 2.0)
    config.ce_loss_coefficient = getattr(args, 'ce_loss_coef', 2.0)
    config.eos_coefficient = getattr(args, 'eos_coef', 0.1)
    config.focal_alpha = getattr(args, 'focal_alpha', 0.25)
    config.rel_sample_negatives = getattr(args, 'rel_sample_negatives', 80)
    config.rel_sample_nonmatching = getattr(args, 'rel_sample_nonmatching', 80)
    config.rel_sample_negatives_largest = getattr(args, 'rel_sample_negatives_largest', True)
    config.rel_sample_nonmatching_largest = getattr(args, 'rel_sample_nonmatching_largest', True)
    config.smoothing = getattr(args, 'smoothing', 1e-14)
    config.auxiliary_loss = getattr(args, 'aux_loss', True)
    config.decoder_layers = getattr(args, 'dec_layers', 6)

    # Build feature extractor (for image preprocessing during data loading)
    min_size = getattr(args, 'min_size', 800)
    max_size = getattr(args, 'max_size', 1333)
    use_augment = getattr(args, 'use_augment', False)
    if use_augment:
        feature_extractor = DeformableDetrFeatureExtractorWithAugmentorNoCrop(
            size=min_size, max_size=max_size
        )
    else:
        feature_extractor = DeformableDetrFeatureExtractor(
            size=min_size, max_size=max_size
        )

    # Frequency bias matrix (computed from training data)
    fg_matrix = getattr(args, 'fg_matrix', None)

    # Build model
    try:
        model = DetrForSceneGraphGeneration.from_pretrained(
            pretrained_path if pretrained_path else architecture,
            config=config,
            ignore_mismatched_sizes=True,
            fg_matrix=fg_matrix,
        )
    except (OSError, ValueError, EnvironmentError, pickle.UnpicklingError):
        # Offline: instantiate model directly without pretrained weights
        model = DetrForSceneGraphGeneration(config, fg_matrix=fg_matrix)

    model.to(device)

    # Load custom checkpoint if provided (handles size mismatches).
    # When using BaseExperiment, ckpt loading is also done in
    # BaseExperiment.test() as the authoritative step; this is a
    # convenience path for standalone / non-Lightning usage.
    ckpt_path = getattr(args, 'ckpt_path', None)
    if ckpt_path:
        from src.exp import BaseExperiment
        state_dict = BaseExperiment._load_checkpoint_state_dict(ckpt_path)
        BaseExperiment._adapt_state_dict(state_dict, model)

    return model, feature_extractor


class EGTR_Method(Base_method):
    """
    EGTR method wrapper for PyTorch Lightning.

    Training batch convention:
        batch = (images, targets)
          images: Tensor[B, 3, H, W] or list[Tensor(3, H, W)]
          targets: list[dict], each dict contains:
            - "labels": LongTensor[num_obj] — object class labels
            - "boxes": FloatTensor[num_obj, 4] — boxes in (cx, cy, w, h), normalized
            - "rel_annotations": LongTensor[num_rel, 3] — (sub_idx, obj_idx, rel_label)
    """

    def __init__(self, **args):
        super().__init__(**args)
        self.feature_extractor = None

    def _build_criterion(self, **args):
        """EGTR criterion is built inside _build_model. Skip base class construction."""
        return None

    def _build_model(self, **args):
        model, feature_extractor = build_egtr(self.hparams)
        self.feature_extractor = feature_extractor
        # EGTR's criterion is built internally during forward when labels are provided
        self.criterion = None
        return model

    # ---------- forward / predict ----------

    def forward(self, images, targets=None, **kwargs):
        """
        Forward pass for EGTR.

        Args:
            images: Tensor or NestedTensor of batched images
            targets: list[dict] or None

        Returns:
            dict with keys: outputs, loss_dict (if targets provided), loss (if targets provided)
        """
        if isinstance(images, NestedTensor):
            pixel_values = images.tensors
            pixel_mask = images.mask
        elif isinstance(images, torch.Tensor):
            pixel_values = images
            pixel_mask = None
        else:
            samples = nested_tensor_from_tensor_list(images)
            pixel_values = samples.tensors
            pixel_mask = samples.mask

        result = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=targets,
            output_attention_states=True,
            output_hidden_states=True,
        )

        out = {
            'outputs': {
                'pred_logits': result.logits,
                'pred_boxes': result.pred_boxes,
                'pred_rel': result.pred_rel,
                'pred_connectivity': result.pred_connectivity,
            }
        }

        if targets is not None and result.loss_dict is not None:
            out['loss_dict'] = result.loss_dict
            out['loss'] = result.loss

        return out

    # ---------- training / validation / test ----------

    def training_step(self, batch, batch_idx):
        images, targets = batch
        targets = self._adapt_targets(targets)

        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        targets = self._adapt_targets(targets)

        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'val_{k}', v, on_step=False, on_epoch=True, sync_dist=True)

        outputs = result['outputs']
        self.val_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
            },
            'targets': [
                {
                    kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                    for kk, vv in t.items()
                }
                for t in targets
            ],
            'loss_dict': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            },
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss,
        })

        return total_loss

    def test_step(self, batch, batch_idx):
        images, targets = batch
        targets = self._adapt_targets(targets)

        result = self.forward(images, targets)
        loss_dict = result.get('loss_dict', {})
        total_loss = result.get('loss', torch.tensor(0.0, device=self.device))

        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'test_{k}', v, on_step=False, on_epoch=True, sync_dist=True)

        outputs = result['outputs']
        self.test_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items()
            },
            'targets': [
                {
                    kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                    for kk, vv in t.items()
                }
                for t in targets
            ],
            'loss_dict': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in loss_dict.items()
            },
            'total_loss': total_loss.detach().cpu() if torch.is_tensor(total_loss) else total_loss,
        })

        return total_loss

    # ---------- helpers ----------

    def _adapt_targets(self, targets):
        """
        Adapt target format from OpenSGG convention to EGTR convention.

        OpenSGG: dict with 'labels', 'boxes', 'rel_annotations' (sparse [N,3])
        EGTR: dict with 'class_labels', 'boxes', 'rel' (dense [num_queries, num_queries, num_rel])

        CRITICAL: rel_matrix MUST be padded to num_queries.  The EGTR loss
        function extends target indices with dummy query slots up to
        num_object_queries, so indexing a GT-sized matrix with query-sized
        indices triggers CUDA device-side assert.
        """
        # Resolve model's num_queries and num_rel_labels from the built model
        model_config = getattr(self.model, 'config', None)
        num_queries = getattr(model_config, 'num_queries', 200) if model_config is not None else 200
        num_rel_labels = getattr(model_config, 'num_rel_labels',
                                 self.hparams.rel_nums) if model_config is not None else self.hparams.rel_nums

        adapted = []
        for t in targets:
            at = {}
            # Map keys
            if 'labels' in t:
                at['class_labels'] = t['labels']
            elif 'class_labels' in t:
                at['class_labels'] = t['class_labels']

            if 'boxes' in t:
                at['boxes'] = t['boxes']

            if 'image_id' in t:
                at['image_id'] = t['image_id']

            if 'orig_size' in t:
                at['orig_size'] = t['orig_size']
            if 'size' in t:
                at['size'] = t['size']

            if 'area' in t:
                at['area'] = t['area']
            if 'iscrowd' in t:
                at['iscrowd'] = t['iscrowd']

            device = t.get('labels', t.get('class_labels')).device

            # Convert sparse rel_annotations to dense rel matrix padded to num_queries
            if 'rel_annotations' in t and 'class_labels' in at:
                rel_matrix = torch.zeros(num_queries, num_queries, num_rel_labels,
                                         device=device)
                for ann in t['rel_annotations']:
                    s, o, p = int(ann[0]), int(ann[1]), int(ann[2])
                    if 0 <= s < num_queries and 0 <= o < num_queries and 0 <= p < num_rel_labels:
                        rel_matrix[s, o, p] = 1.0
                at['rel'] = rel_matrix
            elif 'rel' in t:
                at['rel'] = t['rel']
            else:
                at['rel'] = torch.zeros(num_queries, num_queries, num_rel_labels,
                                        device=device)

            assert at['rel'].ndim == 3, \
                f"rel must be 3D, got shape {at['rel'].shape}"
            assert at['rel'].shape == (num_queries, num_queries, num_rel_labels), \
                f"rel shape mismatch: {at['rel'].shape} vs ({num_queries},{num_queries},{num_rel_labels})"

            adapted.append(at)
        return adapted
