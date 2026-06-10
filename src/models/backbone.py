"""
Visual backbone for classical SGG methods.

Provides ResNet feature extraction + ROI Align for per-object visual features.
Used by Motifs, VCTree, TDE, and CVC methods for PredCLS/SGCLS.

Pipeline:
  Image [3, H, W] → ResNet → feature_map [C, H/32, W/32]
  GT boxes (cxcywh norm) → ROI Align → [N, C, 7, 7]
  Adaptive pool → [N, C]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights
from typing import List, Optional


class ResNetBackbone(nn.Module):
    """ResNet feature extractor for SGG.

    Removes the final classification head and avgpool, keeping only
    convolutional layers. Outputs feature maps at stride 32.

    Args:
        arch: 'resnet50' or 'resnet101'.
        pretrained: Load ImageNet pre-trained weights.
        frozen: Freeze backbone weights during training.
        output_dim: Output feature dimension (2048 for resnet50/101).
    """

    def __init__(self, arch: str = 'resnet50', pretrained: bool = True,
                 frozen: bool = True):
        super().__init__()

        weights = None
        if pretrained:
            weights = (ResNet50_Weights.IMAGENET1K_V1 if arch == 'resnet50'
                       else ResNet101_Weights.IMAGENET1K_V1)

        if arch == 'resnet50':
            resnet = resnet50(weights=weights)
            self.output_dim = 2048
        elif arch == 'resnet101':
            resnet = resnet101(weights=weights)
            self.output_dim = 2048
        else:
            raise ValueError(f"Unsupported arch: {arch}")

        # Keep conv layers only, remove avgpool and fc
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Output stride: 32 (5 downsampling operations)
        # Input [3, H, W] → output [2048, H/32, W/32]

        if frozen:
            self.freeze()

    def freeze(self):
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze for fine-tuning."""
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W] or [3, H, W] tensor, normalized with
                    ImageNet mean/std.

        Returns:
            feature_map: [B, 2048, H/32, W/32] or [2048, H/32, W/32].
        """
        squeeze_out = (images.dim() == 3)
        if squeeze_out:
            images = images.unsqueeze(0)

        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if squeeze_out:
            x = x.squeeze(0)

        return x

    @property
    def stride(self) -> int:
        return 32


class ROIAlignExtractor(nn.Module):
    """Extract ROI features from feature maps using ROI Align.

    Takes backbone feature maps + object boxes → per-object visual features.

    Args:
        output_size: ROI Align output spatial size (default 7).
        pool: Pooling method after ROI Align: 'avg' (adaptive avg pool to 1x1)
              or 'flatten'.
    """

    def __init__(self, output_size: int = 7, pool: str = 'avg'):
        super().__init__()
        self.output_size = output_size
        self.pool = pool

    def forward(self, feature_map: torch.Tensor, boxes: torch.Tensor,
                image_size: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: [C, H_f, W_f] backbone feature map (single image).
            boxes: [N, 4] object boxes in (cx, cy, w, h) normalized [0, 1].
            image_size: [2] tensor (H_img, W_img) of the ORIGINAL image
                        (before backbone downsampling).

        Returns:
            [N, C] per-object visual features.
        """
        C, H_f, W_f = feature_map.shape
        device = feature_map.device
        N = boxes.size(0)

        if N == 0:
            return feature_map.new_zeros(0, C)

        # Convert normalized cxcywh → absolute x1y1x2y2 in image coordinates
        img_h, img_w = image_size[0].float(), image_size[1].float()

        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h

        # Map to feature map coordinates
        scale_x = W_f / img_w
        scale_y = H_f / img_h
        x1_f = x1 * scale_x
        y1_f = y1 * scale_y
        x2_f = x2 * scale_x
        y2_f = y2 * scale_y

        # ROI Align expects [K, 5] format: [batch_idx, x1, y1, x2, y2]
        batch_idx = torch.zeros(N, 1, device=device)
        rois = torch.cat([batch_idx, x1_f.unsqueeze(1), y1_f.unsqueeze(1),
                          x2_f.unsqueeze(1), y2_f.unsqueeze(1)], dim=1)

        # ROI Align: [B, C, H_f, W_f] with [K, 5] rois → [K, C, out_s, out_s]
        fm_batched = feature_map.unsqueeze(0)  # [1, C, H_f, W_f]
        roi_feats = roi_align(fm_batched, rois,
                              output_size=(self.output_size, self.output_size),
                              spatial_scale=1.0,  # we already mapped coords
                              aligned=True)

        if self.pool == 'avg':
            # Adaptive avg pool to 1x1 → [K, C, 1, 1] → [K, C]
            roi_feats = F.adaptive_avg_pool2d(roi_feats, (1, 1))
            return roi_feats.squeeze(-1).squeeze(-1)
        else:
            return roi_feats.flatten(1)


class VisualFeatureExtractor(nn.Module):
    """Complete visual feature extraction pipeline.

    Combines ResNet backbone + ROI Align into a single module.
    Handles variable-sized images by processing them individually.

    Pipeline:
      Image → ResNet backbone → feature map
      Boxes (normalized cxcywh) → ROI Align → per-object features [N, output_dim]
    """

    def __init__(self, arch: str = 'resnet50', pretrained: bool = True,
                 frozen: bool = True, roi_output_size: int = 7):
        super().__init__()
        self.backbone = ResNetBackbone(arch, pretrained, frozen)
        self.roi_align = ROIAlignExtractor(output_size=roi_output_size, pool='avg')
        self.output_dim = self.backbone.output_dim

    def forward(self, images: List[torch.Tensor],
                boxes_list: List[torch.Tensor],
                image_sizes: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extract per-object visual features for a batch of images.

        Args:
            images: list of [3, H_i, W_i] tensors (variable sizes).
            boxes_list: list of [N_i, 4] normalized boxes (cxcywh).
            image_sizes: list of [2] tensors (H, W) of original images.

        Returns:
            list of [N_i, output_dim] per-object visual features.
        """
        results = []
        for img, boxes, img_size in zip(images, boxes_list, image_sizes):
            # Move to backbone device
            img = img.to(next(self.backbone.parameters()).device)
            boxes = boxes.to(img.device)
            img_size = img_size.to(img.device)

            # Feature extraction
            with torch.set_grad_enabled(not all(p.requires_grad == False
                                                for p in self.backbone.parameters())):
                fm = self.backbone(img)  # [C, H_f, W_f]

            # ROI Align
            feats = self.roi_align(fm, boxes, img_size)  # [N, C]
            results.append(feats)

        return results

    def to(self, device):
        """Move to device, returning self for chaining."""
        super().to(device)
        return self


def build_visual_extractor(args) -> VisualFeatureExtractor:
    """Build visual feature extractor from config args."""
    return VisualFeatureExtractor(
        arch=getattr(args, 'backbone_arch', 'resnet50'),
        pretrained=getattr(args, 'backbone_pretrained', True),
        frozen=getattr(args, 'backbone_frozen', True),
        roi_output_size=getattr(args, 'roi_output_size', 7),
    )
