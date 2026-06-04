# src/methods/reltr_method.py
import torch
from .base_method import Base_method

# RelTR 依赖的工具：NestedTensor / converter
from utils.misc import NestedTensor, nested_tensor_from_tensor_list

# 你的 RelTR build()（就是你贴的 reltr.py 里的 build(args）
# 注意：按你工程结构，可能是 from src.models.reltr import build
from src.models.reltr import build as build_reltr


class RelTR_Method(Base_method):
    """
    RelTR method wrapper (compatible with Base_method style).

    训练 batch 约定：
      batch = (images, targets)
        images:  Tensor[B,3,H,W] 或 list[Tensor(3,H,W)]
        targets: list[dict], 每个 dict 至少包含:
          - "labels": LongTensor[num_obj]
          - "boxes":  FloatTensor[num_obj,4]  (cx,cy,w,h) normalized in [0,1]
          - "rel_annotations": LongTensor[num_rel,3]  (sub_idx, obj_idx, rel_label)
    """

    def __init__(self, **args):
        super().__init__(**args)
        # weight_dict 会在 _build_model() 里从 criterion 拿到
        self.weight_dict = None

    def _build_model(self, **args):
        """
        Base_method 会用它来设置 self.model。
        但 RelTR 还需要 criterion/postprocessors，所以我们在这里顺手都建好。
        """
        # 你框架里通常是 self.hparams 保存所有超参（类似 args）
        # reltr.py 的 build(args) 期望 args.backbone/args.lr_backbone/... 等字段存在
        model, criterion, postprocessors = build_reltr(self.hparams)

        # 挂到 self 上，供 training_step / validation_step 使用
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.weight_dict = dict(getattr(criterion, "weight_dict", {}))

        return model

    # ---------- forward / predict ----------

    def forward(self, images, targets=None, return_predictions=False, target_sizes=None, **kwargs):
        """
        forward 用于推理/验证时调用。
        - 如果 targets 不为 None：会额外返回 loss_dict/total_loss
        - return_predictions=True：会额外跑 postprocess 得到像素坐标 boxes（可选）
        """
        samples = self._to_nested_tensor(images)
        outputs = self.model(samples)

        out = {"outputs": outputs}

        if targets is not None:
            targets = self._move_targets_to_device(targets)
            loss_dict, total_loss = self._compute_losses(outputs, targets)
            out["loss_dict"] = loss_dict
            out["loss"] = total_loss

        if return_predictions:
            # postprocess: outputs['pred_boxes'] 是归一化 cxcywh
            # target_sizes: Tensor[B,2] = (H,W) 原图尺寸
            if target_sizes is None:
                # 默认用当前输入张量尺寸（如果你做了 resize/pad，这不一定等于原图！）
                _, _, H, W = samples.tensors.shape
                target_sizes = torch.tensor([[H, W]] * samples.tensors.shape[0], device=self.device)
            else:
                target_sizes = target_sizes.to(self.device)

            bbox_results = self.postprocessors["bbox"](outputs, target_sizes)
            out["bbox_results"] = bbox_results
        
        return out

    # ---------- validation_step / test_step (override to cache matching indices) ----------

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'val_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Cache triplet matching indices from Hungarian matcher (for post-processing)
        triplet_indices = None
        if hasattr(self.criterion, 'indices') and self.criterion.indices is not None:
            triplet_indices = [
                (src.cpu().clone(), tgt.cpu().clone())
                for src, tgt in self.criterion.indices[1]
            ]

        self.val_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items() if k != 'aux_outputs'
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
            'triplet_indices': triplet_indices,  # Hungarian matching: list of (src_idx, tgt_idx) per image
        })

        return total_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        images, targets = batch
        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        loss_dict, total_loss = self._compute_losses(outputs, targets)

        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            if torch.is_tensor(v):
                self.log(f'test_{k}', v, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        # Cache triplet matching indices from Hungarian matcher (for post-processing)
        triplet_indices = None
        if hasattr(self.criterion, 'indices') and self.criterion.indices is not None:
            triplet_indices = [
                (src.cpu().clone(), tgt.cpu().clone())
                for src, tgt in self.criterion.indices[1]
            ]

        self.test_outputs.append({
            'outputs': {
                k: (v.detach().cpu() if torch.is_tensor(v) else v)
                for k, v in outputs.items() if k != 'aux_outputs'
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
            'triplet_indices': triplet_indices,
        })

        return total_loss

    # ---------- training_step ----------

    def training_step(self, batch, batch_idx):
        """
        Lightning-style training_step:
          - 计算 RelTR 的各项 loss
          - self.log(...)
          - return total loss
        """
        images, targets = batch

        samples = self._to_nested_tensor(images)
        targets = self._move_targets_to_device(targets)

        outputs = self.model(samples)
        # print(outputs.keys())
        # exit()
        loss_dict, total_loss = self._compute_losses(outputs, targets)
        
        # logging（你可以按自己习惯改 key）
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)

        # 记录各子项 loss
        for k, v in loss_dict.items():
            # 只记录标量
            if torch.is_tensor(v):
                self.log(f'train_{k}', v, on_step=True, on_epoch=True, prog_bar=False)

        return total_loss

    # ---------- helpers ----------

    def _to_nested_tensor(self, images) -> NestedTensor:
        """
        把 (Tensor 或 list[Tensor]) 转成 NestedTensor，并放到 device。
        """
        if isinstance(images, NestedTensor):
            samples = images
        else:
            samples = nested_tensor_from_tensor_list(images)

        # 确保上 device（NestedTensor 里有 tensors 和 mask）
        tensors = samples.tensors.to(self.device)
        mask = samples.mask.to(self.device) if samples.mask is not None else None
        return NestedTensor(tensors, mask)

    def _move_targets_to_device(self, targets):
        """
        targets: list[dict]，把里面的 Tensor 全部搬到 device
        """
        moved = []
        for t in targets:
            t2 = {}
            for k, v in t.items():
                t2[k] = v.to(self.device) if torch.is_tensor(v) else v
            moved.append(t2)
        return moved

    def _compute_losses(self, outputs, targets):
        """
        criterion(outputs, targets) -> loss_dict
        然后按 criterion.weight_dict 聚合 total_loss
        """
        loss_dict = self.criterion(outputs, targets)

        # 加权求和（只对 weight_dict 里定义的项求和）
        total_loss = 0.0
        if self.weight_dict is None:
            # 兜底：直接把所有 tensor loss 相加
            for v in loss_dict.values():
                if torch.is_tensor(v):
                    total_loss = total_loss + v
            return loss_dict, total_loss

        for k, v in loss_dict.items():
            if k in self.weight_dict:
                total_loss = total_loss + v * self.weight_dict[k]

        return loss_dict, total_loss
