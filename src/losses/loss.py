import torch
import torch.nn as nn
import torch.nn.functional as F
from .reweight_loss import EdgeDensityLoss, FocalLoss

def _stack_targets_if_needed(targets, key, ref_tensor):
    """
    兼容两种 target 形式:
    1) batched dict: {"object_labels": [B,...], ...}
    2) list/tuple of dict: [{...}, {...}, ...]

    返回:
        Tensor or None
    """
    if isinstance(targets, dict):
        value = targets.get(key, None)
        if torch.is_tensor(value):
            return value.to(device=ref_tensor.device)
        return value

    if isinstance(targets, (list, tuple)):
        if len(targets) == 0:
            return None

        values = []
        for t in targets:
            if not isinstance(t, dict) or key not in t:
                return None
            v = t[key]
            if not torch.is_tensor(v):
                return None
            values.append(v)

        try:
            return torch.stack(values, dim=0).to(device=ref_tensor.device)
        except Exception:
            return None

    return None


class HSTRCriterion(nn.Module):
    def __init__(self):
        super().__init__()

        # 统一写死默认权重，和 loss_construction 的无参风格对齐
        self.loss_object_weight = 1.0
        self.loss_relationness_weight = 1.0
        self.loss_predicate_weight = 1.0
        self.loss_proto_align_weight = 0.2
        self.loss_hier_consistency_weight = 0.2
        self.loss_energy_weight = 0.2

    def object_loss(self, object_logits, targets):
        gt = _stack_targets_if_needed(targets, "object_labels", object_logits)
        if gt is None:
            return object_logits.new_tensor(0.0)

        B, T, Nq, No = object_logits.shape
        return F.cross_entropy(
            object_logits.reshape(B * T * Nq, No),
            gt.reshape(B * T * Nq).long(),
            ignore_index=-100
        )

    def relationness_loss(self, relationness_logits_dense, targets):
        gt = _stack_targets_if_needed(targets, "relation_labels_dense", relationness_logits_dense)
        if gt is None:
            return relationness_logits_dense.new_tensor(0.0)

        gt = gt.float()
        return F.binary_cross_entropy_with_logits(
            relationness_logits_dense, gt, reduction="mean"
        )

    def predicate_loss(self, final_predicate_logits, targets):
        gt = _stack_targets_if_needed(targets, "pair_predicate_labels", final_predicate_logits)
        if gt is None:
            return final_predicate_logits.new_tensor(0.0)

        B, K, P = final_predicate_logits.shape
        return F.cross_entropy(
            final_predicate_logits.reshape(B * K, P),
            gt.reshape(B * K).long(),
            ignore_index=-100
        )

    def prototype_alignment_loss(self, prototype_outputs):
        probs_list = prototype_outputs["level_probs"]
        loss = 0.0
        for probs in probs_list:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss += entropy
        return loss / max(len(probs_list), 1)

    def hierarchical_consistency_loss(self, prototype_outputs):
        if len(prototype_outputs["parent_child_logits"]) == 0:
            return prototype_outputs["refined_features"].new_tensor(0.0)

        total_loss = 0.0
        count = 0
        level_probs = prototype_outputs["level_probs"]
        trans_list = prototype_outputs["parent_child_logits"]

        for i, trans_logits in enumerate(trans_list):
            coarse_prob = level_probs[i]
            fine_prob = level_probs[i + 1]
            trans = F.softmax(trans_logits, dim=-1)
            fine_to_coarse = torch.matmul(fine_prob, trans.t())
            total_loss += F.mse_loss(fine_to_coarse, coarse_prob)
            count += 1

        return total_loss / max(count, 1)

    def compatibility_loss(self, energy_scores, targets):
        gt = _stack_targets_if_needed(targets, "pair_compatibility_labels", energy_scores)
        if gt is None:
            return energy_scores.new_tensor(0.0)

        gt = gt.float()
        return F.binary_cross_entropy_with_logits(energy_scores, gt)

    def forward(self, outputs, targets):
        l_obj = self.object_loss(outputs["object_logits"], targets)
        l_rel = self.relationness_loss(outputs["relationness_logits_dense"], targets)
        l_pred = self.predicate_loss(outputs["final_predicate_logits"], targets)
        l_proto = self.prototype_alignment_loss(outputs["prototype_outputs"])
        l_hier = self.hierarchical_consistency_loss(outputs["prototype_outputs"])
        l_energy = self.compatibility_loss(outputs["energy_scores"], targets)

        total = (
            self.loss_object_weight * l_obj
            + self.loss_relationness_weight * l_rel
            + self.loss_predicate_weight * l_pred
            + self.loss_proto_align_weight * l_proto
            + self.loss_hier_consistency_weight * l_hier
            + self.loss_energy_weight * l_energy
        )

        return {
            "loss_total": total,
            "loss_object": l_obj,
            "loss_relationness": l_rel,
            "loss_predicate": l_pred,
            "loss_proto_align": l_proto,
            "loss_hier_consistency": l_hier,
            "loss_energy": l_energy,
        }


LOSS_FACTORY = {
    "ce": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "bce": nn.BCEWithLogitsLoss,
    "focal": FocalLoss,
    "focal_loss": FocalLoss,
    "edge_density": EdgeDensityLoss,
    "edge_density_loss": EdgeDensityLoss,
    "hstrnet_loss": HSTRCriterion,
    "reltr_loss": None,  # RelTR criterion built in reltr_method._build_model
    "egtr_loss": None,   # EGTR criterion built in egtr_method._build_model
    "flowsg_loss": None,  # FlowSG criterion built in flowsg_method._build_model
    "motifs_loss": None,  # Motifs/VCTree/TDE/IMP/Transformer/GPSNet/PENet/SQUAT/SHAGCL criterion built in _build_criterion
    "react_loss": None,   # REACT criterion built in react_method._build_criterion
    "cvc_loss": None,     # CVC
    "usg_loss": None,     # USG criterion built in _build_criterion
}


def loss_construction(loss_name="ce", **kwargs):
    try:
        factory = LOSS_FACTORY[loss_name]
        return factory(**kwargs) if factory is not None else None
    except KeyError:
        raise ValueError(f"Unknown loss type: {loss_name}")
