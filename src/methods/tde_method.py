"""Official-style TDE method wrapper for PyTorch Lightning."""

from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.motifs import build_tde


class TDECriterion(MotifsCriterion):
    """Motifs predicate/object loss plus official TDE auxiliary branch losses."""

    def forward(self, outputs: dict, targets: list) -> dict:
        loss_dict = super().forward(outputs, targets)
        total_loss = loss_dict["loss_total"]
        for name, value in outputs.get("add_losses", {}).items():
            loss_name = f"loss_{name}"
            loss_dict[loss_name] = value
            total_loss = total_loss + value
        loss_dict["loss_total"] = total_loss
        return loss_dict


class TDE_Method(Motifs_Method):
    """TDE Lightning method backed by an official-style causal predictor."""

    def __init__(self, **args):
        self._tde_model = None
        super().__init__(**args)

    def _build_criterion(self, **args):
        return TDECriterion(
            num_predicates=args.get('rel_nums', 51),
        )

    def _build_model(self, **args):
        model = build_tde(self.hparams)
        self._tde_model = model
        return model

    def _extra_model_kwargs(self, target, boxes, labels, return_obj_preds):
        extra = super()._extra_model_kwargs(target, boxes, labels, return_obj_preds)
        if "rel_annotations" in target:
            extra["rel_annotations"] = target["rel_annotations"]
        return extra
