"""
GPS-Net method wrapper for PyTorch Lightning.
"""
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.gpsnet import build_gpsnet


class GPSNet_Method(Motifs_Method):
    """GPS-Net Lightning method — same data flow as Motifs."""

    def _build_model(self, **args):
        return build_gpsnet(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get('rel_nums', 51),
        )
