"""RA-SGG method wrapper for OpenSGG.

The wrapper deliberately reuses the Motifs/PENet two-stage data flow. It is an
implementation-audit adapter and does not assert official ReTAG reproduction.
"""

from src.models.ra_sgg import build_ra_sgg

from .motifs_method import MotifsCriterion, Motifs_Method


class RA_SGG_Method(Motifs_Method):
    """Minimal RA-SGG/ReTAG adapter with Motifs-compatible batches."""

    def _build_model(self, **args):
        return build_ra_sgg(self.hparams)

    def _build_criterion(self, **args):
        return MotifsCriterion(
            num_predicates=args.get("rel_nums", 51),
        )
