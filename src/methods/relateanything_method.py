"""Registry facade for supplied-region relation inference.

Its runtime deliberately bypasses the label-conditioned Lightning datamodule.
The complete model/losses live in src.modules.relateanything.
"""
from src.relateanything import RelateAnythingModel


class RelateAnything_Method(RelateAnythingModel):
    """Checkpoint-backed RelateAnything method with its distinct input contract."""
