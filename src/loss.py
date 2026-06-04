# Backward-compatibility shim — the canonical source is src/losses/
# Kept here so that `from src.loss import *` in base_method.py continues to work.

from src.losses import (
    HSTRCriterion,
    LOSS_FACTORY,
    loss_construction,
)
