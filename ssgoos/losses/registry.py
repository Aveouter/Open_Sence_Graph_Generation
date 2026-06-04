# ssgoos/losses/registry.py
"""
Loss function registry for SGG framework.

Replaces the hardcoded LOSS_FACTORY dict in src/loss.py.
"""

from ssgoos.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register('ce')
def build_ce(config):
    """Cross-entropy loss."""
    import torch.nn as nn
    return nn.CrossEntropyLoss()


@LOSS_REGISTRY.register('mse')
def build_mse(config):
    """Mean squared error loss."""
    import torch.nn as nn
    return nn.MSELoss()


@LOSS_REGISTRY.register('bce')
def build_bce(config):
    """Binary cross-entropy loss."""
    import torch.nn as nn
    return nn.BCEWithLogitsLoss()


@LOSS_REGISTRY.register('reltr_loss')
def build_reltr_loss(config):
    """
    RelTR loss is built internally by the model adapter.
    Returns None to signal that the adapter handles loss construction.
    """
    return None


@LOSS_REGISTRY.register('hstrnet_loss')
def build_hstrnet_loss(config):
    """HSTRNet composite loss criterion."""
    from ssgoos.losses.loss import HSTRCriterion
    return HSTRCriterion(config)


def loss_construction(loss_name="ce"):
    """
    Build a loss module by name.

    This replaces the original LOSS_FACTORY dict lookup.
    """
    if loss_name not in LOSS_REGISTRY:
        raise KeyError(
            f"Loss '{loss_name}' not found in LOSS_REGISTRY. "
            f"Available: {LOSS_REGISTRY.list()}"
        )
    builder = LOSS_REGISTRY.get(loss_name)
    # The builder is a function, not a class, so we call it
    return builder(None)  # config not needed for simple losses
