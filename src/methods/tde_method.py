"""
TDE method wrapper for PyTorch Lightning.

Integrates TDE (Total Direct Effect, Tang et al., CVPR 2020) into OpenSGG.

TDE extends Motifs with causal intervention: during inference, it subtracts
the counterfactual prediction (using mean visual features) from the factual
prediction, removing context bias.

Key difference from standard Motifs:
  - During training: behaves identically to Motifs (no TDE intervention)
  - During inference: applies `logits = logits_factual - logits_counterfactual`
"""
import torch
from .motifs_method import Motifs_Method, MotifsCriterion
from src.models.motifs import build_tde


class TDE_Method(Motifs_Method):
    """TDE Lightning method.

    Extends Motifs with causal debiasing at inference time.
    During training, TDE behaves identically to Motifs.
    """

    def __init__(self, **args):
        # TDE uses the same criterion but model is TDEModel
        self._tde_model = None
        super().__init__(**args)

    def _build_model(self, **args):
        model = build_tde(self.hparams)
        self._tde_model = model
        return model

    def _compute_mean_visual_feat(self):
        """Compute mean visual features for the counterfactual baseline.

        Uses per-class embedding prototypes as visual feature substitutes.
        When a real backbone is integrated, override to iterate over the
        training dataloader and compute the actual mean ROI feature.
        """
        print("[TDE] Computing mean visual features for counterfactual...")

        with torch.no_grad():
            visual_extractor = getattr(self, '_visual_extractor', None)
            if isinstance(visual_extractor, torch.nn.Embedding):
                embed_weight = visual_extractor.weight.data
                mean_feat = embed_weight.mean(dim=0).to(self.device)
                self._tde_model.set_mean_visual_feat(mean_feat)
                print(f"[TDE] Mean visual feature set (dim={mean_feat.size(0)})")
                return

        # Fallback: zero features
        visual_dim = getattr(self.hparams, 'visual_dim', 2048)
        self._tde_model.set_mean_visual_feat(torch.zeros(visual_dim, device=self.device))
        print("[TDE] Using zero mean visual feature (fallback)")

    def on_test_start(self):
        """Compute mean visual features before evaluation."""
        super().on_test_start()
        if self._tde_model is not None:
            self._compute_mean_visual_feat()

    def on_validation_epoch_start(self):
        """Ensure TDE baseline is set."""
        if self._tde_model is not None:
            if self._tde_model.mean_visual_feat.sum() == 0:
                self._compute_mean_visual_feat()
