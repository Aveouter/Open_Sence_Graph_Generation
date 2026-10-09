"""OpenSGG boundary for the complete, separately licensed RelateAnything model.

The public input is an original image plus pixel-xyxy regions. Object labels are
not model inputs. See reproduction/adr/0012 and 0013 for protocol/source identity.
"""
from pathlib import Path


class RelateAnythingModel:
    """Strict checkpoint-backed inference; never constructs a random fallback."""

    def __init__(self, predictor):
        self.predictor = predictor

    @classmethod
    def from_checkpoint(cls, checkpoint, *, predicates=None, device="cpu",
                        full_vocabulary=False, img_size=448, weights="ema",
                        calibration=True):
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"RelateAnything checkpoint not found: {checkpoint}")
        from src.modules.relateanything.provenance import verify_source
        verify_source()
        from src.modules.relateanything import RelateAnything

        predictor = RelateAnything.from_checkpoint(
            str(checkpoint), predicates=predicates, device=device, strict=True,
            full_vocabulary=full_vocabulary, img_size=img_size, weights=weights,
            calibration=calibration, text_student=str(checkpoint.parent / "text_student.pt"),
        )
        return cls(predictor)

    def predict(self, image, boxes_xyxy, *, masks=None, box_scores=None,
                topk=20, max_boxes=100, decompose=False):
        return self.predictor.predict(image, boxes_xyxy, masks=masks,
                                      box_scores=box_scores, topk=topk,
                                      max_boxes=max_boxes, decompose=decompose)

    def set_vocabulary(self, predicates):
        self.predictor.set_vocabulary(predicates)
        return self

    def forward_regions(self, images, boxes_cxcywh, box_counts, *, cov=None, fill=None):
        """Official packed batch forward; normalized boxes, no label/target input."""
        region = {} if cov is None else {"cov": cov, "fill": fill}
        return self.predictor.model(images, boxes_cxcywh, box_counts=box_counts,
                                    targets=None, **region)
