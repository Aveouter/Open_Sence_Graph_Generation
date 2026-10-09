"""Official supplied-region metrics with explicit coverage accounting."""


def load_benchmark_bank(checkpoint, predicates):
    """Read the hash-audited release bank in its original predicate order.

    Official releases store names as a NumPy object array; pickle loading matches
    the official API. The CLI provenance gate runs before this deserialization.
    """
    from pathlib import Path

    import numpy as np

    with np.load(Path(checkpoint).resolve().parent / "predicate_embeddings.npz", allow_pickle=True) as bank:
        if bank["names"].tolist() != list(predicates):
            raise ValueError("release predicate order differs from checkpoint")
        embeddings = bank["W"].copy()
    if embeddings.ndim != 2 or embeddings.shape[0] != len(predicates) or not np.isfinite(embeddings).all():
        raise ValueError("release embedding bank has invalid dimensions or values")
    return embeddings


def encode_benchmark_predicates(predicates, text_student, *, device="cpu"):
    """Use the benchmark's training-template ensemble in the released text space."""
    from src.modules.relateanything.text.student import encode_texts_student
    from src.modules.relateanything.vocabulary import TRAIN_TEMPLATES

    return encode_texts_student(predicates, text_student, templates=TRAIN_TEMPLATES, device=device)


class RegionEvaluator:
    """Preserve official aggregation; reject silent denominator reductions."""

    def __init__(self, *, num_predicates, topk=(20, 50, 100), protocol="A1", match_matrix=None):
        from src.modules.relateanything.eval.evaluator import (
            SGClsEvaluator,
            SoftSGClsEvaluator,
        )

        self.protocol = protocol
        if protocol == "A1":
            self.evaluator = SGClsEvaluator(topk=topk, num_predicates=num_predicates,
                                            graph_constraint=True, score_mode="sigmoid")
        elif protocol == "A3" and match_matrix is not None:
            import torch
            self.evaluator = SoftSGClsEvaluator(group_of=torch.arange(match_matrix.shape[1]),
                                                match_matrix=match_matrix, topk=topk,
                                                graph_constraint=True, score_mode="sigmoid")
        else:
            raise ValueError("A3 requires an official text-space match matrix; protocol must be A1 or A3")
        self.reset()

    def reset(self):
        self.evaluator.reset()
        self.seen_images = 0
        self.empty_predictions = 0
        self.empty_ground_truth = 0
        self.scored_relations = 0

    def update(self, outputs, targets):
        if len(targets) != outputs["logits"].shape[0]:
            raise ValueError("one region target is required per image")
        for mask, target in zip(outputs["valid_mask"], targets, strict=True):
            self.seen_images += 1
            self.empty_predictions += int(not mask.any())
            relations = target.get("relations")
            self.empty_ground_truth += int(relations is None or len(relations) == 0)
            self.scored_relations += 0 if relations is None else len(relations)
        self.evaluator.update(outputs, targets)

    def compute(self):
        if not self.seen_images or self.empty_predictions or self.empty_ground_truth:
            raise RuntimeError(
                f"official region evaluator would reduce the denominator: seen={self.seen_images}, "
                f"empty_predictions={self.empty_predictions}, empty_ground_truth={self.empty_ground_truth}")
        return {f"regions_{self.protocol}_{key}": value for key, value in self.evaluator.compute().items()}


def evaluate_batches(model, loader, *, num_predicates, protocol="A1", match_matrix=None,
                     eval_budget=500, device="cpu"):
    """Evaluate official packed batches; callers must establish full input gates.

    No GT relations or object labels reach the network. No batch is filtered on
    failure. This reusable loop is separate from the CLI's provenance gate.
    """
    import torch

    evaluator = RegionEvaluator(num_predicates=num_predicates, protocol=protocol, match_matrix=match_matrix)
    network = model.predictor.model
    network.eval()
    original_budget = network.sampler.final_budget
    network.sampler.final_budget = min(eval_budget, network.sampler.geo_budget)
    device = torch.device(device)
    try:
        with torch.inference_mode():
            for images, boxes, box_counts, targets in loader:
                cov, fill = getattr(targets, "cov", None), getattr(targets, "fill", None)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.bfloat16):
                    output = model.forward_regions(
                        images.to(device), boxes.to(device), box_counts.to(device),
                        cov=None if cov is None else cov.to(device),
                        fill=None if fill is None else fill.to(device))
                evaluator.update(output, targets)
    finally:
        network.sampler.final_budget = original_budget
    return {"metrics": evaluator.compute(), "seen_images": evaluator.seen_images,
            "scored_relations": evaluator.scored_relations,
            "empty_predictions": evaluator.empty_predictions,
            "empty_ground_truth": evaluator.empty_ground_truth}
