import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.penet import PENetContext


def _tiny_penet() -> PENetContext:
    model = PENetContext(
        num_classes=8,
        num_predicates=6,
        visual_dim=32,
        hidden_dim=16,
        context_hidden_dim=8,
        embed_dim=8,
        pooling_dim=32,
        use_freq_bias=False,
        dropout=0.0,
        train_pairs_per_image=4,
        train_positive_fraction=0.5,
    )
    return model


def _inputs(num_objects=4):
    torch.manual_seed(7)
    visual_feats = torch.randn(num_objects, 32)
    boxes = torch.tensor(
        [
            [0.10, 0.10, 0.25, 0.25],
            [0.40, 0.10, 0.20, 0.25],
            [0.15, 0.50, 0.30, 0.20],
            [0.55, 0.55, 0.20, 0.30],
        ],
        dtype=torch.float32,
    )[:num_objects]
    labels = torch.arange(1, num_objects + 1, dtype=torch.long)
    return visual_feats, boxes, labels


def test_penet_relation_features_disabled_by_default():
    model = _tiny_penet().eval()
    visual_feats, boxes, labels = _inputs(3)

    with torch.no_grad():
        out = model(visual_feats, boxes, labels)

    assert "relation_features" not in out
    assert out["pair_indices"].shape == (6, 2)
    assert out["rel_logits"].shape == (6, 6)


def test_penet_relation_features_schema_and_forbidden_fields():
    model = _tiny_penet().eval()
    visual_feats, boxes, labels = _inputs(3)
    union_feats = torch.randn(6, 32)
    image_size = torch.tensor([600.0, 800.0])

    with torch.no_grad():
        out = model(
            visual_feats,
            boxes,
            labels,
            union_feats=union_feats,
            _image_size=image_size,
            return_relation_features=True,
        )

    feats = out["relation_features"]
    assert feats["schema"] == "penet_relation_feature_exposure_v1"
    assert feats["feature_mode"] == "eval_all_directed_pairs"
    assert feats["pair_indices"].shape == (6, 2)
    assert feats["subject_visual"].shape == (6, 16)
    assert feats["object_visual"].shape == (6, 16)
    assert feats["subject_context"].shape == (6, 16)
    assert feats["object_context"].shape == (6, 16)
    assert feats["union_feature"].shape == (6, 32)
    assert feats["union_semantic"].shape == (6, 16)
    assert feats["relation_pre_proj"].shape == (6, 16)
    assert feats["relation_proj"].shape == (6, 32)
    assert feats["predicate_proto_proj"].shape == (6, 32)
    assert feats["boxes_xyxy_or_cxcywh_norm"].shape == (3, 4)
    assert feats["object_labels"].shape == (3,)
    assert torch.equal(feats["image_size_hw"], image_size)

    forbidden = {
        "rel_logits",
        "frequency-biased_logits",
        "object_pair_prior",
        "candidate_retrieval_scores",
        "text_scores",
        "clip_scores",
        "vlm_scores",
    }
    assert forbidden.isdisjoint(feats.keys())
    for key, value in feats.items():
        if torch.is_tensor(value):
            assert not value.requires_grad, key


def test_penet_relation_feature_mode_uses_all_pairs_even_when_training():
    model = _tiny_penet().train()
    visual_feats, boxes, labels = _inputs(4)
    rel_annotations = torch.tensor([[0, 1, 3], [2, 3, 4]], dtype=torch.long)

    out = model(
        visual_feats,
        boxes,
        labels,
        rel_annotations=rel_annotations,
        return_relation_features=True,
    )

    # N * (N - 1) all directed pairs. Training pair sampling would cap this at
    # train_pairs_per_image=4, so this assertion protects the diagnostic path.
    assert out["relation_features"]["pair_indices"].shape == (12, 2)
    assert out["pair_indices"].shape == (12, 2)
    assert out["add_losses"] == {}
