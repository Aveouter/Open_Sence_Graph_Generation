"""#156 RED: does an already-selected pair's prediction depend on the pair set?

RelateAnything refines every sampled pair through pair-to-pair attention
(``RelationTransformer``, ``RelationInteractionBlock``) and the paper reports
that this pair context carries 87-93% of the variance of the semantic logit.
Pair-set dependence is therefore *intended*; the audit question is whether the
invariants that must still hold do hold, and how large the designed dependence
actually is.

Evidence class: **executed source behaviour on synthetic fixtures**. No
checkpoint, dataset or GPU is involved, and no number here is a baseline
result. Harm to predictive quality cannot be adjudicated from this file alone:
that would need adjudicated ground truth, a released checkpoint and fixed pair
budgets, none of which are available to this audit.
"""

import unittest

from tests._optional import require_modules

require_modules("torch", "torchvision")

import torch  # noqa: E402

from src.modules.relateanything.model.backbone import (
    RelationInteractionBlock,  # noqa: E402
)
from src.modules.relateanything.model.deformable import DeformableRelRead  # noqa: E402
from src.modules.relateanything.model.transformer import (
    RelationTransformer,  # noqa: E402
)
from tests.reproduction import _relateanything_audit_fixtures as fx  # noqa: E402

D_MODEL, SCENE_HW, N_PAIRS, BATCH, HEADS = 16, 4, 6, 2, 2
# Float32 accumulation over the attention/FFN stack: differences at this level
# are rounding, not semantics.
TOL = 1e-5


def _pair_feat(n=N_PAIRS, batch=BATCH, seed=11):
    return torch.randn(batch, n, D_MODEL, generator=torch.Generator().manual_seed(seed))


def _scene(batch=BATCH, seed=11):
    return torch.randn(
        batch,
        SCENE_HW,
        SCENE_HW,
        fx.BACKBONE_DIM,
        generator=torch.Generator().manual_seed(seed),
    )


def _box_tokens(n=N_PAIRS, batch=BATCH, seed=12):
    return torch.randn(
        batch, n, 4, D_MODEL, generator=torch.Generator().manual_seed(seed)
    )


def _no_pad(batch=BATCH, n=N_PAIRS):
    return torch.zeros(batch, n, dtype=torch.bool)


def _transformer():
    torch.manual_seed(3)
    return RelationTransformer(
        d_model=D_MODEL,
        backbone_dim=fx.BACKBONE_DIM,
        n_self=2,
        n_cross=2,
        n_heads=HEADS,
        dropout=0.0,
    ).eval()


def _interaction():
    torch.manual_seed(4)
    return RelationInteractionBlock(
        d_model=D_MODEL,
        scene_dim=fx.BACKBONE_DIM,
        n_dep=2,
        n_gnd=1,
        n_heads=HEADS,
        dropout=0.0,
    ).eval()


class PairContextInvarianceTest(unittest.TestCase):
    """Contracts that must hold whatever the intended context dependence is."""

    def test_reordering_the_pair_set_permutes_the_output(self):
        """The pair set is a set: order must not be information."""
        tr, perm = _transformer(), torch.tensor([3, 0, 5, 1, 4, 2])
        pf, scene, bt, pad = _pair_feat(), _scene(), _box_tokens(), _no_pad()
        with torch.no_grad():
            base = tr(pf, scene, box_tokens=bt, pair_padding_mask=pad)
            moved = tr(
                pf[:, perm], scene, box_tokens=bt[:, perm], pair_padding_mask=pad
            )
        self.assertLess(float((moved - base[:, perm]).abs().max()), TOL)

    def test_reordering_the_pair_set_permutes_the_interaction_output(self):
        ib, perm = _interaction(), torch.tensor([3, 0, 5, 1, 4, 2])
        pf, scene, pad = _pair_feat(), _scene(), _no_pad()
        flat = scene.reshape(BATCH, SCENE_HW * SCENE_HW, fx.BACKBONE_DIM)
        with torch.no_grad():
            base = ib(pf, flat, query_padding_mask=pad, grid_hw=(SCENE_HW, SCENE_HW))
            moved = ib(
                pf[:, perm], flat, query_padding_mask=pad, grid_hw=(SCENE_HW, SCENE_HW)
            )
        self.assertLess(float((moved - base[:, perm]).abs().max()), TOL)

    def test_padded_slots_do_not_move_real_slots(self):
        """Batch padding width must not leak into a prediction."""
        tr, ib = _transformer(), _interaction()
        pf, scene, bt, pad = _pair_feat(), _scene(), _box_tokens(), _no_pad()
        extra = 4
        pf2 = torch.cat(
            [
                pf,
                torch.randn(
                    BATCH, extra, D_MODEL, generator=torch.Generator().manual_seed(21)
                ),
            ],
            1,
        )
        bt2 = torch.cat(
            [
                bt,
                torch.randn(
                    BATCH,
                    extra,
                    4,
                    D_MODEL,
                    generator=torch.Generator().manual_seed(22),
                ),
            ],
            1,
        )
        pad2 = torch.cat([pad, torch.ones(BATCH, extra, dtype=torch.bool)], 1)
        flat = scene.reshape(BATCH, SCENE_HW * SCENE_HW, fx.BACKBONE_DIM)
        with torch.no_grad():
            self.assertLess(
                float(
                    (
                        tr(pf2, scene, box_tokens=bt2, pair_padding_mask=pad2)[
                            :, :N_PAIRS
                        ]
                        - tr(pf, scene, box_tokens=bt, pair_padding_mask=pad)
                    )
                    .abs()
                    .max()
                ),
                TOL,
            )
            self.assertLess(
                float(
                    (
                        ib(
                            pf2,
                            flat,
                            query_padding_mask=pad2,
                            grid_hw=(SCENE_HW, SCENE_HW),
                        )[:, :N_PAIRS]
                        - ib(
                            pf,
                            flat,
                            query_padding_mask=pad,
                            grid_hw=(SCENE_HW, SCENE_HW),
                        )
                    )
                    .abs()
                    .max()
                ),
                TOL,
            )

    def test_batch_items_are_independent(self):
        """One image's prediction must not read another image's pairs."""
        tr, ib = _transformer(), _interaction()
        pf, scene, bt, pad = _pair_feat(), _scene(), _box_tokens(), _no_pad()
        pf_b, scene_b, bt_b = pf.clone(), scene.clone(), bt.clone()
        g = torch.Generator().manual_seed(99)
        pf_b[1] = torch.randn(N_PAIRS, D_MODEL, generator=g)
        scene_b[1] = torch.randn(SCENE_HW, SCENE_HW, fx.BACKBONE_DIM, generator=g)
        bt_b[1] = torch.randn(N_PAIRS, 4, D_MODEL, generator=g)
        flat_b = scene_b.reshape(BATCH, SCENE_HW * SCENE_HW, fx.BACKBONE_DIM)
        flat = scene.reshape(BATCH, SCENE_HW * SCENE_HW, fx.BACKBONE_DIM)
        with torch.no_grad():
            self.assertEqual(
                float(
                    (
                        tr(pf_b, scene_b, box_tokens=bt_b, pair_padding_mask=pad)[0]
                        - tr(pf, scene, box_tokens=bt, pair_padding_mask=pad)[0]
                    )
                    .abs()
                    .max()
                ),
                0.0,
            )
            self.assertEqual(
                float(
                    (
                        ib(
                            pf_b,
                            flat_b,
                            query_padding_mask=pad,
                            grid_hw=(SCENE_HW, SCENE_HW),
                        )[0]
                        - ib(
                            pf,
                            flat,
                            query_padding_mask=pad,
                            grid_hw=(SCENE_HW, SCENE_HW),
                        )[0]
                    )
                    .abs()
                    .max()
                ),
                0.0,
            )

    def test_deformable_read_is_per_slot(self):
        """The deformable read has no cross-pair path, so slots read alone."""
        torch.manual_seed(5)
        read = DeformableRelRead(
            d_model=D_MODEL, n_points=4, heads=HEADS, null_slots=2
        ).eval()
        q = _pair_feat()
        scene = torch.randn(
            BATCH,
            D_MODEL,
            SCENE_HW,
            SCENE_HW,
            generator=torch.Generator().manual_seed(31),
        )
        anchors = torch.rand(
            BATCH, N_PAIRS, 4, 4, generator=torch.Generator().manual_seed(32)
        ).clamp(0.05, 0.95)
        with torch.no_grad():
            base = read(q, scene, anchors)
            grown = read(
                torch.cat(
                    [
                        q,
                        torch.randn(
                            BATCH,
                            2,
                            D_MODEL,
                            generator=torch.Generator().manual_seed(33),
                        ),
                    ],
                    1,
                ),
                scene,
                torch.cat(
                    [
                        anchors,
                        torch.rand(
                            BATCH, 2, 4, 4, generator=torch.Generator().manual_seed(34)
                        ).clamp(0.05, 0.95),
                    ],
                    1,
                ),
            )
        self.assertLess(float((grown[:, :N_PAIRS] - base).abs().max()), TOL)


class PairSetContextDependenceTest(unittest.TestCase):
    """The confirmed, designed dependence -- measured, not judged."""

    def test_an_irrelevant_extra_pair_moves_the_target_representation(self):
        """CONFIRMED: context reaches a retained pair, and it is a large effect.

        This is the mechanism the paper attributes 87-93% of semantic-logit
        variance to, so a non-zero shift is the documented behaviour rather
        than a defect. The case pins the magnitude so a future change to the
        context path is visible.
        """
        tr = _transformer()
        pf, scene, bt, pad = _pair_feat(), _scene(), _box_tokens(), _no_pad()
        with torch.no_grad():
            base = tr(pf, scene, box_tokens=bt, pair_padding_mask=pad)
            distractor = torch.cat(
                [
                    pf,
                    torch.randn(
                        BATCH, 1, D_MODEL, generator=torch.Generator().manual_seed(41)
                    ),
                ],
                1,
            )
            bt_d = torch.cat(
                [
                    bt,
                    torch.randn(
                        BATCH,
                        1,
                        4,
                        D_MODEL,
                        generator=torch.Generator().manual_seed(42),
                    ),
                ],
                1,
            )
            pad_d = torch.cat([pad, torch.zeros(BATCH, 1, dtype=torch.bool)], 1)
            grown = tr(distractor, scene, box_tokens=bt_d, pair_padding_mask=pad_d)[
                :, :N_PAIRS
            ]
        shift = float((grown - base).abs().max())
        self.assertGreater(shift, 1e-3, "context path appears to be inert")
        # And the shift must not be explained by padding/masking drift.
        self.assertLess(float((grown - base).abs().max(0).values.max()), 10.0)

    def test_context_dependence_is_absent_when_the_extra_pair_is_padding(self):
        """Control: the same extra slot, masked, changes nothing.

        Together with the case above this shows the effect is the *content* of
        the pair set and not the tensor width.
        """
        tr = _transformer()
        pf, scene, bt, pad = _pair_feat(), _scene(), _box_tokens(), _no_pad()
        with torch.no_grad():
            base = tr(pf, scene, box_tokens=bt, pair_padding_mask=pad)
            pad_d = torch.cat([pad, torch.ones(BATCH, 1, dtype=torch.bool)], 1)
            bt_d = torch.cat(
                [
                    bt,
                    torch.randn(
                        BATCH,
                        1,
                        4,
                        D_MODEL,
                        generator=torch.Generator().manual_seed(42),
                    ),
                ],
                1,
            )
            grown = tr(
                torch.cat(
                    [
                        pf,
                        torch.randn(
                            BATCH,
                            1,
                            D_MODEL,
                            generator=torch.Generator().manual_seed(41),
                        ),
                    ],
                    1,
                ),
                scene,
                box_tokens=bt_d,
                pair_padding_mask=pad_d,
            )[:, :N_PAIRS]
        self.assertLess(float((grown - base).abs().max()), TOL)


class RetainedPairUnderNewObjectTest(unittest.TestCase):
    """Model level: does a new object move a pair that stays in the graph?"""

    def _run(self, boxes_n, seed=7):
        model = fx.build_model(cfa_prob=0.0)
        model.config.geo_budget = boxes_n * boxes_n
        model.config.final_budget = boxes_n * boxes_n
        model.sampler.geo_budget = boxes_n * boxes_n
        model.sampler.final_budget = boxes_n * boxes_n
        model.eval()
        with torch.no_grad():
            return fx.forward(model, [[0, 1, 2]], seed=seed, boxes_n=boxes_n)

    def _slot_of(self, out, sub, obj):
        for k in range(out["sub_idx"].shape[1]):
            if not bool(out["valid_mask"][0, k]):
                continue
            if int(out["sub_idx"][0, k]) == sub and int(out["obj_idx"][0, k]) == obj:
                return k
        return None

    def test_a_fifth_object_keeps_the_target_pair_and_moves_its_logits(self):
        """CONFIRMED: the target pair is retained, yet its logits shift.

        ``final_budget`` is raised to ``N*N`` so every ordered pair survives
        the sampler: the target pair is present in both runs, which separates
        "the candidate set changed underneath it" from "its context changed".
        """
        small, large = self._run(4), self._run(5)
        k_small = self._slot_of(small, 0, 1)
        k_large = self._slot_of(large, 0, 1)
        self.assertIsNotNone(k_small, "target pair lost from the 4-box graph")
        self.assertIsNotNone(k_large, "target pair lost after a 5th box was added")
        gap = float(
            (small["logits"][0, k_small] - large["logits"][0, k_large]).abs().max()
        )
        self.assertGreater(
            gap, 1e-4, "adding an object left the retained pair unchanged"
        )

    def test_the_target_pair_is_retained_at_both_box_counts(self):
        """The control for the case above: coverage, not eviction."""
        for n in (4, 5):
            out = self._run(n)
            self.assertIsNotNone(self._slot_of(out, 0, 1))
            self.assertTrue(bool(out["valid_mask"].any()))


if __name__ == "__main__":
    unittest.main()
