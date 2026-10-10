"""#155 RED: is CFA partner selection annotation-order dependent?

Multi-positive ordered pairs are reduced to a single predicate by
``RelatednessPairSampler._gt_grid`` (``scatter_``, last one wins), and that
single value drives ``RelSGG._mix_partners`` for same-predicate feature
augmentation. This module executes the pinned source to decide whether that
reduction makes the training forward pass depend on the *order* of relation
rows, for a semantically identical annotation set.

Evidence class: **executed source behaviour on synthetic fixtures**. There is
no checkpoint, no dataset and no metric here, so nothing below is baseline
evidence or a performance claim. See the audit note in ``reproduction/``.

The suite is split by the role each case plays in the audit:

* invariant cases -- contracts that *should* hold; a failure is a real defect;
* control cases    -- establish that the harness isolates CFA;
* characterisation -- pins behaviour that was *confirmed* to be order
  dependent, so the finding goes stale loudly if upstream changes it.
"""

import unittest

from tests._optional import require_modules

require_modules("torch", "torchvision")

import torch  # noqa: E402

from tests.reproduction import _relateanything_audit_fixtures as fx  # noqa: E402

# (0,1) carries TWO positives {2,7}; (2,3) carries {2}; (0,2) carries {7}.
# Order A puts 2 last for (0,1); order B puts 7 last. Same labelled set.
ORDER_A = [[0, 1, 7], [0, 1, 2], [2, 3, 2], [0, 2, 7]]
ORDER_B = [[0, 1, 2], [0, 1, 7], [2, 3, 2], [0, 2, 7]]

# Negative-control fixture: every pair has exactly one predicate, so row order
# cannot change anything.
SINGLE_A = [[0, 1, 2], [2, 3, 7], [0, 2, 5]]
SINGLE_B = [[2, 3, 7], [0, 2, 5], [0, 1, 2]]


class CfaAnnotationOrderTest(unittest.TestCase):
    """Executed behaviour of the pinned sampler and head under row permutation."""

    # -- invariants: these must hold -----------------------------------

    def test_slot_targets_keep_every_positive_under_permutation(self):
        """`build_slot_targets` is set-valued, so row order must not matter."""
        from src.modules.relateanything.training.losses import build_slot_targets

        sub = torch.tensor([[0, 2, 0, 0]])
        obj = torch.tensor([[1, 3, 2, 1]])
        valid = torch.tensor([[True, True, True, True]])

        def targets(rows):
            return [{"relations": torch.as_tensor(rows, dtype=torch.long)}]

        hot_a, _ = build_slot_targets(sub, obj, valid, targets(ORDER_A), 8)
        hot_b, _ = build_slot_targets(sub, obj, valid, targets(ORDER_B), 8)
        self.assertTrue(
            torch.equal(hot_a, hot_b),
            "slot targets changed under a pure row permutation",
        )
        # The multi-positive pair keeps *both* predicates, not just the last.
        self.assertTrue(bool(hot_a[0, 0, 2]) and bool(hot_a[0, 0, 7]))

    def test_duplicate_relation_rows_leave_slot_targets_unchanged(self):
        """Edge case: an exactly duplicated row is idempotent for multi-hot."""
        from src.modules.relateanything.training.losses import build_slot_targets

        sub, obj, valid = (
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            torch.tensor([[True]]),
        )
        once, _ = build_slot_targets(
            sub, obj, valid, [{"relations": torch.tensor([[0, 1, 2]])}], 8
        )
        twice, _ = build_slot_targets(
            sub, obj, valid, [{"relations": torch.tensor([[0, 1, 2], [0, 1, 2]])}], 8
        )
        self.assertTrue(torch.equal(once, twice))

    def test_empty_relations_produce_no_targets_and_no_predicate(self):
        """Edge case: a pair list with no rows must not invent a label."""
        from src.modules.relateanything.training.losses import build_slot_targets

        empty = [{"relations": torch.zeros(0, 3, dtype=torch.long)}]
        sub, obj, valid = (
            torch.tensor([[0]]),
            torch.tensor([[1]]),
            torch.tensor([[True]]),
        )
        hot, _ = build_slot_targets(sub, obj, valid, empty, 8)
        self.assertFalse(bool(hot.any()))
        self.assertTrue(bool((fx.gt_pred_labels([]) < 0).all()))

    def test_single_positive_pairs_are_order_invariant(self):
        """Negative control: with one predicate per pair, order cannot bite."""
        self.assertTrue(
            torch.equal(fx.gt_pred_labels(SINGLE_A), fx.gt_pred_labels(SINGLE_B))
        )
        self.assertEqual(fx.eligible_partners(SINGLE_A), {})

    # -- harness controls ----------------------------------------------

    def test_cfa_disabled_makes_the_training_forward_order_invariant(self):
        """With CFA off the two orders must agree to the last bit.

        This is the control that attributes any difference seen with CFA on to
        CFA alone: no other part of the training forward pass (sampler losses,
        slot-target lookup, loss reduction) is order sensitive.
        """
        model = fx.build_model(cfa_prob=0.0)
        a = fx.forward(model, ORDER_A)
        b = fx.forward(model, ORDER_B)
        self.assertEqual(float((a["logits"] - b["logits"]).abs().max()), 0.0)
        self.assertEqual(
            float((a["pair_features"] - b["pair_features"]).abs().max()), 0.0
        )

    # -- characterisation: the confirmed finding ------------------------

    def test_gt_grid_reduces_a_multi_positive_pair_to_the_last_row(self):
        """Source-confirmed: `_gt_grid` is not set-valued for duplicates."""
        # (0,1) is slot 0*4+1 = 1.
        self.assertEqual(int(fx.gt_pred_labels(ORDER_A)[1]), 2)  # 2 scattered last
        self.assertEqual(int(fx.gt_pred_labels(ORDER_B)[1]), 7)  # 7 scattered last

    def test_cfa_partner_selection_follows_the_annotation_order(self):
        """CONFIRMED: the same label set yields different augmentation partners.

        In both orders the multi-positive pair (0,1) ends up in a
        partner-eligible group of the same size, so the augmentation *budget*
        is unchanged -- only *which* slot it is mixed with changes.
        """
        a, b = fx.eligible_partners(ORDER_A), fx.eligible_partners(ORDER_B)
        # Order A: (0,1) is tagged 2, so it partners with the slot that has 2.
        self.assertEqual(a, {2: [(0, 1), (2, 3)]})
        # Order B: (0,1) is tagged 7, so it partners with the slot that has 7.
        self.assertEqual(b, {7: [(0, 1), (0, 2)]})
        self.assertNotEqual(a, b)

    def test_cfa_enabled_makes_the_training_forward_order_dependent(self):
        """CONFIRMED: the difference reaches the logits of the model."""
        model = fx.build_model(cfa_prob=1.0)
        a = fx.forward(model, ORDER_A)
        b = fx.forward(model, ORDER_B)
        logit_gap = float((a["logits"] - b["logits"]).abs().max())
        feat_gap = float((a["pair_features"] - b["pair_features"]).abs().max())
        self.assertGreater(
            logit_gap, 1e-3, "expected the CFA partner swap to move the logits"
        )
        self.assertGreater(feat_gap, 1e-3)
        # The gap is a training-protocol effect, not a numerical artefact: it
        # is many orders of magnitude above the float noise floor measured by
        # the cfa_prob=0 control (exactly 0.0) .
        self.assertLess(logit_gap, 1.0)

    def test_cfa_keeps_the_augmented_slot_in_a_group_of_the_same_size(self):
        """The augmentation budget is order invariant even though partners are not."""
        sizes_a = sorted(len(v) for v in fx.eligible_partners(ORDER_A).values())
        sizes_b = sorted(len(v) for v in fx.eligible_partners(ORDER_B).values())
        self.assertEqual(sizes_a, sizes_b)

    def test_order_dependence_survives_the_released_cfa_hyperparameters(self):
        """CONFIRMED under the shipped recipe, not just the deterministic one.

        The cases above pin ``cfa_alpha`` high so that the mixing coefficient is
        deterministic and the partner swap is the only variable. The released
        config is ``cfa_prob=0.5, cfa_alpha=1.0``
        (``src/modules/relateanything/config.py``), where the coefficient is
        drawn per step; the partner *assignment* is unaffected by either, so the
        order dependence must persist. Checked over several seeds because the
        mixing draw is now stochastic.
        """
        self.assertEqual(fx.tiny_config().cfa_alpha, 1e6)  # harness override
        from src.modules.relateanything.config import RelSGGConfig

        released = RelSGGConfig()
        self.assertEqual((released.cfa_prob, released.cfa_alpha), (0.5, 1.0))

        moved = 0
        for seed in (7, 11, 23):
            model_a = fx.build_model(cfa_prob=0.5, cfa_alpha=1.0)
            model_b = fx.build_model(cfa_prob=0.5, cfa_alpha=1.0)
            gap = float(
                (
                    fx.forward(model_a, ORDER_A, seed=seed)["logits"]
                    - fx.forward(model_b, ORDER_B, seed=seed)["logits"]
                )
                .abs()
                .max()
            )
            if gap > 1e-6:
                moved += 1
        self.assertGreater(
            moved, 0, "released CFA settings produced no order dependence"
        )

    def test_single_positive_fixture_stays_order_invariant_with_cfa_on(self):
        """Negative control with CFA active: nothing to mix, nothing changes."""
        model = fx.build_model(cfa_prob=1.0)
        a = fx.forward(model, SINGLE_A)
        b = fx.forward(model, SINGLE_B)
        self.assertEqual(float((a["logits"] - b["logits"]).abs().max()), 0.0)

    # -- the training signal itself ------------------------------------

    def test_loss_and_gradients_are_order_invariant_with_cfa_off(self):
        """Control for the two cases below: no CFA, no difference anywhere.

        Bit-exact agreement on the loss *and* on every parameter gradient is
        what licenses attributing the differences found with CFA on to CFA,
        rather than to some other order-sensitive path in the step.
        """
        model = fx.build_model(cfa_prob=0.0)
        loss_a, norms_a, grad_a = fx.loss_and_grad(model, ORDER_A)
        loss_b, norms_b, grad_b = fx.loss_and_grad(model, ORDER_B)
        self.assertEqual(loss_a, loss_b)
        self.assertEqual(norms_a, norms_b)
        self.assertEqual(float((grad_a - grad_b).abs().max()), 0.0)

    def test_loss_and_gradients_differ_between_orders_when_cfa_is_on(self):
        """CONFIRMED: the swap reaches the loss and the gradient, not only the logits.

        This is the strongest statement this audit supports: a semantically
        identical annotation set, differing only in row order, produces a
        different training signal under the released augmentation. It is still
        a statement about the *optimisation step*, not about model quality --
        nothing here measures whether either gradient is better, and no
        checkpoint or benchmark was involved.
        """
        model = fx.build_model(cfa_prob=1.0)
        loss_a, norms_a, grad_a = fx.loss_and_grad(model, ORDER_A)
        loss_b, norms_b, grad_b = fx.loss_and_grad(model, ORDER_B)
        self.assertGreater(abs(loss_a - loss_b), 1e-9, "loss unchanged between orders")
        self.assertGreater(
            float((grad_a - grad_b).abs().max()),
            1e-9,
            "gradients unchanged between orders",
        )
        # At least one parameter block must actually move.
        moved = [n for n in norms_a if abs(norms_a[n] - norms_b[n]) > 1e-9]
        self.assertTrue(moved, "no parameter gradient norm changed")
        # Sanity: both are finite, real training signals.
        self.assertTrue(all(v == v and abs(v) < 1e6 for v in norms_a.values()))

    def test_gradient_difference_is_concentrated_where_cfa_acts(self):
        """Where the gradients move: the pair and query projections, not the vocabulary.

        ``_mix_entities`` rewrites ``v_sub`` / ``v_obj_k`` before they are
        projected, so the terms fed by those features should move while the
        vocabulary matrix, which is an input rather than a parameter here,
        should not.
        """
        model = fx.build_model(cfa_prob=1.0)
        _, norms_a, _ = fx.loss_and_grad(model, ORDER_A)
        _, norms_b, _ = fx.loss_and_grad(model, ORDER_B)
        moved = {n for n in norms_a if abs(norms_a[n] - norms_b[n]) > 1e-9}
        self.assertTrue(
            any("pair_proj" in n for n in moved),
            f"pair_proj did not move; moved={sorted(moved)}",
        )
        self.assertTrue(
            any("sub_text_proj" in n or "obj_text_proj" in n for n in moved),
            f"text projections did not move; moved={sorted(moved)}",
        )


if __name__ == "__main__":
    unittest.main()
