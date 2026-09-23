"""Behavioral regressions for the Phase I v2 causal contracts."""

from __future__ import annotations

import copy
import types
import unittest

from tests._optional import require_modules

require_modules("torch")

import torch  # noqa: E402

from tools.relational_emergence.v2.data import (  # noqa: E402
    build_batch,
    build_scenes,
    split_groups,
)
from tools.relational_emergence.v2.leakage import (  # noqa: E402
    AuditConfig,
    evaluate,
    group_split,
    inject_nonlinear,
    permute_labels,
    records_from_scenes,
)
from tools.relational_emergence.v2.model import (  # noqa: E402
    ModelConfig,
    build_model,
    parameter_count,
    select_batch,
    shuffle_frames,
    train_model,
)
from tools.relational_emergence.v2.protocol import MECHANISMS, Protocol  # noqa: E402
from tools.relational_emergence.v2.transplant import (  # noqa: E402
    evaluate_transplant,
    matched_sources,
    replace_query_code,
    score_predictions,
)
from tools.relational_emergence.v2.validation import evaluator_controls  # noqa: E402


class SceneFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol = Protocol(groups_per_tuple=4)
        cls.rows = build_scenes(cls.protocol)
        cls.batch, cls.targets = build_batch(cls.rows, cls.protocol)

    def model(self, arm="P-R"):
        torch.manual_seed(18)
        return build_model(arm, self.protocol, ModelConfig())


class TestData(SceneFixture):
    def test_four_real_objects_and_multiple_queried_slots(self):
        self.assertEqual(self.targets.shape[2:], (4, 4))
        self.assertGreater(len({tuple(r["query"]) for r in self.rows}), 4)
        self.assertTrue(all(r["mechanism"] in MECHANISMS for r in self.rows))

    def test_future_targets_and_metadata_cannot_reach_encoder(self):
        rows = copy.deepcopy(self.rows[:2])
        original, _ = build_batch(rows, self.protocol)
        for row in rows:
            row["mechanism"] = "SECRET"
            row["mechanism_exposed"] = "SECRET"
            for frame in row["states"][self.protocol.cutoff + 1 :]:
                for obj in frame:
                    obj[:] = [100.0] * 4
        altered, targets = build_batch(rows, self.protocol)
        self.assertTrue(torch.all(targets == 100))
        for key in original:
            self.assertTrue(torch.equal(original[key], altered[key]), key)
        self.assertNotIn("mechanism", original)
        self.assertNotIn("future", original)

    def test_past_actions_stop_before_cutoff(self):
        cut = self.protocol.cutoff
        incoming = self.batch["frames"][0, 1:, :, 2:]
        self.assertTrue(
            torch.equal(incoming, torch.tensor(self.rows[0]["actions"][:cut]))
        )
        self.assertTrue(
            torch.equal(
                self.batch["actions"][0, 0], torch.tensor(self.rows[0]["actions"][cut])
            )
        )

    def test_short_episodes_are_rejected_without_padding(self):
        row = copy.deepcopy(self.rows[0])
        row["states"].pop()
        with self.assertRaisesRegex(ValueError, "padding"):
            build_batch([row], self.protocol)

    def test_initial_snapshot_identical_mechanism_history_can_diverge(self):
        model = self.model()
        cells = {}
        for index, row in enumerate(self.rows):
            cells.setdefault((row["group"], row["regime"]), []).append(index)
        divergent = 0
        with torch.no_grad():
            initial = torch.tensor([r["states"][0] for r in self.rows])
            snapshots = model.snapshot_codes(initial, self.batch["structural"])
            history = model.infer_mechanism(self.batch)
            for indices in cells.values():
                for i in indices[1:]:
                    self.assertTrue(torch.equal(snapshots[i], snapshots[indices[0]]))
                    divergent += int(not torch.equal(history[i], history[indices[0]]))
        self.assertGreater(divergent, 0)

    def test_exposure_includes_free_when_twins_diverge(self):
        free = [r for r in self.rows if r["mechanism"] == "free"]
        self.assertTrue(any(r["mechanism_exposed"] for r in free))
        self.assertTrue(all(not any(r["mechanism_active"]) for r in free))

    def test_split_never_separates_counterfactual_base_group(self):
        split = split_groups(self.rows)
        group_sets = [
            {self.rows[i]["group"] for i in indices} for indices in split.values()
        ]
        for i, left in enumerate(group_sets):
            for right in group_sets[i + 1 :]:
                self.assertFalse(left & right)
        fit = {self.rows[i]["family"] for i in split["train"]}
        self.assertFalse(fit & {self.rows[i]["family"] for i in split["test"]})


class TestRecurrence(SceneFixture):
    def test_hand_transition_and_first_step_agree(self):
        model = self.model()

        def transition(self, current, structural, action, mechanism, **kwargs):
            return 2 * current + torch.cat([action, action], dim=-1)

        model.transition = types.MethodType(transition, model)
        batch = select_batch(self.batch, [0])
        code = model.infer_mechanism(batch)
        actual = model(batch, steps=3, mechanism=code)
        current = batch["state"]
        for step in range(3):
            current = 2 * current + torch.cat([batch["actions"][:, step]] * 2, dim=-1)
            torch.testing.assert_close(actual[:, step], current)
        torch.testing.assert_close(
            actual[:, 0], model(batch, steps=1, mechanism=code)[:, 0]
        )

    def test_earlier_action_affects_later_predictions(self):
        for arm in ("P-R", "P-G"):
            model = self.model(arm)
            batch = select_batch(self.batch, [0])
            modified = {k: v.clone() for k, v in batch.items()}
            modified["actions"][:, 0, 0, 0] += 5
            with torch.no_grad():
                code = model.infer_mechanism(batch)
                before, after = (
                    model(batch, mechanism=code),
                    model(modified, mechanism=code),
                )
            self.assertTrue(
                torch.all((before - after).flatten(2).abs().sum(2) > 0), arm
            )

    def test_predicted_state_perturbation_propagates(self):
        model = self.model()
        batch = select_batch(self.batch, [0])
        code = model.infer_mechanism(batch)
        expected = model(batch, steps=3, mechanism=code)
        original = model.transition
        calls = []

        def perturbed(self, *args, **kwargs):
            result = original(*args, **kwargs)
            calls.append(result)
            return result + 1 if len(calls) == 1 else result

        model.transition = types.MethodType(perturbed, model)
        actual = model(batch, steps=3, mechanism=code)
        for step in (1, 2):
            self.assertFalse(torch.equal(expected[:, step], actual[:, step]))

    def test_training_and_evaluation_use_identical_recurrence(self):
        model = self.model()
        batch = select_batch(self.batch, [0, 1])
        model.train()
        training = model(batch)
        model.eval()
        torch.testing.assert_close(training, model(batch))

    def test_cross_object_influence_has_no_edge_bypass(self):
        model = self.model()
        batch = select_batch(self.batch, [0])
        other = {k: v.clone() for k, v in batch.items()}
        other["state"][:, 1] += 10
        other["structural"][:, 1] += 10
        other["frames"][:, :, 1] += 10
        other["actions"][:, :, 1] += 10
        with torch.no_grad():
            torch.testing.assert_close(
                model(batch, block_edges=True)[:, :, 0],
                model(other, block_edges=True)[:, :, 0],
            )
            self.assertFalse(torch.equal(model(batch)[:, :, 0], model(other)[:, :, 0]))

    def test_global_has_one_scene_code_and_matched_capacity(self):
        global_model, relational = self.model("P-G"), self.model()
        self.assertFalse(hasattr(global_model, "state_encoder"))
        self.assertFalse(hasattr(global_model, "interaction_update"))
        self.assertEqual(
            global_model.infer_mechanism(select_batch(self.batch, [0])).ndim, 2
        )
        self.assertLess(
            abs(parameter_count(global_model) - parameter_count(relational))
            / parameter_count(relational),
            0.01,
        )

    def test_train_checkpoint_reports_prediction_only(self):
        _, result = train_model(
            "P-R",
            self.protocol,
            ModelConfig(epochs=2),
            self.batch,
            self.targets,
            split_groups(self.rows),
            11,
        )
        self.assertTrue(
            torch.isfinite(torch.tensor(result["validation_prediction_loss"]))
        )
        self.assertIn(result["best_epoch"], (0, 1))


class TestControls(SceneFixture):
    def test_static_codes_ignore_velocity_and_other_snapshots(self):
        for arm in ("InitialStatic-R", "PostState-R"):
            model = self.model(arm)
            batch = select_batch(self.batch, [0, 1])
            changed = {k: v.clone() for k, v in batch.items()}
            changed["frames"] += 100
            changed["state"][:, :, 2:] += 100
            changed["actions"] += 100
            unused = (
                "post_positions" if arm == "InitialStatic-R" else "initial_positions"
            )
            changed[unused] += 100
            torch.testing.assert_close(
                model.infer_mechanism(batch), model.infer_mechanism(changed)
            )

    def test_shuffle_preserves_multiset_and_varies_per_episode(self):
        frames = torch.arange(11).view(1, 11, 1, 1).expand(256, 11, 4, 4).float()
        shuffled, order = shuffle_frames(frames, torch.Generator().manual_seed(7))
        torch.testing.assert_close(
            shuffled[:, :, 0, 0].sort(1).values, frames[:, :, 0, 0]
        )
        self.assertGreater(len(set(tuple(r) for r in order.tolist())), 200)
        # No fixed column reliably identifies any one original position.
        for column in range(11):
            self.assertLess(int(torch.bincount(order[:, column]).max()), 50)

    def test_shuffle_applies_in_eval_as_well_as_train(self):
        model = self.model("Shuffle-R")
        model.eval()
        batch = select_batch(self.batch, [0, 1, 2])
        a = model.observed_frames(batch, torch.Generator().manual_seed(1))
        b = model.observed_frames(batch, torch.Generator().manual_seed(2))
        self.assertFalse(torch.equal(a, b))

    def test_known_answer_evaluators_pass(self):
        controls = evaluator_controls()
        self.assertEqual(controls["status"], "PASS")
        self.assertEqual(controls["real_decoder_calibration"], "PENDING")


class TestLeakage(SceneFixture):
    def test_balanced_identical_observables_cannot_beat_compatibility_ce(self):
        # Each context has every compatible label exactly once. Jensen's
        # inequality bounds any context-only classifier at log(K), regardless
        # of training. Float32 accumulation previously produced a false p=.03.
        protocol = Protocol()
        records = records_from_scenes(build_scenes(protocol))
        result = evaluate(records, protocol, AuditConfig())
        self.assertFalse(result["detected"])
        self.assertLessEqual(result["max_delta_ce"], 1e-12)

    def test_all_six_feature_sets_and_both_trained_estimators(self):
        records = records_from_scenes(self.rows)
        result = evaluate(records, self.protocol, AuditConfig(epochs=2, permutations=3))
        self.assertEqual(len(result["probes"]), 12)
        self.assertTrue(
            all("p_fwer" in r and "delta_ce" in r for r in result["probes"].values())
        )

    def test_split_and_permutation_keep_group_identity_and_compatibility(self):
        records = records_from_scenes(self.rows)
        train, test = group_split(records, 3)
        self.assertFalse(
            {records[i]["group"] for i in train} & {records[i]["group"] for i in test}
        )
        duplicated = [*records, *records]
        labels = permute_labels(duplicated, 12).tolist()
        self.assertEqual(labels[: len(records)], labels[len(records) :])
        for record, label in zip(duplicated, labels, strict=True):
            self.assertIn(MECHANISMS[label], record["stratum"])

    def test_nonlinear_control_keeps_initial_state_and_base_groups(self):
        records = records_from_scenes(self.rows)
        injected = inject_nonlinear(records, 0.2, 7)
        for original, changed in zip(records, injected, strict=True):
            self.assertEqual(original["group"], changed["group"])
            self.assertEqual(original["x0"], changed["x0"])
        self.assertTrue(any(any(r["filler"][-4:]) for r in injected))


class TestTransplant(SceneFixture):
    def test_entire_transplant_path_fires_for_true_code_and_not_alias(self):
        rows = [
            {
                "group": f"group{g}",
                "tuple": "fixture",
                "regime": "rich",
                "mechanism": mechanism,
                "compatible": list(MECHANISMS),
                "filler": g // 2,
                "query": [0, 3],
                "mechanism_exposed": True,
            }
            for g in range(8)
            for mechanism in MECHANISMS
        ]
        codes = torch.eye(4).repeat(8, 1)
        batch = {"code": codes, "state": torch.zeros(32, 4, 4)}

        class Fixture(torch.nn.Module):
            pairwise = False

            def __init__(self, alias=False):
                super().__init__()
                self.alias = alias

            def infer_mechanism(self, part, generator=None):
                return torch.zeros_like(part["code"]) if self.alias else part["code"]

            def forward(self, part, mechanism=None):
                code = part["code"] if mechanism is None else mechanism
                velocity = 1 + code @ torch.arange(4).float()
                return torch.stack(
                    [
                        part["state"] + (t + 1) * velocity[:, None, None]
                        for t in range(3)
                    ],
                    dim=1,
                )

        targets = Fixture()(batch)
        for level in ("within_filler", "cross_filler"):
            positive = evaluate_transplant(
                Fixture(), rows, batch, targets, level=level, scale=torch.ones(4)
            )
            negative = evaluate_transplant(
                Fixture(alias=True),
                rows,
                batch,
                targets,
                level=level,
                scale=torch.ones(4),
            )
            self.assertGreater(positive["primary"]["ci_low"], 0)
            self.assertEqual(negative["primary"]["gain_wrong_minus_correct"], 0)
            self.assertEqual(negative["unscorable_pair_rate"], 1)

    def test_source_twins_are_aligned_and_context_dyads_disjoint(self):
        for level in ("within_filler", "cross_filler"):
            rows = self.rows
            pairs, _ = matched_sources(rows, level)
            self.assertTrue(pairs)
            used = {}
            for target, correct, wrong, cluster in pairs:
                a, b, c = rows[target], rows[correct], rows[wrong]
                self.assertEqual(b["group"], c["group"])
                self.assertEqual(a["mechanism"], b["mechanism"])
                self.assertNotEqual(b["mechanism"], c["mechanism"])
                self.assertEqual(a["filler"] == b["filler"], level == "within_filler")
                for row in (a, b):
                    self.assertEqual(used.setdefault(row["group"], cluster), cluster)

    def test_only_query_edges_are_transplanted_with_role_alignment(self):
        model = self.model()
        code = torch.zeros(1, 12, 16)
        source = torch.arange(12).view(1, 12, 1).expand_as(code).float()
        changed = replace_query_code(
            model, code, source, {"query": [0, 3]}, {"query": [2, 1]}
        )
        self.assertTrue(
            torch.all(
                changed[:, model.edges.index((0, 3))] == model.edges.index((2, 1))
            )
        )
        self.assertTrue(
            torch.all(
                changed[:, model.edges.index((3, 0))] == model.edges.index((1, 2))
            )
        )
        self.assertEqual(int((changed.sum(-1) != 0).sum()), 2)

    def test_zero_sensitivity_is_scored_and_reported(self):
        result = score_predictions(
            {arm: [1.0] * 3 for arm in ("self", "correct", "wrong")},
            ["a", "b", "c"],
            [0.0] * 3,
            [True, False, True],
            unpaired_groups=[],
        )
        self.assertEqual(result["primary"]["gain_wrong_minus_correct"], 0)
        self.assertEqual(result["unscorable_pair_rate"], 1)
        self.assertEqual(result["primary"]["n_pairs"], 3)
        self.assertEqual(result["off_manifold_exclusions"], 0)


if __name__ == "__main__":
    unittest.main()
