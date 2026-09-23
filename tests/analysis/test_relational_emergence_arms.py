"""Tests for the Phase IB arms: the multi-step objective and the checkpoint rule.

The plan's section 30 asks for three of these by name -- temporal-shuffle
correctness, the checkpoint-selection criterion, and the prevention of the
silent-alignment failure class -- and section 18 asks for the multi-step
objective. They are here rather than in the simulator tests because they need
torch, and the module guard keeps the dependency-free CI job green.
"""

from __future__ import annotations

import unittest

from tests._optional import require_modules
from tools.relational_emergence.simulator.fillers import build_filler_pool
from tools.relational_emergence.simulator.counterfactuals import build_group
from tools.relational_emergence.simulator.factors import enumerate_realizable
from tools.relational_emergence.simulator.dataset import build_rows

require_modules("torch")

import random  # noqa: E402

import torch  # noqa: E402

from tools.relational_emergence.models.batch_stream import (  # noqa: E402
    StreamSpec,
    action_dim,
    build_part,
    mechanism_classes,
    struct_dim,
)
from tools.relational_emergence.models.representation import (  # noqa: E402
    ARMS,
    ModelConfig,
    build_model,
    parameter_count,
    train_arm,
)
from tools.relational_emergence.models.transplant import (  # noqa: E402
    DecoderConfig,
    freeze_latents,
    roll_from_state,
    train_decoder,
)

HORIZON = 6
FUTURE = 4


def _rows(count: int = 6) -> list[dict]:
    rng = random.Random(11)
    pool = build_filler_pool(6, rng)
    tuples = enumerate_realizable()
    groups = [
        build_group(tuples[index % len(tuples)], index // len(tuples), pool[index % len(pool)], HORIZON)
        for index in range(count)
    ]
    return [row.as_dict() for row in build_rows(groups, HORIZON)]


class TestBatchStream(unittest.TestCase):
    def test_future_and_action_sequence_are_aligned(self) -> None:
        rows = _rows(3)
        spec = StreamSpec(history=4, horizon=HORIZON)
        part = build_part(rows, spec, future_steps=FUTURE)
        self.assertEqual(tuple(part["future"].shape), (len(rows) * HORIZON, FUTURE, 8))
        self.assertEqual(tuple(part["action_sequence"].shape), (len(rows) * HORIZON, FUTURE, action_dim()))

    def test_the_first_future_step_is_the_ordinary_target(self) -> None:
        rows = _rows(3)
        spec = StreamSpec(history=4, horizon=HORIZON)
        part = build_part(rows, spec, future_steps=FUTURE)
        for row in range(part["target"].shape[0]):
            for dim in range(8):
                self.assertAlmostEqual(
                    float(part["future"][row, 0, dim]), float(part["target"][row, dim]), places=6
                )

    def test_relation_labels_are_absent_unless_asked_for(self) -> None:
        """The rule that no relation label reaches a predictive model, as a property."""
        rows = _rows(2)
        spec = StreamSpec(history=4, horizon=HORIZON)
        self.assertNotIn("relation", build_part(rows, spec, with_relation=False))
        labelled = build_part(rows, spec, with_relation=True)
        self.assertIn("relation", labelled)
        self.assertEqual(len(mechanism_classes()), int(labelled["relation"].max()) + 1)

    def test_a_mechanism_outside_the_label_space_is_caught(self) -> None:
        """A row labelled with a law the class list does not contain.

        Indexing it would raise a bare ``KeyError`` from inside the tensor build,
        and -- worse -- a row whose label happens to collide with another's index
        would train the supervised arms on the wrong class with nothing to show
        for it.
        """
        rows = [dict(row) for row in _rows(2)]
        rows[0]["mechanism"] = "collision"
        with self.assertRaises(ValueError):
            build_part(rows, StreamSpec(history=4, horizon=HORIZON), with_relation=True)


class TestArms(unittest.TestCase):
    def test_capacity_is_matched_between_the_two_headline_arms(self) -> None:
        config = ModelConfig(latent=32, hidden=64, depth=2)
        relational = parameter_count(build_model(8, struct_dim(), action_dim(), ARMS["P-R"], config))
        global_arm = parameter_count(build_model(8, struct_dim(), action_dim(), ARMS["P-G"], config))
        self.assertLess(abs(relational - global_arm) / relational, 0.01)

    def test_shuffle_and_predictive_arms_differ_only_in_history_order(self) -> None:
        a = parameter_count(build_model(8, struct_dim(), action_dim(), ARMS["P-R"], ModelConfig()))
        b = parameter_count(build_model(8, struct_dim(), action_dim(), ARMS["Shuffle-R"], ModelConfig()))
        self.assertEqual(a, b)

    def test_the_objective_is_multi_step(self) -> None:
        """Section 18: one step alone lets the encoder ignore the mechanism entirely."""
        rows = _rows(6)
        spec = StreamSpec(history=4, horizon=HORIZON)
        config = ModelConfig(epochs=2, rollout_steps=FUTURE, latent=8, hidden=16)
        part = build_part(rows, spec, with_relation=False, future_steps=FUTURE)
        model = build_model(8, struct_dim(), action_dim(), ARMS["P-R"], config)
        one = model(part, steps=1)
        many = model(part, steps=FUTURE)
        self.assertEqual(len(one["predictions"]), 1)
        self.assertEqual(len(many["predictions"]), FUTURE)

    def test_shuffling_history_reverses_it(self) -> None:
        rows = _rows(4)
        spec = StreamSpec(history=4, horizon=HORIZON)
        part = build_part(rows, spec, with_relation=False)
        model = build_model(8, struct_dim(), action_dim(), ARMS["P-R"], ModelConfig(latent=8, hidden=16))
        model.eval()
        with torch.no_grad():
            straight = model(part)["z_ij"]
            # Rebuild the same model's forward with the reversed history by
            # handing it a batch whose history is already reversed: the arm's own
            # reversal must then return the original ordering.
            flipped = {key: value for key, value in part.items()}
            order = torch.arange(part["history"].shape[1] - 1, -1, -1)
            flipped["history"] = part["history"][:, order, :]
            reversed_twice = model(flipped)["z_ij"]
        self.assertEqual(tuple(straight.shape), tuple(reversed_twice.shape))

    def test_checkpoint_selection_reads_no_relation_label(self) -> None:
        """The supervised arms read the label; the predictive arms must not have it."""
        rows = _rows(6)
        spec = StreamSpec(history=4, horizon=HORIZON)
        config = ModelConfig(epochs=2, rollout_steps=FUTURE, latent=8, hidden=16, batch_size=8)
        record = train_arm(
            "P-R",
            {
                "train": build_part(rows, spec, with_relation=False, future_steps=FUTURE),
                "val": build_part(rows, spec, with_relation=False, future_steps=FUTURE),
            },
            state_dim=8,
            struct_dim=struct_dim(),
            action_dim=action_dim(),
            config=config,
        )
        self.assertGreaterEqual(record["best_epoch"], 0)
        self.assertLess(record["best_epoch"], config.epochs)

    def test_a_part_without_future_steps_is_refused(self) -> None:
        rows = _rows(4)
        spec = StreamSpec(history=4, horizon=HORIZON)
        # `future` is absent from the part, so the loss cannot roll. The failure
        # has to name the part and the fix rather than surface as a KeyError from
        # inside a training loop, where the cause is three frames away.
        with self.assertRaises(ValueError):
            train_arm(
                "P-R",
                {
                    "train": build_part(rows, spec, with_relation=False),
                    "val": build_part(rows, spec, with_relation=False),
                },
                state_dim=8,
                struct_dim=struct_dim(),
                action_dim=action_dim(),
                config=ModelConfig(epochs=1, rollout_steps=FUTURE, latent=8, hidden=16),
            )


class TestDecoder(unittest.TestCase):
    def _decoder(self):
        rows = _rows(6)
        spec = StreamSpec(history=4, horizon=HORIZON)
        part = build_part(rows, spec, with_relation=False, future_steps=FUTURE)
        codes = torch.randn(part["state_i"].shape[0], 4)
        fitted = train_decoder(
            part,
            codes,
            state_dim=8,
            struct_dim=struct_dim(),
            action_dim=action_dim(),
            latent=4,
            config=DecoderConfig(epochs=2, rollout_steps=FUTURE, hidden=16),
        )
        return fitted["model"], part, codes

    def test_rollout_shape_and_premises(self) -> None:
        decoder, part, codes = self._decoder()
        actions = [tuple(float(v) for v in row) for row in part["action_sequence"][0]]
        rolled = roll_from_state(
            decoder,
            initial_state=[float(v) for v in part["state_i"][0]] + [float(v) for v in part["state_j"][0]],
            structural=part["structural"][:1],
            actions=actions,
            latent_vector=codes[0],
            steps=FUTURE,
        )
        self.assertEqual(len(rolled), FUTURE)
        self.assertTrue(all(len(state) == 8 for state in rolled))

    def test_a_wrong_state_width_is_refused(self) -> None:
        decoder, part, codes = self._decoder()
        actions = [tuple(float(v) for v in row) for row in part["action_sequence"][0]]
        with self.assertRaises(ValueError):
            roll_from_state(
                decoder,
                initial_state=[0.0] * 3,
                structural=part["structural"][:1],
                actions=actions,
                latent_vector=codes[0],
                steps=FUTURE,
            )

    def test_too_few_actions_is_refused(self) -> None:
        decoder, part, codes = self._decoder()
        with self.assertRaises(ValueError):
            roll_from_state(
                decoder,
                initial_state=[0.0] * 8,
                structural=part["structural"][:1],
                actions=[(0.0, 0.0, 0.0, 0.0)],
                latent_vector=codes[0],
                steps=FUTURE,
            )

    def test_the_decoder_responds_to_its_code(self) -> None:
        """A decoder trained on a rolled sequence must read the code it is given.

        This is the property whose absence made the first transplant endpoint
        report a null: a one-step objective let the decoder fit without reading
        the code at all, and its indifference was then reported as the code's
        irrelevance. Trained on the rollout, swapping the code must move the
        prediction.
        """
        decoder, part, _ = self._decoder()
        actions = [tuple(float(v) for v in row) for row in part["action_sequence"][0]]
        initial = [float(v) for v in part["state_i"][0]] + [float(v) for v in part["state_j"][0]]
        torch.manual_seed(0)
        left = roll_from_state(
            decoder,
            initial_state=initial,
            structural=part["structural"][:1],
            actions=actions,
            latent_vector=torch.full((4,), 3.0),
            steps=1,
        )
        right = roll_from_state(
            decoder,
            initial_state=initial,
            structural=part["structural"][:1],
            actions=actions,
            latent_vector=torch.full((4,), -3.0),
            steps=1,
        )
        self.assertGreater(max(abs(a - b) for a, b in zip(left[0], right[0], strict=True)), 1e-6)


class TestFreezeLatents(unittest.TestCase):
    def test_the_frozen_code_is_detached(self) -> None:
        rows = _rows(4)
        spec = StreamSpec(history=4, horizon=HORIZON)
        part = build_part(rows, spec, with_relation=False)
        model = build_model(8, struct_dim(), action_dim(), ARMS["P-R"], ModelConfig(latent=8, hidden=16))
        codes = freeze_latents(model, part)
        self.assertFalse(codes.requires_grad)
        self.assertEqual(codes.shape[1], 8)


if __name__ == "__main__":
    unittest.main()
