"""Behavior checks for the RelateAnything preflight CLI; fixtures are not weights."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

CHECKER = Path(__file__).resolve().parents[2] / "tools/reproduction/check_relateanything_official_inputs.py"


class RelateAnythingPreflightTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_check(self, *extra):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--official-root", str(self.root / "official"),
             "--snapshot", str(self.root / "snapshot"), "--data-root", str(self.root / "vg150"), *extra],
            cwd=self.root, capture_output=True, text=True,
        )
        self.assertIn(result.returncode, (0, 2), result.stderr)
        self.assertTrue(result.stdout.strip(), result.stderr)
        return result.returncode, json.loads(result.stdout)

    def test_missing_assets_block_evaluation_with_actionable_paths(self):
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertFalse(report["can_evaluate"])
        self.assertIn(str(self.root / "snapshot/model.pth"), report["missing_paths"])
        self.assertIn(str(self.root / "vg150/test/meta.json"), report["missing_paths"])

    def test_student_is_resolved_from_snapshot_even_with_cwd_decoy(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "text_student.pt").write_bytes(b"snapshot student")
        (self.root / "text_student.pt").write_bytes(b"wrong student")
        _, report = self.run_check()
        self.assertEqual(report["text_student"], str(snapshot / "text_student.pt"))
        self.assertIn(str(snapshot / "text_student.pt"), report["reference_command"])

    def test_a3_reports_released_embedding_fields_incompatible_with_pinned_runner(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        # ZIP member names suffice for this layout check; not inference inputs.
        with zipfile.ZipFile(snapshot / "predicate_embeddings.npz", "w") as archive:
            archive.writestr("names.npy", b"fixture")
            archive.writestr("W.npy", b"fixture")
        code, report = self.run_check("--protocol", "A3")
        self.assertEqual(code, 2)
        self.assertIn("a3_embedding_layout_incompatible", report["blockers"])
        self.assertEqual(report["embedding_fields"], ["W", "names"])

    def test_a3_rejects_calibration_from_another_text_space(self):
        calibration = self.root / "tau.json"
        calibration.write_text(json.dumps({"pred_embeds": "other.npz", "chosen": {"tau": 0.72}}))
        code, report = self.run_check("--protocol", "A3", "--tau-calibration", str(calibration))
        self.assertEqual(code, 2)
        self.assertIn("a3_calibration_text_space_mismatch", report["blockers"])

    def test_existing_assets_do_not_replace_strict_load_and_full_split_evidence(self):
        source_files = ("benchmark/eval_zeroshot.py", "relsgg/checkpoint.py", "relsgg/eval/evaluator.py",
                        "relsgg/data/dataset.py", "pyproject.toml", "LICENSE", "NOTICE")
        for name in source_files:
            path = self.root / "official" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture source")
        for name in ("model.pth", "text_student.pt", "tokenizer.json", "tokenizer_config.json"):
            path = self.root / "snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture asset")
        for name in ("meta.json", "file_names.json", "img_meta.npy", "boxes.npy", "box_cats.npy", "rels.npy"):
            path = self.root / "vg150/test" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture pack")
        code, report = self.run_check()
        self.assertEqual(code, 2)
        self.assertEqual(report["missing_paths"], [])
        self.assertIn("strict_load_unverified", report["blockers"])
        self.assertIn("full_split_unverified", report["blockers"])
        self.assertFalse(report["reproduction_ready"])

    def test_load_evidence_for_different_weights_is_rejected(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "model.pth").write_bytes(b"new weights")
        (snapshot / "text_student.pt").write_bytes(b"student")
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"strict_load": {
            "strict": True, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [],
            "model_sha256": hashlib.sha256(b"old weights").hexdigest(),
            "text_student_sha256": hashlib.sha256(b"student").hexdigest(),
        }}))
        _, report = self.run_check("--evidence", str(evidence))
        self.assertIn("strict_load_artifact_mismatch", report["blockers"])

    def test_full_split_evidence_rejects_a_tiny_slice(self):
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"full_split": {
            "dataset": "vg150", "split": "test", "expected_images": 26404,
            "loaded_images": 8, "valid_input_images": 8, "failed_images": 0,
            "limit": 8, "pack_sha256": {},
        }}))
        _, report = self.run_check("--evidence", str(evidence))
        self.assertIn("full_split_count_mismatch", report["blockers"])
        self.assertFalse(report["can_evaluate"])

    def test_clean_load_evidence_is_bound_to_the_official_source_files(self):
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "model.pth").write_bytes(b"weights")
        (snapshot / "text_student.pt").write_bytes(b"student")
        evidence = self.root / "evidence.json"
        evidence.write_text(json.dumps({"strict_load": {
            "strict": True, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [],
            "model_sha256": hashlib.sha256(b"weights").hexdigest(),
            "text_student_sha256": hashlib.sha256(b"student").hexdigest(),
            "source_sha256": {},
        }}))
        _, report = self.run_check("--evidence", str(evidence))
        self.assertIn("strict_load_source_mismatch", report["blockers"])
        self.assertIn("strict_load_unverified", report["blockers"])

    def test_complete_a1_input_evidence_allows_evaluation_without_claiming_reproduction(self):
        evidence_path, commit = self.complete_a1_fixture()
        code, report = self.run_check("--evidence", str(evidence_path), "--official-commit", commit)
        self.assertEqual(code, 0)
        self.assertTrue(report["can_evaluate"])
        self.assertFalse(report["reproduction_ready"])
        self.assertEqual(report["blockers"], [])

    def complete_a1_fixture(self):
        # Synthetic attestations test the gate, not the model or any baseline metric.
        source_files = ("benchmark/eval_zeroshot.py", "relsgg/checkpoint.py", "relsgg/eval/evaluator.py",
                        "relsgg/data/dataset.py", "pyproject.toml", "LICENSE", "NOTICE")
        pack_files = ("meta.json", "file_names.json", "img_meta.npy", "boxes.npy", "box_cats.npy", "rels.npy")
        for directory, names in (("official", source_files), ("snapshot", ("model.pth", "text_student.pt", "tokenizer.json", "tokenizer_config.json")), ("vg150/test", pack_files)):
            for name in names:
                path = self.root / directory / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture: " + name.encode())
        official = self.root / "official"
        subprocess.run(["git", "init", str(official)], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(official), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(official), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"], check=True, capture_output=True)
        commit = subprocess.check_output(["git", "-C", str(official), "rev-parse", "HEAD"], text=True).strip()
        hashes = lambda directory, names: {name: hashlib.sha256((self.root / directory / name).read_bytes()).hexdigest() for name in names}
        evidence = {
            "official_commit": commit, "snapshot_revision": "1" * 40, "data_revision": "2" * 40,
            "snapshot_sha256": hashes("snapshot", ("model.pth", "text_student.pt", "tokenizer.json", "tokenizer_config.json")),
            "strict_load": {"strict": True, "missing_keys": [], "unexpected_keys": [], "shape_mismatches": [], "remapped_keys": [],
                            "model_sha256": hashes("snapshot", ("model.pth",))["model.pth"],
                            "text_student_sha256": hashes("snapshot", ("text_student.pt",))["text_student.pt"],
                            "source_sha256": hashes("official", source_files)},
            "full_split": {"dataset": "vg150", "split": "test", "expected_images": 26404,
                           "loaded_images": 26404, "valid_input_images": 26404, "failed_images": 0,
                           "limit": 0, "pack_sha256": hashes("vg150/test", pack_files)},
        }
        evidence_path = self.root / "evidence.json"
        evidence_path.write_text(json.dumps(evidence))
        return evidence_path, commit

    def test_pack_changed_after_full_split_audit_blocks_previously_complete_evidence(self):
        evidence, commit = self.complete_a1_fixture()
        (self.root / "vg150/test/rels.npy").write_bytes(b"changed relations")
        code, report = self.run_check("--evidence", str(evidence), "--official-commit", commit)
        self.assertEqual(code, 2)
        self.assertIn("full_split_artifact_mismatch", report["blockers"])

    def test_partial_load_blocks_otherwise_complete_a1_evidence(self):
        evidence, commit = self.complete_a1_fixture()
        content = json.loads(evidence.read_text())
        content["strict_load"]["missing_keys"] = ["vocab_head.W"]
        evidence.write_text(json.dumps(content))
        code, report = self.run_check("--evidence", str(evidence), "--official-commit", commit)
        self.assertEqual(code, 2)
        self.assertIn("strict_load_unverified", report["blockers"])

    def test_deployment_calibration_cannot_substitute_for_a3_matcher_calibration(self):
        calibration = self.root / "calibration.json"
        calibration.write_text(json.dumps({"a": 0.5651, "b": -1.9623, "score": "sigmoid"}))
        _, report = self.run_check("--protocol", "A3", "--tau-calibration", str(calibration))
        self.assertIn("a3_matcher_calibration_invalid", report["blockers"])

    def test_a3_requires_bound_runner_and_vocabulary_evidence_even_for_legacy_fields(self):
        evidence, commit = self.complete_a1_fixture()
        embeddings = self.root / "snapshot/predicate_embeddings.npz"
        with zipfile.ZipFile(embeddings, "w") as archive:
            archive.writestr("predicates.npy", b"fixture")
            archive.writestr("embeddings.npy", b"fixture")
        calibration = self.root / "tau.json"
        calibration.write_text(json.dumps({"pred_embeds": str(embeddings), "chosen": {"tau": 0.72}}))
        code, report = self.run_check("--protocol", "A3", "--tau-calibration", str(calibration),
                                      "--evidence", str(evidence), "--official-commit", commit)
        self.assertEqual(code, 2)
        self.assertIn("a3_runner_contract_unverified", report["blockers"])

    def test_remapped_checkpoint_cannot_pass_as_clean_strict_load(self):
        evidence, commit = self.complete_a1_fixture()
        content = json.loads(evidence.read_text())
        content["strict_load"]["remapped_keys"] = ["head.weight -> vocab_head.W"]
        evidence.write_text(json.dumps(content))
        code, report = self.run_check("--evidence", str(evidence), "--official-commit", commit)
        self.assertEqual(code, 2)
        self.assertIn("strict_load_unverified", report["blockers"])

    def test_report_records_current_asset_hashes_and_matches_saved_output(self):
        evidence, commit = self.complete_a1_fixture()
        output = self.root / "external/preflight.json"
        _, report = self.run_check("--evidence", str(evidence), "--official-commit", commit, "--output", str(output))
        self.assertEqual(json.loads(output.read_text()), report)
        expected = hashlib.sha256((self.root / "snapshot/model.pth").read_bytes()).hexdigest()
        self.assertEqual(report["artifact_sha256"]["snapshot"]["model.pth"], expected)

    def test_invalid_evidence_and_corrupt_embedding_archive_fail_closed(self):
        evidence = self.root / "evidence.json"
        evidence.write_text("[]")
        snapshot = self.root / "snapshot"
        snapshot.mkdir()
        (snapshot / "predicate_embeddings.npz").write_bytes(b"not a zip archive")
        code, report = self.run_check("--evidence", str(evidence), "--protocol", "A3")
        self.assertEqual(code, 2)
        self.assertIn("evidence_invalid", report["blockers"])
        self.assertIn("a3_embedding_archive_invalid", report["blockers"])

    def test_apache_parent_source_does_not_require_the_later_agpl_notice(self):
        evidence, _ = self.complete_a1_fixture()
        (self.root / "official/NOTICE").unlink()
        _, report = self.run_check("--evidence", str(evidence), "--official-commit",
                                  "4a07de9d06f2e3f14309753b7907cf1d3a263b08")
        self.assertNotIn(str(self.root / "official/NOTICE"), report["missing_paths"])
        self.assertNotIn("NOTICE", report["artifact_sha256"]["source"])

    def test_port_consumer_accepts_release_schema_only_with_bound_consumer_evidence(self):
        evidence_path, commit = self.complete_a1_fixture()
        embeddings = self.root / "snapshot/predicate_embeddings.npz"
        with zipfile.ZipFile(embeddings, "w") as archive:
            archive.writestr("names.npy", b"fixture")
            archive.writestr("W.npy", b"fixture")
        calibration = self.root / "tau.json"
        calibration.write_text(json.dumps({"pred_embeds": str(embeddings), "chosen": {"tau": 0.72}}))
        evidence = json.loads(evidence_path.read_text())
        repo = Path(__file__).resolve().parents[2]
        evidence["consumer_sha256"] = {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in ("src/relateanything.py", "src/relateanything_cli.py", "src/relateanything_evaluation.py")
        }
        evidence["a3"] = {"checkpoint_pred_embeds": str(embeddings.resolve()),
                          "embedding_sha256": hashlib.sha256(embeddings.read_bytes()).hexdigest(),
                          "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
                          "vocabulary_order_verified": True, "runner_layout_verified": True}
        evidence_path.write_text(json.dumps(evidence))
        code, report = self.run_check("--protocol", "A3", "--consumer", "opensgg",
                                      "--tau-calibration", str(calibration), "--evidence", str(evidence_path),
                                      "--official-commit", commit)
        self.assertEqual(code, 0, report["blockers"])
        evidence["consumer_sha256"]["src/relateanything.py"] = "0" * 64
        evidence_path.write_text(json.dumps(evidence))
        code, report = self.run_check("--protocol", "A3", "--consumer", "opensgg",
                                      "--tau-calibration", str(calibration), "--evidence", str(evidence_path),
                                      "--official-commit", commit)
        self.assertEqual(code, 2)
        self.assertIn("consumer_identity_unverified", report["blockers"])


if __name__ == "__main__":
    unittest.main()
