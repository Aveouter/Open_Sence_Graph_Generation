from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "tools" / "reproduction" / "build_artifact_manifest.py"

spec = importlib.util.spec_from_file_location("build_artifact_manifest", SCRIPT)
assert spec is not None and spec.loader is not None
manifest_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest_mod)


class ArtifactManifestTest(unittest.TestCase):
    def test_artifact_kind_classifies_common_blockers(self) -> None:
        self.assertEqual(
            manifest_mod.artifact_kind("missing_sgb_vg_inputs"), "vg_inputs"
        )
        self.assertEqual(
            manifest_mod.artifact_kind("missing_pretrained_detector_checkpoint"),
            "detector_checkpoint",
        )
        self.assertEqual(
            manifest_mod.artifact_kind("missing_official_memory_bank_features"),
            "memory_bank",
        )
        self.assertEqual(
            manifest_mod.artifact_kind("missing_official_tde_checkpoints"), "checkpoint"
        )

    def test_public_blocker_removes_local_absolute_path(self) -> None:
        blocker = str(
            Path("/").joinpath("redacted", "datasets", "vg", "VG-SGG-with-attri.h5")
        )

        self.assertEqual(manifest_mod.public_blocker(blocker), "VG-SGG-with-attri.h5")

    def test_build_manifest_preserves_non_reproduction_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = root / "docs" / "reproduction" / "evidence_gate_summary.json"
            detail = (
                root / "docs" / "reproduction" / "freq" / "sgb_freq_input_check.json"
            )
            detail.parent.mkdir(parents=True)
            summary.parent.mkdir(parents=True, exist_ok=True)
            detail.write_text(
                json.dumps(
                    {
                        "status": "BLOCKED_MISSING_SGB_VG_INPUTS",
                        "missing": ["VG-SGG-with-attri.h5"],
                    }
                )
            )
            summary.write_text(
                json.dumps(
                    {
                        "status": "BLOCKED",
                        "results": [
                            {
                                "key": "freq",
                                "label": "FREQ",
                                "status": "BLOCKED",
                                "reported_status": "BLOCKED_MISSING_SGB_VG_INPUTS",
                                "output": "docs/reproduction/freq/sgb_freq_input_check.json",
                                "blockers": ["missing_sgb_vg_inputs"],
                                "next_action": "Provide SGB-format VG inputs",
                            }
                        ],
                    }
                )
            )

            manifest = manifest_mod.build_manifest(root, summary)

        self.assertEqual(manifest["claim"], "not reproduction-ready")
        self.assertEqual(manifest["artifact_count"], 1)
        self.assertEqual(manifest["artifacts"][0]["kind"], "vg_inputs")
        self.assertEqual(manifest["baselines"]["freq"]["artifact_count"], 1)


if __name__ == "__main__":
    unittest.main()
