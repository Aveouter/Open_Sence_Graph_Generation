from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.reproduction.check_penet_official_inputs import checkpoint_candidates


class PENetOfficialInputCheckTest(unittest.TestCase):
    def test_checkpoint_candidates_do_not_cross_protocol_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sgdet_dir = root / "PE-NET_SGDet"
            sgdet_dir.mkdir()
            ckpt = sgdet_dir / "model_final.pth"
            ckpt.write_bytes(b"sgdet checkpoint sentinel")

            self.assertEqual(checkpoint_candidates(root, "PE-NET_PredCls"), [])
            self.assertEqual(checkpoint_candidates(root, "PE-NET_SGCls"), [])
            candidates = checkpoint_candidates(root, "PE-NET_SGDet")
            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0]["path"], str(ckpt))


if __name__ == "__main__":
    unittest.main()
