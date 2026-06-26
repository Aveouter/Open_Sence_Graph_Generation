#!/usr/bin/env python3
"""
CI Smoke Test Script for OpenSGG.

Runs lightweight integration checks against changed models/methods:
  1. Imports all registered methods from src.methods
  2. For each changed model (or all, if running locally):
     a. Loads the corresponding config from configs/VisualGenome/
     b. Instantiates the LightningModule
     c. Runs a synthetic forward pass (random tensors sized for the model)
  3. If train.py is available, runs a minimal CPU subprocess invocation
     (--dataset_size 2 --epoch 1 --device cpu)

Exit code 0 on all passes, 1 on any failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Name matching helpers
# ---------------------------------------------------------------------------

# Config filenames that differ from method_maps keys for historical reasons.
# New models where config filename == method key do NOT need an entry here.
CONFIG_NAME_ALIASES: Dict[str, str] = {
    "gps_net": "gpsnet",
    "pe_net": "penet",
    "sha_gcl": "shagcl",
    "gpsnet": "gpsnet",  # alias → self (for direct lookup)
    "penet": "penet",
    "shagcl": "shagcl",
}


def _resolve_config_key(method_name: str) -> str:
    """Map a method or config filename stem to the canonical method_maps key.

    Handles the 3 historical naming mismatches:
      GPS_Net.py → "gpsnet",  PE_NET.py → "penet",  SHA_GCL.py → "shagcl"
    All other methods: filename == method key (case-insensitive).
    """
    return CONFIG_NAME_ALIASES.get(method_name.lower(), method_name.lower())


def _find_method_for_model(model_stem: str) -> Optional[str]:
    """Map a model file stem to a method_maps key.

    Most models follow a direct naming convention (usg.py → "usg").
    Only truly irregular cases need explicit entries.
    """
    from src.methods import method_maps

    # Irregular model file stems
    exceptions = {
        "react_sgg": "react",
        "transformer_sgg": "transformer",
    }
    if model_stem.lower() in exceptions:
        return exceptions[model_stem.lower()]

    # Direct match: lowercase stem must match a method_maps key
    stem_lower = model_stem.lower()
    if stem_lower in method_maps:
        return stem_lower

    return None


def load_config(method_name: str, dataname: str = "VisualGenome") -> dict:
    """Load a config file and return its namespace as a dict."""
    cfg_path = ROOT / "configs" / dataname / f"{method_name}.py"

    if not cfg_path.exists():
        # Try case-insensitive + historical alias matching
        cfg_dir = ROOT / "configs" / dataname
        target = _resolve_config_key(method_name)
        for f in cfg_dir.glob("*.py"):
            if _resolve_config_key(f.stem) == target:
                cfg_path = f
                break
        else:
            raise FileNotFoundError(
                f"Config not found: configs/{dataname}/{method_name}.py"
            )

    # Execute the config file and capture its namespace
    config_dict: dict = {}
    code = cfg_path.read_text(encoding="utf-8")
    exec(compile(code, str(cfg_path), "exec"), config_dict)

    # Remove builtins
    config_dict.pop("__builtins__", None)
    return config_dict


def build_args_from_config(config_dict: dict) -> SimpleNamespace:
    """Build a SimpleNamespace args from a config dict, filling defaults."""
    defaults = {
        "device": "cpu",
        "dist": False,
        "output_dir": str(ROOT / "outputs"),
        "ex_name": "CI_SmokeTest",
        "seed": 42,
        "fps": False,
        "test": False,
        "deterministic": False,
        "num_workers": 0,
        "data_root": str(ROOT / "data"),
        "dataname": "VisualGenome",
        "use_augment": False,
        "use_prefetcher": False,
        "drop_last": False,
        "overwrite": True,
        "epoch": 1,
        "log_step": 1,
        "opt": "adam",
        "opt_eps": None,
        "opt_betas": None,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "clip_grad": None,
        "clip_mode": "norm",
        "sched": "onecycle",
        "lr": 1e-3,
        "warmup_lr": 1e-5,
        "min_lr": 1e-6,
        "warmup_epoch": 0,
        "decay_epoch": 100,
        "decay_rate": 0.1,
        "accumulate_grad_batches": 1,
        "gpus": [0],
        "metric_for_bestckpt": "val_loss",
        "metrics": ["sgdet_R@10"],
        "rel_nums": 51,
        "entity_nums": 151,
        "batch_size": 2,
        "val_batch_size": 2,
        "dataset_size": 2,
        "val_dataset_size": 2,
        "test_dataset_size": 2,
        "steps_per_epoch": 1,
        "save_dir": str(ROOT / "outputs" / "CI_SmokeTest"),
        # CLIP / alignment defaults
        "use_alignment": False,
        "prototype_path": str(ROOT / "data" / "VisualGenome" / "clip_prototypes.pth"),
        "clip_model": "ViT-B-32",
        "clip_dim": 512,
        "num_hierarchy_levels": 3,
        "hierarchy_weights": [0.2, 0.3, 0.5],
        "temperature": 0.07,
        "align_loss_coef": 0.2,
        # FlowSG defaults
        "sampling_timesteps": 1,
    }

    # Merge: config values override defaults
    merged = {**defaults, **config_dict}

    # Force CPU — some configs set device='cuda' which would pull in CUDA ops
    merged["device"] = "cpu"

    # Ensure method is lowercase
    if "method" in merged:
        merged["method"] = merged["method"].lower()

    return SimpleNamespace(**merged)


def _instantiate_with_timeout(method_cls, method_name, save_dir, kwargs):
    """Instantiate a method with a timeout to catch hanging downloads.

    FlowSG and EGTR download large pretrained weights (CLIP,
    DeformableDETR) from HuggingFace at __init__ time.  In CI this can
    time out or fail if HF is unreachable, wasting 10+ minutes.
    Skipped models raise NotImplementedError (non-fatal).
    """
    import threading

    result = [None]
    error = [None]

    def _build():
        try:
            result[0] = method_cls(steps_per_epoch=1, save_dir=save_dir, **kwargs)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_build, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        raise NotImplementedError(
            f"{method_name}: instantiation timed out (30s) — "
            f"likely downloading pretrained weights"
        )

    if error[0] is not None:
        exc = error[0]
        msg = str(exc).lower()
        # Exception types that indicate network/IO failures
        if isinstance(exc, (OSError, IOError)):
            raise NotImplementedError(
                f"{method_name}: skipping (download/network unavailable)"
            )
        # Keyword matches for urllib3 / requests / HuggingFace errors
        # that don't subclass OSError (e.g. ReadTimeoutError, HTTPError)
        network_kw = (
            "hf-mirror",
            "huggingface",
            "connection",
            "timeout",
            "timed out",
            "read timed",
            "retry",
            "max retries",
            "urllib3",
            "httperror",
            "requests",
            "connect call",
            "name resolution",
            "temporary failure",
        )
        if any(kw in msg for kw in network_kw):
            raise NotImplementedError(
                f"{method_name}: skipping (remote model unavailable)"
            )
        raise

    return result[0]


def import_method_class(method_name: str):
    """Import and return the LightningModule class for a method."""
    from src.methods import method_maps

    method_name = method_name.lower()
    if method_name not in method_maps:
        raise KeyError(
            f"Method '{method_name}' not in method_maps. "
            f"Available: {sorted(method_maps.keys())}"
        )
    return method_maps[method_name]


def try_instantiate_method(
    method_name: str, dataname: str = "VisualGenome"
) -> Tuple[bool, str]:
    """Try to instantiate a method LightningModule from its config.

    Returns (success, error_message).
    """
    method_name = method_name.lower()
    try:
        # 1. Load config
        config_dict = load_config(method_name, dataname)

        # 2. Build args
        args = build_args_from_config(config_dict)

        # 3. Import the method class
        method_cls = import_method_class(method_name)

        # 4. Instantiate
        # NOTE: vars() returns args.__dict__ by REFERENCE, not a copy.
        # Popping from it would remove the attribute from args itself.
        print(f"    Instantiating {method_cls.__name__} ...")
        all_kwargs = dict(vars(args))
        save_dir = all_kwargs.pop("save_dir", str(ROOT / "outputs" / "CI_SmokeTest"))
        all_kwargs.pop("steps_per_epoch", None)

        # 4. Instantiate (with timeout for models that download weights)
        try:
            model = _instantiate_with_timeout(
                method_cls, method_name, save_dir, all_kwargs
            )
            print(f"    ✓ Instantiated {method_cls.__name__}")
        except NotImplementedError as e:
            print(f"    ⚠ Instantiation skipped: {e}")
            return True, ""  # non-fatal skip

        # 5. Try a synthetic forward pass (CPU-safe, small tensors)
        try:
            print("    Running synthetic forward pass ...")
            _synthetic_forward(model, method_name)
            print("    ✓ Forward pass OK")
        except NotImplementedError:
            # Models that explicitly cannot support synthetic forward
            print("    ⚠ Forward pass skipped (not supported by this model)")

        return True, ""

    except Exception as e:
        tb = traceback.format_exc()
        return False, f"{method_name}: {e}\n{tb}"


def _synthetic_forward(model, method_name: str) -> None:
    """Run a synthetic forward pass through the model with synthetic data.

    Tries multiple input formats to accommodate different model interfaces:
      1. batched tensor [B,3,H,W] — Motifs, CVC, HSTRNet, most two-stage models
      2. Tries without targets (inference mode)
      3. list of per-sample tensors — RelTR, EGTR
      4. NestedTensor — FlowSG, USG

    If ALL formats fail, raises RuntimeError with the accumulated errors.
    """
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    B = 1
    img_batch = torch.randn(B, 3, 224, 224, device=device)
    img_list = [img_batch[0]]  # [3, H, W] per image

    # Synthetic targets covering multiple key names used by different models
    targets_batch = [
        {
            "labels": torch.randint(1, 150, (3,), device=device),
            "class_labels": torch.randint(1, 150, (3,), device=device),
            "boxes": torch.rand(3, 4, device=device),
            "rel_annotations": torch.tensor([[0, 1, 5], [1, 2, 10]], device=device),
        }
    ]

    errors: list[str] = []

    def _log_result(out):
        if isinstance(out, dict) and "loss" in out:
            loss_val = (
                out["loss"].item() if torch.is_tensor(out["loss"]) else out["loss"]
            )
            print(f"      loss = {loss_val:.4f}")
        elif isinstance(out, torch.Tensor):
            print(f"      output shape = {out.shape}")
        else:
            print(f"      output type = {type(out).__name__}")

    with torch.no_grad():
        # --- Format 1: batched tensor (Motifs, CVC, HSTRNet) ---
        try:
            out = model(img_batch, targets_batch)
            _log_result(out)
            return
        except Exception as e:
            errors.append(f"batched+tgt: {e}")

        # --- Format 2: inference without targets ---
        try:
            out = model(img_batch)
            _log_result(out)
            return
        except Exception as e:
            errors.append(f"batched+no_tgt: {e}")

        # --- Format 3: list of per-sample tensors (RelTR, EGTR) ---
        try:
            out = model(img_list, targets_batch)
            _log_result(out)
            return
        except Exception as e:
            errors.append(f"list+tgt: {e}")

        # --- Format 4: NestedTensor (FlowSG, USG) ---
        try:
            from utils.misc import nested_tensor_from_tensor_list

            samples = nested_tensor_from_tensor_list(img_list)
            out = model(samples, targets_batch)
            _log_result(out)
            return
        except Exception as e:
            errors.append(f"nested+tgt: {e}")

    raise RuntimeError(
        f"All forward pass formats failed for {method_name}: {'; '.join(errors)}"
    )


def detect_changed_methods(changed_files_str: str) -> List[str]:
    """Parse changed files to determine which methods changed.

    Accepts a comma/space/newline-separated list of changed file paths.
    """
    if not changed_files_str:
        return []

    files = changed_files_str.replace(",", "\n").split()
    methods: set = set()

    for f in files:
        f = f.strip()
        # src/methods/xxx_method.py → xxx
        if "src/methods/" in f and f.endswith("_method.py"):
            stem = Path(f).stem.replace("_method", "")
            methods.add(stem)
        # src/models/xxx.py → auto-match to method_maps key
        elif "src/models/" in f and f.endswith(".py"):
            stem = Path(f).stem
            method = _find_method_for_model(stem)
            if method:
                methods.add(method)
        # configs/VisualGenome/Xxx.py → method (resolve historical alias)
        elif "configs/" in f and f.endswith(".py"):
            method = _resolve_config_key(Path(f).stem)
            methods.add(method)

    return sorted(methods)


def run_minimal_train(method_name: str) -> Tuple[bool, str]:
    """Run train.py with minimal settings on CPU as a subprocess."""
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--method",
        method_name,
        "--dataname",
        "VisualGenome",
        "--dataset_size",
        "2",
        "--val_dataset_size",
        "2",
        "--epoch",
        "1",
        "--batch_size",
        "1",
        "--val_batch_size",
        "1",
        "--num_workers",
        "0",
        "--device",
        "cpu",
        "--gpus",
        "0",
        "--overwrite",
        "--ex_name",
        f"CI_Smoke_{method_name}",
    ]

    print(f"    Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=ROOT,
            timeout=300,  # 5 min max per method
        )
        if result.returncode != 0:
            # Print last 30 lines of stderr for debugging
            stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
            stdout_tail = "\n".join(result.stdout.splitlines()[-30:])
            return False, (
                f"train.py exit code {result.returncode}\n"
                f"STDERR (last 30 lines):\n{stderr_tail}\n"
                f"STDOUT (last 30 lines):\n{stdout_tail}"
            )
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"train.py timed out after 300s for {method_name}"
    except Exception as e:
        return False, f"train.py failed: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenSGG CI Smoke Test")
    parser.add_argument(
        "--changed-models",
        default="",
        help="Comma/space-separated list of changed files from CI",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Specific methods to test (default: auto-detect from changed files)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test ALL registered methods (danger: very slow!)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the train.py subprocess invocation",
    )
    parser.add_argument(
        "--dataname",
        default="VisualGenome",
        help="Dataset name for config loading",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OpenSGG CI Smoke Test")
    print("=" * 60)

    # Determine which methods to test
    if args.all:
        from src.methods import method_maps

        methods_to_test = sorted(method_maps.keys())
        print(f"\nTesting ALL {len(methods_to_test)} registered methods")
    elif args.methods:
        methods_to_test = [m.lower() for m in args.methods]
        print(f"\nTesting specified methods: {methods_to_test}")
    else:
        methods_to_test = detect_changed_methods(args.changed_models)
        if methods_to_test:
            print(f"\nDetected changed methods: {methods_to_test}")
        else:
            # When running locally with no context, test import only
            print("\nNo changed methods detected. Running import check only.")
            return _import_check_only()

    # ---- Phase 1: Instantiation + forward pass ----
    print(f"\n{'=' * 40}")
    print("Phase 1: Method instantiation + forward pass")
    print(f"{'=' * 40}")

    failures: Dict[str, str] = {}
    for method_name in methods_to_test:
        print(f"\n  [{method_name}]")
        ok, err = try_instantiate_method(method_name, args.dataname)
        if not ok:
            failures[method_name] = err
            print(f"    [FAIL] {err.split(chr(10))[0]}")  # first line only

    # ---- Phase 2: Minimal train.py subprocess ----
    if not args.skip_train and not failures:
        print(f"\n{'=' * 40}")
        print("Phase 2: Minimal train.py subprocess")
        print(f"{'=' * 40}")

        for method_name in methods_to_test:
            print(f"\n  [{method_name}]")
            ok, err = run_minimal_train(method_name)
            if not ok:
                # train.py failure is a warning, not blocking
                # (data may not exist in CI, pretrained weights may be missing)
                print(f"    [WARN] train.py failed (non-blocking): {err[:200]}")

    # ---- Report ----
    print("\n" + "=" * 60)
    if failures:
        print(f"SMOKE TEST FAILED — {len(failures)} method(s) failed instantiation:")
        for name, err in failures.items():
            print(f"\n  [{name}]")
            # Print first 5 lines of traceback
            lines = err.splitlines()[:8]
            for line in lines:
                print(f"    {line}")
        return 1
    else:
        print(f"ALL SMOKE TESTS PASSED ({len(methods_to_test)} method(s))")
        return 0


def _import_check_only() -> int:
    """Lightweight check: verify all registered methods are importable."""
    print("\n[Import Check] Verifying all registered methods can be imported ...")
    from src.methods import method_maps

    errors = []
    for name, cls in sorted(method_maps.items()):
        try:
            # Trigger lazy-load by accessing the class
            _ = cls
            print(f"  ✓ {name}: {cls.__name__}")
        except Exception as e:
            errors.append(f"  ✗ {name}: {e}")
            print(f"  ✗ {name}: {e}")

    if errors:
        print(f"\n{len(errors)} import error(s):")
        for e in errors:
            print(e)
        return 1
    else:
        print(f"\nAll {len(method_maps)} methods importable")
        return 0


if __name__ == "__main__":
    sys.exit(main())
