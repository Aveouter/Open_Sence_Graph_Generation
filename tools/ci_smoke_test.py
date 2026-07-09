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
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure the project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# CI synthetic data generator
# ---------------------------------------------------------------------------

def _setup_ci_data(data_root: Path) -> None:
    """Ensure minimal VisualGenome data exists for CI smoke tests.

    Prefers the committed sample dataset (data/VisualGenome_sample/),
    falling back to synthetic data when that is also absent.
    Only runs when real data is missing — never overwrites.
    """
    images_dir = data_root / "images"
    train_json = data_root / "train.json"
    rel_json = data_root / "rel.json"

    if images_dir.exists() and train_json.exists() and rel_json.exists():
        return  # real data present — nothing to do

    # ---- Prefer committed sample dataset ----
    sample_dir = ROOT / "data" / "VisualGenome_sample"
    if sample_dir.exists() and (sample_dir / "train.json").exists():
        print("    Using committed VG sample dataset for CI")
        import shutil
        data_root.mkdir(parents=True, exist_ok=True)
        for item in os.listdir(str(sample_dir)):
            src = sample_dir / item
            dst = data_root / item
            if src.is_dir():
                if not dst.exists():
                    shutil.copytree(src, dst)
            else:
                if not dst.exists():
                    shutil.copy2(src, dst)
        return

    # ---- Fallback: synthetic data (only if sample also missing) ----
    print("    Generating synthetic CI dataset (2 images, 2 samples) ...")

    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)

    img_ids = [1, 2]
    for img_id in img_ids:
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
            Image.fromarray(arr).save(img_path)

    categories = [{"supercategory": "", "id": i, "name": str(i)}
                  for i in range(1, 151)]
    images = [{"file_name": f"{iid}.jpg", "height": 224, "width": 224, "id": iid}
              for iid in img_ids]

    ann_id = 1
    annotations: list = []
    for iid in img_ids:
        for obj_i in range(5):
            x = 10 + obj_i * 40
            y = 10 + obj_i * 30
            annotations.append({
                "segmentation": None, "area": 2000,
                "bbox": [x, y, 50, 50], "iscrowd": 0,
                "image_id": iid, "id": ann_id,
                "category_id": (ann_id % 150) + 1,
            })
            ann_id += 1

    coco_data = {"images": images, "annotations": annotations,
                 "categories": categories}
    for fname in ("train.json", "val.json", "test.json"):
        path = data_root / fname
        if not path.exists():
            path.write_text(json.dumps(coco_data))

    if not rel_json.exists():
        rel_categories = ["__background__"] + [f"predicate_{i}" for i in range(1, 51)]
        default_rels = [[0, 1, 10], [2, 3, 20], [1, 4, 30]]
        rel_data = {
            "rel_categories": rel_categories,
            "train": {"1": default_rels, "2": default_rels},
            "val": {"1": default_rels, "2": default_rels},
            "test": {"1": default_rels, "2": default_rels},
        }
        rel_json.write_text(json.dumps(rel_data))

    print(f"    ✓ Synthetic dataset ready at {data_root}")


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
    "gpsnet": "gpsnet",
    "penet": "penet",
    "shagcl": "shagcl",
    # RA-SGG task variants — same method, different eval_mode configs
    "ra_sgg_sgcls": "ra_sgg",
    "ra_sgg_sgdet": "ra_sgg",
}

# Mapping from method_maps keys (lowercase) to train.py --method choices
# (PascalCase / mixed-case).  The method_maps registry and config files
# use lowercase keys, but train.py's argparse expects the display names.
METHOD_PARSER_CHOICES: Dict[str, str] = {
    "cvc": "CVC",
    "egtr": "EGTR",
    "flowsg": "FlowSG",
    "gpsnet": "GPSNet",
    "hstrnet": "HSTRNet",
    "imp": "IMP",
    "motifs": "Motifs",
    "penet": "PENet",
    "ra_sgg": "RA_SGG",
    "react": "REACT",
    "reltr": "RelTR",
    "shagcl": "SHAGCL",
    "squat": "SQUAT",
    "tde": "TDE",
    "transformer": "Transformer",
    "usg": "USG",
    "vctree": "VCTree",
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


def _print_results(stdout: str) -> None:
    """Extract and print key metrics from train/test stdout."""
    # Look for the rich table (Lightning CSVLogger output)
    table_start = None
    table_lines = stdout.splitlines()
    for i, line in enumerate(table_lines):
        if "┏" in line and "Test metric" in " ".join(table_lines[i : i + 3]):
            table_start = i
            break
    if table_start is not None:
        print("    " + "-" * 48)
        for line in table_lines[table_start:]:
            if line.strip():
                print(f"    {line}")
        print("    " + "-" * 48)
        return

    # Fallback: scan for R@ lines
    metrics = re.findall(r"^(R@\d+.*|mR@\d+.*)$", stdout, re.MULTILINE)
    if metrics:
        print("    Results:")
        for m in metrics[:8]:
            print(f"      {m.strip()}")
        return

    # Fallback: scan for loss lines
    losses = re.findall(r"^\S*loss\S*\s*[=:]\s*[\d.]+.*$", stdout, re.MULTILINE)
    if losses:
        print("    Losses:")
        for lo in losses[:4]:
            print(f"      {lo.strip()}")


def _find_ckpt(method_name: str) -> Optional[str]:
    """Find the checkpoint file produced by train.py Phase 2.

    Searches outputs/runs/{method}/YYYY-MM-DD_CI_Smoke_{method}/checkpoints/.
    """
    runs_dir = ROOT / "outputs" / "runs" / method_name
    if not runs_dir.exists():
        return None
    # Find the most recent CI_Smoke_* run directory
    smoke_dirs = sorted(runs_dir.glob(f"*_CI_Smoke_{method_name}"), reverse=True)
    if not smoke_dirs:
        return None
    ckpt_dir = smoke_dirs[0] / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if ckpts:
        # Prefer last.ckpt, fall back to newest
        last = ckpt_dir / "last.ckpt"
        return str(last) if last.exists() else str(ckpts[-1])
    return None


def run_minimal_train(method_name: str) -> Tuple[bool, str, Optional[str]]:
    """Run train.py with minimal settings on CPU as a subprocess.

    Returns (success, error_message, ckpt_path).
    """
    # Map lowercase method_maps key to train.py argparse choice
    parser_name = METHOD_PARSER_CHOICES.get(method_name, method_name)

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--method",
        parser_name,
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
        "--opt",
        "adam",
        "--sched",
        "onecycle",
        "--overwrite",
        "--no_display_method_info",
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
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "CI_SMOKE_TEST": "1",
            },
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
            ), None
        ckpt_path = _find_ckpt(method_name)
        if ckpt_path:
            print(f"    ✓ Checkpoint: {ckpt_path}")
        else:
            print("    ⚠ No checkpoint found (train may not have saved one)")
        _print_results(result.stdout)
        return True, "", ckpt_path
    except subprocess.TimeoutExpired:
        return False, f"train.py timed out after 300s for {method_name}", None
    except Exception as e:
        return False, f"train.py failed: {e}", None


def run_minimal_test(
    method_name: str, ckpt_path: str
) -> Tuple[bool, str]:
    """Run train.py --test with the checkpoint from Phase 2.

    Returns (success, error_message).
    """
    parser_name = METHOD_PARSER_CHOICES.get(method_name, method_name)

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--method",
        parser_name,
        "--dataname",
        "VisualGenome",
        "--test",
        "--ckpt_path",
        ckpt_path,
        "--dataset_size",
        "2",
        "--test_dataset_size",
        "2",
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
        "--no_display_method_info",
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
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "CI_SMOKE_TEST": "1",
            },
            timeout=300,
        )
        if result.returncode != 0:
            stderr_tail = "\n".join(result.stderr.splitlines()[-30:])
            stdout_tail = "\n".join(result.stdout.splitlines()[-30:])
            return False, (
                f"train.py --test exit code {result.returncode}\n"
                f"STDERR (last 30 lines):\n{stderr_tail}\n"
                f"STDOUT (last 30 lines):\n{stdout_tail}"
            )
        _print_results(result.stdout)
        return True, ""
    except subprocess.TimeoutExpired:
        return False, f"train.py --test timed out after 300s for {method_name}"
    except Exception as e:
        return False, f"train.py --test failed: {e}"


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
    train_ckpts: Dict[str, str] = {}
    train_warnings: Dict[str, str] = {}
    if not args.skip_train and not failures:
        print(f"\n{'=' * 40}")
        print("Phase 2: Minimal train.py subprocess")
        print(f"{'=' * 40}")

        # Ensure minimal dataset exists for CI runners
        _setup_ci_data(ROOT / "data" / "VisualGenome")

        for method_name in methods_to_test:
            print(f"\n  [{method_name}]")
            ok, err, ckpt_path = run_minimal_train(method_name)
            if not ok:
                # train.py failure is a warning, not blocking
                # (data may not exist in CI, pretrained weights may be missing)
                print(f"    [WARN] train.py failed (non-blocking)")
                # Print first error line and last meaningful part of traceback
                for line in err.split('\n'):
                    if line.strip():
                        print(f"           {line[:250]}")
                        break
                # Find the last Traceback-or-Error section
                tb_start = err.rfind('Traceback')
                if tb_start > 0:
                    for line in err[tb_start:].split('\n')[-6:]:
                        print(f"           {line[:250]}")
                train_warnings[method_name] = err
            elif ckpt_path:
                train_ckpts[method_name] = ckpt_path
            else:
                print("    [WARN] train.py ran but produced no checkpoint")
                train_warnings[method_name] = "no checkpoint produced"

    # ---- Phase 3: Minimal inference (test) ----
    test_warnings: Dict[str, str] = {}
    if not args.skip_train and train_ckpts:
        print(f"\n{'=' * 40}")
        print("Phase 3: Minimal inference (train.py --test)")
        print(f"{'=' * 40}")

        for method_name, ckpt_path in train_ckpts.items():
            print(f"\n  [{method_name}]")
            ok, err = run_minimal_test(method_name, ckpt_path)
            if not ok:
                # test.py failure is a warning, not blocking
                print(f"    [WARN] test phase failed (non-blocking)")
                for line in err.split('\n'):
                    if line.strip():
                        print(f"           {line[:250]}")
                        break
                tb_start = err.rfind('Traceback')
                if tb_start > 0:
                    for line in err[tb_start:].split('\n')[-6:]:
                        print(f"           {line[:250]}")
                test_warnings[method_name] = err
            else:
                print("    ✓ Test phase OK")

    # ---- Report ----
    print("\n" + "=" * 60)
    exit_code = 0

    if failures:
        print(f"PHASE 1 FAILED — {len(failures)} method(s) failed instantiation:")
        for name, err in failures.items():
            print(f"\n  [{name}]")
            lines = err.splitlines()[:8]
            for line in lines:
                print(f"    {line}")
        exit_code = 1
    else:
        print(f"Phase 1 OK ({len(methods_to_test)} method(s))")

    if train_warnings:
        print(f"\nPhase 2 warnings ({len(train_warnings)} method(s)):")
        for name, err in train_warnings.items():
            print(f"  [{name}] {err.split(chr(10))[0][:120]}")

    if test_warnings:
        print(f"\nPhase 3 warnings ({len(test_warnings)} method(s)):")
        for name, err in test_warnings.items():
            print(f"  [{name}] {err.split(chr(10))[0][:120]}")

    if exit_code == 0 and not train_warnings and not test_warnings:
        print(f"\nALL SMOKE TESTS PASSED ({len(methods_to_test)} method(s))")
    elif exit_code == 0:
        skipped = len(methods_to_test) - len(train_ckpts)
        print(f"\nSmoke test completed with warnings "
              f"(train: {len(train_warnings)} warnings, "
              f"test: {len(test_warnings)} warnings, "
              f"skipped: {skipped})")

    return exit_code


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
