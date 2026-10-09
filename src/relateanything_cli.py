"""Standalone region runtime selected through OpenSGG's main CLI."""
import json
import math
import runpy
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def read_regions(path):
    boxes = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(boxes, list) or any(
            not isinstance(box, list) or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, (float, int))
                   or not math.isfinite(value) for value in box)
            or box[0] >= box[2] or box[1] >= box[3] for box in boxes)):
        raise ValueError("regions must be an array of pixel xyxy boxes with finite, positive extents")
    return boxes


def serialize_triplets(predictions):
    if isinstance(predictions, dict):
        return {key: serialize_triplets(value) for key, value in predictions.items()}
    records = []
    for triplet in predictions:
        record = asdict(triplet)
        record["subject_box"] = record["subject_box"].tolist()
        record["object_box"] = record["object_box"].tolist()
        records.append(record)
    return records


def load_region_config(args):
    config_path = args.config_file or Path(__file__).resolve().parents[1] / "configs/VisualGenome/RelateAnything.py"
    config = runpy.run_path(str(config_path))
    if config.get("method") != "RelateAnything" or config.get("task") != "supplied_regions":
        raise ValueError("RelateAnything requires its supplied_regions config")
    if (config.get("split") != "test" or config.get("limit") != 0
            or config.get("graph_constraint") is not True):
        raise ValueError("RelateAnything benchmark config requires test, limit=0 and graph constraint")
    return config


def write_report(args, report):
    text = json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    if args.relation_output:
        output = Path(args.relation_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def run_evaluation(args):
    config = load_region_config(args)
    expected = {"img_size": 448, "max_objects": 100, "eval_budget": 500, "weights": "ema"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"evaluation requires official benchmark settings: {expected}")
    if Path(args.ckpt_path).name != "model.pth":
        raise ValueError("evaluation must load the audited model.pth release checkpoint")
    if args.dataset_size or args.test_dataset_size or args.val_dataset_size:
        raise ValueError("RelateAnything baseline evaluation cannot use dataset-size subsets")
    root = Path(__file__).resolve().parents[1]
    gate_command = [sys.executable, str(root / "tools/reproduction/check_relateanything_official_inputs.py"),
                    "--official-root", args.relation_official_root,
                    "--official-commit", "4a07de9d06f2e3f14309753b7907cf1d3a263b08",
                    "--snapshot", str(Path(args.ckpt_path).resolve().parent),
                    "--data-root", args.data_root, "--evidence", args.relation_evidence,
                    "--protocol", args.relation_protocol, "--consumer", "opensgg"]
    if args.relation_tau_calibration:
        gate_command += ["--tau-calibration", args.relation_tau_calibration]
    gate = subprocess.run(gate_command, capture_output=True, text=True)
    if gate.returncode:
        print(gate.stdout, file=sys.stderr)
        raise ValueError(f"RelateAnything input gate blocked evaluation: {gate.stderr.strip()}")
    input_report = json.loads(gate.stdout)
    from torch.utils.data import DataLoader

    from src.modules.relateanything.data import RelationDataset, collate_fn
    from src.relateanything import RelateAnythingModel
    from src.relateanything_evaluation import evaluate_batches

    dataset = RelationDataset(args.data_root, "test", resolution=config["img_size"], max_objects=config["max_objects"])
    if len(dataset) != 26404:
        raise ValueError(f"official VG150 full test pack requires 26404 images, got {len(dataset)}")
    model = RelateAnythingModel.from_checkpoint(
        args.ckpt_path, device=args.device, img_size=config["img_size"], weights=config["weights"],
        predicates=None if args.relation_protocol == "A3" else dataset.predicate_names,
        full_vocabulary=args.relation_protocol == "A3", calibration=False)
    matrix = None
    if args.relation_protocol == "A3":
        from src.modules.relateanything.eval.evaluator import build_cross_match_matrix
        from src.relateanything_evaluation import (
            encode_benchmark_predicates,
            load_benchmark_bank,
        )

        calibration = json.loads(Path(args.relation_tau_calibration).read_text(encoding="utf-8"))
        train_embeddings = load_benchmark_bank(args.ckpt_path, model.predictor.predicates)
        benchmark_embeddings = encode_benchmark_predicates(dataset.predicate_names, model.predictor.text_student,
                                                     device=args.device).cpu().numpy()
        matrix = build_cross_match_matrix(model.predictor.predicates, dataset.predicate_names,
                                           train_embeddings, benchmark_embeddings,
                                           tau_eval=calibration["chosen"]["tau"])
        if not matrix.any(dim=0).all():
            raise ValueError("A3 matcher leaves a GT predicate without an accepted spelling")
    loader = DataLoader(dataset, batch_size=args.val_batch_size or 64,
                        num_workers=args.num_workers if args.num_workers is not None else 8,
                        shuffle=False, collate_fn=collate_fn)
    report = evaluate_batches(model, loader, num_predicates=len(dataset.predicate_names),
                              protocol=args.relation_protocol, match_matrix=matrix,
                              eval_budget=config["eval_budget"], device=args.device)
    if report["seen_images"] != len(dataset):
        raise RuntimeError("scored image denominator differs from the full input pack")
    packed_relations = int(dataset.img_meta[:, 6].sum())
    report.update({"packed_relations": packed_relations,
                   "relations_dropped_by_object_cap": packed_relations - report["scored_relations"],
                   "max_objects": config["max_objects"], "img_size": config["img_size"],
                   "eval_budget": config["eval_budget"], "weights": config["weights"],
                   "amp": args.device.startswith("cuda"), "amp_dtype": "bfloat16"})
    report.update({"method": "RelateAnything", "task": "supplied_regions", "reproduced": False,
                   "protocol": args.relation_protocol, "input_audit": input_report,
                   "checkpoint": str(Path(args.ckpt_path).resolve()), "split": "test",
                   "loaded_images": len(dataset), "graph_constraint": True})
    write_report(args, report)
    return 0


def main(args):
    if not args.ckpt_path:
        print("RelateAnything requires --ckpt_path; random initialization is not supported.", file=sys.stderr)
        return 2
    try:
        if args.test and not args.image:
            if not args.relation_evidence or not args.relation_official_root:
                raise ValueError("evaluation requires --relation_evidence and --relation_official_root before loading weights")
            return run_evaluation(args)
        if not args.image or not args.regions:
            raise ValueError("RelateAnything prediction requires --image and --regions")
        boxes = read_regions(args.regions)
        if args.full_vocabulary and args.predicate_vocabulary:
            raise ValueError("choose --full_vocabulary or --predicate_vocabulary, not both")
        if args.relation_topk <= 0:
            raise ValueError("--relation_topk must be positive")
        predicates = None
        if args.predicate_vocabulary:
            predicates = json.loads(Path(args.predicate_vocabulary).read_text(encoding="utf-8"))
            if (not isinstance(predicates, list) or not predicates
                    or any(not isinstance(value, str) or not value.strip() for value in predicates)
                    or len(set(predicates)) != len(predicates)):
                raise ValueError("predicate vocabulary must be a nonempty list of unique strings")
        config = load_region_config(args)
        import numpy as np
        from PIL import Image

        from src.relateanything import RelateAnythingModel

        masks = np.load(args.relation_masks, allow_pickle=False) if args.relation_masks else None
        scores = json.loads(Path(args.relation_box_scores).read_text(encoding="utf-8")) if args.relation_box_scores else None
        with Image.open(args.image) as source:
            image = source.convert("RGB")
        model = RelateAnythingModel.from_checkpoint(
            args.ckpt_path, predicates=predicates, full_vocabulary=args.full_vocabulary,
            device=args.device, img_size=config["img_size"], weights=config["weights"],
        )
        predictions = model.predict(image, boxes, masks=masks, box_scores=scores,
                                    topk=args.relation_topk, max_boxes=config["max_objects"],
                                    decompose=args.relation_decompose)
        report = {
            "method": "RelateAnything", "task": "supplied_regions", "reproduced": False,
            "checkpoint": str(Path(args.ckpt_path).resolve()),
            "attribution": "Based on RelateAnything by Maëlic Neau (https://github.com/Maelic/RelateAnything)",
            "predicates": model.predictor.predicates, "predictions": serialize_triplets(predictions),
        }
        write_report(args, report)
        return 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"RelateAnything: {error}", file=sys.stderr)
        return 2
