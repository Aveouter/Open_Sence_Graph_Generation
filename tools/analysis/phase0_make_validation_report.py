#!/usr/bin/env python3
"""Create Phase 0 validation report from primitive discovery outputs."""

import argparse
import csv
import json
from pathlib import Path


def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def row_for(model, root):
    root = Path(root)
    cov = load_json(root / "features" / "coverage_report.json") or load_json(root / "coverage" / "coverage_summary.json")
    val = load_json(root / "discovery" / "family_validity.json")
    return {
        "model": model,
        "coverage": cov.get("coverage", 0.0) if cov else None,
        "num_features": cov.get("num_features", cov.get("on_family_matched", 0)) if cov else 0,
        "family_validity": val.get("best_config", {}).get("family_validity") if val else None,
        "r2": val.get("best_config", {}).get("r2") if val else None,
        "shared_f": val.get("best_config", {}).get("shared_f") if val else None,
        "private_f": val.get("best_config", {}).get("private_f") if val else None,
        "v_margin": val.get("v_margin") if val else None,
        "passes_gate": val.get("passes_reltr_gate") if val else False,
    }


def fmt(x):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser(description="Make Phase 0 validation report")
    parser.add_argument("--reltr_dir", required=True)
    parser.add_argument("--motifs_dir", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    rows = [row_for("RelTR", args.reltr_dir), row_for("Motifs", args.motifs_dir)]
    reltr = rows[0]
    motifs = rows[1]
    if reltr["passes_gate"]:
        decision = "GO" if motifs["passes_gate"] else "WEAK GO"
    else:
        decision = "NO-GO"

    out_csv = Path(args.output_csv); out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    out_json = Path(args.output_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"decision": decision, "rows": rows}, f, indent=2)

    md = []
    md.append("# Phase 0 Primitive Discovery Validation Report\n")
    md.append(f"**Decision:** {decision}\n")
    md.append("## Summary\n")
    md.append("| Model | Coverage | V(F) | R² | Shared_F | Private_F | V_margin | Gate |\n")
    md.append("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for r in rows:
        md.append(
            f"| {r['model']} | {fmt(r['coverage'])} | {fmt(r['family_validity'])} | "
            f"{fmt(r['r2'])} | {fmt(r['shared_f'])} | {fmt(r['private_f'])} | "
            f"{fmt(r['v_margin'])} | {r['passes_gate']} |\n"
        )
    md.append("\n## Interpretation\n\n")
    if decision == "GO":
        md.append("RelTR passes the primary Phase 0 gate and Motifs is directionally supportive. Proceed to Phase 1 RelTR-Primitive vs B6.\n")
    elif decision == "WEAK GO":
        md.append("RelTR passes the primary gate, but Motifs does not provide full supplementary support. Proceed only with narrowed RelTR-first claims or run extra diagnostics.\n")
    else:
        md.append("RelTR fails the primary Phase 0 gate. Do not implement full Primitive-SGG; reconsider the method premise or use diagnosis-only path.\n")
    md.append("\n## Risks\n\n")
    md.append("- Phase 0 uses a lightweight diagnostic primitive discovery module, not the full primitive-query bottleneck.\n")
    md.append("- Passing Phase 0 supports the existence of shared/private structure; it does not prove the full method will improve SGG metrics.\n")
    md.append("- Random family baselines are approximate and should be repeated in final experiments.\n")

    out_md = Path(args.output_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.writelines(md)
    print(f"Wrote {out_md} decision={decision}")


if __name__ == "__main__":
    main()
