import os
import json
import argparse

import numpy as np
import torch

import yaml
import open_clip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="scripts/output")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # device
    device = torch.device(args.device if (args.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # load yaml
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    items = cfg.get("predicates_description", None)
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError("YAML must contain a non-empty list: predicates_description")

    preds = []
    paras_2d = []
    for i, it in enumerate(items):
        name = (it.get("name") or "").strip()
        paras = it.get("paragraphs", [])
        if not name:
            raise ValueError(f"predicates_description[{i}].name is empty")
        if not isinstance(paras, list) or len(paras) == 0 or not all(isinstance(x, str) and x.strip() for x in paras):
            raise ValueError(f"predicates_description[{i}].paragraphs must be non-empty strings")
        preds.append(name)
        paras_2d.append([p.strip() for p in paras])

    # same paragraphs per predicate (ensemble size)
    M = len(paras_2d[0])
    if not all(len(x) == M for x in paras_2d):
        lens = [len(x) for x in paras_2d]
        raise ValueError(f"All predicates must have same #paragraphs. Got: {lens}")

    # open_clip
    model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(device).eval()

    # flatten texts
    flat_texts = [t for row in paras_2d for t in row]  # R*M
    R = len(paras_2d)

    # encode in batches
    feats = []
    with torch.no_grad():
        for i in range(0, len(flat_texts), args.batch_size):
            batch = flat_texts[i:i + args.batch_size]
            tok = tokenizer(batch).to(device)
            f = model.encode_text(tok)
            f = f / f.norm(dim=-1, keepdim=True)  # per-paragraph norm
            feats.append(f.detach().cpu())
    feats = torch.cat(feats, dim=0).view(R, M, -1)  # [R, M, E]

    # mean pooling + normalize
    text_rel = feats.mean(dim=1)
    text_rel = text_rel / text_rel.norm(dim=-1, keepdim=True)  # [R, E]

    # save
    np.save(os.path.join(args.out_dir, "text_rel.npy"), text_rel.numpy().astype(np.float32))

    meta = {
        "dataset_name": cfg.get("dataset_name", ""),
        "entity_count": cfg.get("entity_count", None),
        "relation_count": cfg.get("relation_count", None),
        "relations_in_yaml": cfg.get("relations", []),
        "predicates_in_desc": preds,
        "paragraphs_per_predicate": M,
        "clip_model": args.model,
        "pretrained": args.pretrained,
        "device": str(device),
        "text_rel_shape": list(text_rel.shape),
        "note": "OpenCLIP paragraph-ensemble prototypes: paragraph L2norm -> mean -> L2norm.",
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    pred_to_paras = {p: paras_2d[i] for i, p in enumerate(preds)}
    with open(os.path.join(args.out_dir, "predicate_paragraphs.json"), "w", encoding="utf-8") as f:
        json.dump(pred_to_paras, f, ensure_ascii=False, indent=2)

    norms = text_rel.norm(dim=-1).numpy()
    print(f"[OK] text_rel.npy saved to: {os.path.join(args.out_dir, 'text_rel.npy')}")
    print(f"[Shape] {tuple(text_rel.shape)}  paragraphs_per_predicate={M}")
    print(f"[Check] mean(norm)={norms.mean():.4f}, min(norm)={norms.min():.4f}, max(norm)={norms.max():.4f}")


if __name__ == "__main__":
    main()
