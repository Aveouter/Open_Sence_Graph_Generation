#!/usr/bin/env python3
"""EXP-016B: Visual groundability audit."""
import json, sys, warnings, argparse
from pathlib import Path; from collections import defaultdict
warnings.filterwarnings("ignore")
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
import numpy as np; import torch

def load_text_protos(device):
    from src.models.gen_sgg import CLIPTextPrototypes
    rel=_PROJECT_ROOT/"data/VisualGenome"/"rel.json"
    with open(rel) as f: d=json.load(f)
    pn=[c for c in d["rel_categories"] if c!="__background__"]
    cp=CLIPTextPrototypes(pn,dim=512).to(device)
    tp_raw=cp().detach().cpu().numpy()
    tp=tp_raw/(np.linalg.norm(tp_raw,axis=1,keepdims=True)+1e-8)
    return cp().detach().cpu().numpy()

def load_roi(path):
    feats={}
    with open(path) as f:
        for line in f:
            d=json.loads(line)
            feats[(d["img_id"],d["sub_idx"],d["obj_idx"])]={
                "s":np.array(d["subj_feat"]),"o":np.array(d["obj_feat"]),
                "u":np.array(d["union_feat"]),"e":d["exposed_label"]}
    return feats

def filter_C(hm,ek):
    from tools.analysis.exp012_split_audit import build_train_pair_freq
    from pycocotools.coco import COCO
    tp=_PROJECT_ROOT/"outputs/gen_sgg/manifests"/"exp009_train_hidden_head_or_coarse_one_seed42.json"
    with open(tp) as f: tm=json.load(f)
    tc=COCO(str(_PROJECT_ROOT/"data/VisualGenome"/"train.json"))
    tpf,_=build_train_pair_freq(tm,tc)
    k=[]
    for p in hm["pairs"]:
        key=(p["image_id"],p["sub_idx"],p["obj_idx"])
        if not p.get("hidden_labels"): continue
        line=ek.get(key)
        if line is None: continue
        sl,ol=line["subj_label"],line["obj_label"]
        c=tpf.get((sl,ol))
        if c is None or sum(c.values())==0: continue
        for h in p["hidden_labels"]:
            if c.get(h,0)==0: k.append((p,line,h))
    return k

def filter_E(hm,ek):
    k=[]
    for p in hm["pairs"]:
        key=(p["image_id"],p["sub_idx"],p["obj_idx"])
        if not p.get("hidden_labels"): continue
        line=ek.get(key)
        if line is None: continue
        for h in p["hidden_labels"]:
            if line["ranked"].index(h)>3: k.append((p,line,h))
    return k

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clip_roi_cache",required=True)
    p.add_argument("--val_hidden_manifest",required=True)
    p.add_argument("--output_dir",required=True)
    args=p.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Loading CLIP text prototypes...",flush=True)
    tp=load_text_protos(device)
    print("Loading CLIP ROI features...",flush=True)
    rf=load_roi(args.clip_roi_cache)
    print(f"  pairs: {len(rf)}",flush=True)
    with open(args.val_hidden_manifest) as f: hm=json.load(f)
    ep=_PROJECT_ROOT/"outputs/gen_sgg/exp010_prior_strata/seed42/scores_exposed_conditional.jsonl"
    ek={}
    with open(ep) as f:
        for line in f: d=json.loads(line); ek[(d["image_id"],d["sub_idx"],d["obj_idx"])]=d
    kC=filter_C(hm,ek); kE=filter_E(hm,ek)
    print(f"C: {len(kC)}  E: {len(kE)}",flush=True)
    res={"C":[],"E":[],"summary":{}}
    for sn,kept in [("C",kC),("E",kE)]:
        for (p_,line,h) in kept:
            key=(p_["image_id"],p_["sub_idx"],p_["obj_idx"])
            roi=rf.get(key)
            if roi is None: continue
            exposed=roi["e"]
            s512=roi["s"][:512]; o512=roi["o"][:512]; u512=roi["u"][:512]
            v=(s512+o512+u512)/3.0
            v=v/(np.linalg.norm(v)+1e-8)
            cs=v@tp.T
            ce=float(cs[exposed]); ch=float(cs[h])
            rh=int((cs>ch).sum())+1; re=int((cs>ce).sum())+1
            t5=np.argsort(-cs)[:5].tolist()
            res[sn].append({"img_id":p_["image_id"],"sub_idx":p_["sub_idx"],"obj_idx":p_["obj_idx"],
                "exposed":exposed,"hidden":h,"cos_exposed":ce,"cos_hidden":ch,
                "rank_exposed":re,"rank_hidden":rh,"hidden_in_CLIP_top5":h in t5,
                "CLIP_prefers_hidden":ch>ce})
    for sn in ["C","E"]:
        d=res[sn]
        if not d: continue
        n=len(d)
        h5=sum(1 for x in d if x["hidden_in_CLIP_top5"])
        ha=sum(1 for x in d if x["CLIP_prefers_hidden"])
        mr=np.mean([x["rank_hidden"] for x in d])
        mc=np.mean([x["cos_hidden"] for x in d])
        me=np.mean([x["cos_exposed"] for x in d])
        res["summary"][sn]={"n":n,"h5_ct":h5,"h5_rate":h5/n,"ha_ct":ha,"ha_rate":ha/n,
            "mean_rank_h":float(mr),"mean_cos_h":float(mc),"mean_cos_e":float(me),
            "cos_gap":float(me-mc)}
        print(f"\nSplit {sn} (n={n}):",flush=True)
        print(f"  CLIP top-5: {h5}/{n}={h5/n:.3f}",flush=True)
        print(f"  CLIP prefers hidden: {ha}/{n}={ha/n:.3f}",flush=True)
        print(f"  mean cos: hidden={mc:.4f} exposed={me:.4f} gap={me-mc:.4f}",flush=True)
        print(f"  mean rank: hidden={mr:.1f}",flush=True)
    gc=res["summary"].get("C",{}).get("h5_rate",0)
    ge=res["summary"].get("E",{}).get("h5_rate",0)
    gp=gc>=0.15 or ge>=0.15
    res["summary"]["G16B-VISUAL"]={"pass":gp,"threshold":">=15%","C":gc,"E":ge}
    print(f"\nG16B-VISUAL: {'PASS' if gp else 'FAIL'} (C={gc:.3f} E={ge:.3f})",flush=True)
    with open(out/"groundability.json","w") as f: json.dump(res,f,indent=2,default=float)
    print(f"Saved to {out}/groundability.json",flush=True)

if __name__=="__main__": main()
