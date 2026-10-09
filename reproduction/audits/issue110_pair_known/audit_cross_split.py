#!/usr/bin/env python3
"""Read-only VG150 pair-known evaluation leakage audit (Python stdlib only).

Rebuilds OpenSGG's relation-table SHA and compares actual ood.train/ood.dev
against known.eval rather than trusting known.train/known.eval validation.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relation_order(payload):
    rows=[]; h=hashlib.sha256()
    for img, triples in payload['train'].items():
        for sub,obj,pred in triples:
            row=(int(img),int(sub),int(obj),int(pred))
            h.update(f'{row[0]}:{row[1]}:{row[2]}:{row[3]};'.encode())
            rows.append(row)
    return rows,h.hexdigest()


def part(payload,name,n):
    seq=payload[name]
    if len(seq)!=len(set(seq)) or any(type(x)!=int or not 0<=x<n for x in seq):
        raise ValueError(f'invalid {name} indices')
    if payload.get('counts',{}).get('n_'+name,len(seq))!=len(seq):
        raise ValueError(f'mismatched {name} count')
    return set(seq)


def intersects(a,b,rows):
    aimg={rows[i][0] for i in a}; bimg={rows[i][0] for i in b}
    shared=aimg&bimg; common=a&b
    return {'train_rows':len(a),'eval_rows':len(b),
            'overlap_rows':len(common),'overlap_images':len(shared),
            'eval_rows_in_train_images':sum(rows[i][0] in shared for i in b),
            'sample_shared_images':sorted(shared)[:10],
            'sample_shared_rows':sorted(common)[:10],
            'clean':not shared and not common}


def pair_coverage(train_coco,rows,fit,evaluation):
    anns=collections.defaultdict(list)
    for ann in train_coco['annotations']:
        anns[int(ann['image_id'])].append(int(ann['category_id']))
    def key(i,directed):
        im,sub,obj,_=rows[i]; a,b=anns[im][sub],anns[im][obj]
        return (a,b) if directed else tuple(sorted((a,b)))
    result={}
    for directed in (False,True):
        seen={key(i,directed) for i in fit}
        count=sum(key(i,directed) in seen for i in evaluation)
        result['directed' if directed else 'undirected']={
            'eval_rows_pair_seen_in_ACTUAL_train':count,
            'eval_rows_pair_UNSEEN_in_ACTUAL_train':len(evaluation)-count,
            'train_unique_pairs':len(seen)}
    return result


def calibrated_conflict_counts(train_coco,rows,fit,evaluation,alpha,temperature):
    """Recalculate B1_lookup for the ACTUAL fit rows, with explicit dev params."""
    if not alpha>0 or not temperature>0:
        raise ValueError('alpha and temperature must be positive')
    labels=collections.defaultdict(list)
    for ann in train_coco['annotations']:
        labels[int(ann['image_id'])].append(int(ann['category_id']))
    def pair(i,directed):
        im,sub,obj,_=rows[i]
        a,b=labels[im][sub],labels[im][obj]
        return (a,b) if directed else tuple(sorted((a,b)))
    results={}
    for directed in (False,True):
        pairs=collections.defaultdict(collections.Counter)
        totals=collections.Counter()
        for i in fit:
            pred=rows[i][3]-1
            if not 0<=pred<50: raise ValueError('VG50 raw predicate id outside 1..50')
            pairs[pair(i,directed)][pred]+=1
            totals[pred]+=1
        n=sum(totals.values())
        global_prob=[totals[j]/n for j in range(50)]
        cached={}; eligible=wrong=0
        thresholds={str(t):0 for t in (0.5,0.7,0.9)}
        for i in evaluation:
            k=pair(i,directed)
            if k not in pairs: continue
            eligible+=1
            if k not in cached:
                counts=pairs[k]; den=sum(counts.values())+alpha
                probs=[(counts[j]+alpha*global_prob[j])/den for j in range(50)]
                if temperature!=1:
                    powered=[max(x,0.)**(1/temperature) for x in probs]
                    norm=sum(powered)
                    probs=[x/norm for x in powered] if norm else [1/50]*50
                cached[k]=(max(range(50),key=lambda j:probs[j]),max(probs))
            prediction,confidence=cached[k]
            if prediction!=rows[i][3]-1:
                wrong+=1
                for t in (0.5,0.7,0.9):
                    if confidence>t: thresholds[str(t)]+=1
        results['directed' if directed else 'undirected']={
            'actual_fit_pair_eligible':eligible,
            'wrong_argmax_seen_pair':wrong,
            'high_confidence_wrong_thresholds':thresholds,
            'unseen_in_actual_fit':len(evaluation)-eligible,
            'alpha':alpha,'temperature':temperature}
    return results


def calibration_from_cells(path):
    payload=load(path)
    cells=[c for c in payload['cells'] if c.get('cell')=='vg50__pair_known__B1_lookup']
    if len(cells)!=1: raise ValueError('cannot uniquely find vg50__pair_known__B1_lookup calibration')
    return float(cells[0]['calibration']['alpha']),float(cells[0]['calibration']['temperature'])


def audit(splits_dir,rel_json,train_json=None,prior_cells=None):
    ood=load(splits_dir/'split_pair_ood.json')
    known=load(splits_dir/'split_pair_known.json')
    rows, sha=relation_order(load(rel_json)); n=len(rows)
    for name,payload in (('pair_ood',ood),('pair_known',known)):
        actual=payload['_meta']['relation_order_sha256']
        if actual!=sha: raise ValueError(f'{name} relation-order hash mismatch: {actual} != {sha}')
    train=part(ood,'train',n); dev=part(ood,'dev',n)
    evaluation=part(known,'eval',n); knowntrain=part(known,'train',n)
    knowndev=part(known,'dev',n)
    checks={'ACTUAL_fit_vs_known_eval':intersects(train,evaluation,rows),
            'ACTUAL_validation_vs_known_eval':intersects(dev,evaluation,rows),
            'known_OWN_train_vs_known_eval':intersects(knowntrain,evaluation,rows),
            'known_OWN_dev_vs_known_eval':intersects(knowndev,evaluation,rows)}
    clean=checks['ACTUAL_fit_vs_known_eval']['clean'] and checks['ACTUAL_validation_vs_known_eval']['clean']
    result={'scope':'READ_ONLY_ORIGINAL_INDEX_AUDIT','status':'CLEAN' if clean else 'CONTAMINATED',
            'relation_sha256':sha,'n_rows':n,'checks':checks,
            'actual_fit':'pair_ood.train','actual_validation':'pair_ood.dev',
            'evaluation':'pair_known.eval',
            'notes':['Source-set disjointness is tested against actual fitting and validation rows.',
                     'Own pair_known split clean != actual fitting subset clean.',
                     'Source-check is not a training rerun or independent benchmark replication.']}
    if train_json:
        labels=load(train_json)
        result['pair_support']=pair_coverage(labels,rows,train,evaluation)
        if prior_cells:
            alpha,temp=calibration_from_cells(prior_cells)
            result['prior_conflict']=calibrated_conflict_counts(labels,rows,train,evaluation,alpha,temp)
            result['prior_conflict_provenance']='Historical pair_known dev-selected alpha/temperature; actual fitting rows from pair_ood.train. Confidence numbers are diagnostics, not valid held-out scores unless disjointness passes.'
        else:
            result['prior_conflict']='NOT_EVALUATED: pass --prior-cells to use dev-calibrated prior parameters'
    else:
        result['pair_support']='NOT_EVALUATED: pass --train-json for ordered/unordered pair support'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--splits-dir',type=Path,required=True)
    p.add_argument('--rel-json',type=Path,required=True)
    p.add_argument('--train-json',type=Path,default=None)
    p.add_argument('--prior-cells',type=Path,default=None,help='original matrix/prior_cells.json containing dev-selected alpha/temperature')
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    if args.prior_cells and not args.train_json: p.error('--prior-cells requires --train-json')
    try: result=audit(args.splits_dir,args.rel_json,args.train_json,args.prior_cells)
    except (ValueError,KeyError,IndexError,FileNotFoundError) as err:
        print('AUDIT_NOT_COMPLETED: '+str(err),file=sys.stderr)
        return 2
    payload=json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(payload,encoding='utf-8')
    print(payload,end='')
    return 0 if result['status']=='CLEAN' else 3

if __name__=='__main__': raise SystemExit(main())
