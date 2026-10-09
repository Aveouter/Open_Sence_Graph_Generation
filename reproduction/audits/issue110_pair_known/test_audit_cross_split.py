import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from audit_cross_split import audit


def fixture(root, leak=False):
    splitdir=root/'splits';splitdir.mkdir()
    rels={'10':[[0,1,1],[1,0,2]], '11':[[0,1,1]], '12':[[0,1,2]],'13':[[0,1,1]],'14':[[1,0,2]]}
    rel=root/'rel.json';rel.write_text(json.dumps({'train':rels}))
    train=root/'train.json'
    train.write_text(json.dumps({'annotations':[{'image_id':int(i),'category_id':cls} for i in rels for cls in (1,2)]}))
    h=hashlib.sha256()
    for image, triples in rels.items():
        for s,o,p in triples:h.update(f'{image}:{s}:{o}:{p};'.encode())
    meta={'relation_order_sha256':h.hexdigest()}
    ood={'_meta':meta,'train':[0,1,2],'dev':[3],'eval':[4,5]}
    known={'_meta':meta,'train':[0,1,3],'dev':[],'eval':[2,5] if leak else [4,5]}
    (splitdir/'split_pair_ood.json').write_text(json.dumps(ood))
    (splitdir/'split_pair_known.json').write_text(json.dumps(known))
    return splitdir,rel,train


class AuditTests(unittest.TestCase):
    def test_detects_actual_cross_split_leakage_while_own_split_is_clean(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp),True)
            report=audit(*args)
            self.assertEqual(report['status'],'CONTAMINATED')
            self.assertEqual(report['checks']['ACTUAL_fit_vs_known_eval']['overlap_images'],1)
            self.assertEqual(report['checks']['ACTUAL_fit_vs_known_eval']['overlap_rows'],1)
            self.assertTrue(report['checks']['known_OWN_train_vs_known_eval']['clean'])
            self.assertIn('directed',report['pair_support'])
    def test_clean_split(self):
        with tempfile.TemporaryDirectory() as temp:
            args=fixture(Path(temp),False)
            report=audit(*args)
            self.assertEqual(report['status'],'CLEAN')
            self.assertEqual(report['checks']['ACTUAL_fit_vs_known_eval']['overlap_images'],0)
    def test_calibrated_prior_conflict_reads_original_matrix_cell(self):
        with tempfile.TemporaryDirectory() as temp:
            s,r,c=fixture(Path(temp),False)
            matrix=Path(temp)/'prior_cells.json'
            matrix.write_text(json.dumps({'cells':[{'cell':'vg50__pair_known__B1_lookup',
                 'calibration':{'alpha':1.0,'temperature':1.0}}]}))
            report=audit(s,r,c,matrix)
            self.assertEqual(report['prior_conflict']['undirected']['alpha'],1.0)
            self.assertIn('high_confidence_wrong_thresholds',report['prior_conflict']['directed'])

    def test_actual_validation_overlap_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            s,r,c=fixture(Path(temp),False)
            path=s/'split_pair_known.json'; data=json.loads(path.read_text())
            data['eval']=[3,5]; data['train']=[0,1,2]
            path.write_text(json.dumps(data))
            report=audit(s,r,c)
            self.assertEqual(report['status'],'CONTAMINATED')
            self.assertEqual(report['checks']['ACTUAL_fit_vs_known_eval']['overlap_rows'],0)
            self.assertEqual(report['checks']['ACTUAL_validation_vs_known_eval']['overlap_rows'],1)

    def test_bad_relation_sha_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            splitdir,rel,train=fixture(Path(temp))
            path=splitdir/'split_pair_known.json'
            data=json.loads(path.read_text())
            data['_meta']['relation_order_sha256']='bad'
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                audit(splitdir,rel,train)

if __name__=='__main__': unittest.main()
