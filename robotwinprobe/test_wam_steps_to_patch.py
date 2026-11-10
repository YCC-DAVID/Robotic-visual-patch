"""Protect minimum-step claims and immutable reference accounting."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from wam_orthogonal_control import intervention_masks
from wam_steps_to_patch import condition_rows,reference


class StepSearchTests(unittest.TestCase):
    def test_default_control_is_preserved_and_bad_masks_are_rejected(self):
        self.assertEqual(intervention_masks({}),dict(clean=[],orthogonal_first=[0],orthogonal_all=list(range(10))))
        for values in [[-1],[10],[0,0],[2,1]]:
            with self.assertRaises(AssertionError):intervention_masks({'intervention_masks':{'bad':values}})

    def test_partial_failures_and_runtime_errors_cannot_confirm_minimum(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);dest=root/'episodes/single_10';dest.mkdir(parents=True)
            protocol={'intervention_masks':{'single_10':[9]}}
            for seed in [22000,22001]:(dest/f'seed{seed}.json').write_text(json.dumps(dict(seed=seed,success=False)))
            self.assertFalse(condition_rows(root,protocol)[0]['reliable'])
            final=dest/'seed22002.json';final.write_text(json.dumps(dict(seed=22002,error='CUDA OOM')))
            row=condition_rows(root,protocol)[0]
            self.assertEqual(row['completed'],2);self.assertFalse(row['reliable'])
            final.write_text(json.dumps(dict(seed=22002,success=False)))
            row=condition_rows(root,protocol)[0]
            self.assertTrue(row['reliable']);self.assertEqual(row['failures'],3)

    def test_reuse_keeps_results_and_source_bytes_with_explicit_lineage(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'original.json'
            value=dict(seed=22000,condition='orthogonal_first',success=True,protocol_sha256='old')
            source.write_text(json.dumps(value));source.with_suffix('.npz').write_bytes(b'original-array-content')
            before=source.read_bytes();target=root/'new/episodes/single_01/seed22000.json'
            reference(source,target,'single_01','new')
            result=json.loads(target.read_text())
            self.assertEqual(result['success'],value['success'])
            self.assertEqual(result['source_protocol_sha256'],'old')
            self.assertEqual(result['source_record_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(result['protocol_sha256'],'new');self.assertTrue(result['reference_reused'])
            self.assertEqual(source.read_bytes(),before)
            self.assertEqual(target.with_suffix('.npz').read_bytes(),source.with_suffix('.npz').read_bytes())


if __name__=='__main__':unittest.main()
