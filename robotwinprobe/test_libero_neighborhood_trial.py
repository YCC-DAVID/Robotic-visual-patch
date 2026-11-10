"""CPU end-to-end audit of the pilot with analytic policy substitutes.

These tests verify orchestration/accounting, not VLA or renderer effectiveness.
They use the actual immutable input plans but never load a policy or touch CUDA.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import libero_neighborhood_trial as trial


class AnalyticServers:
    def __init__(self, root):
        self.root=root
        self.calls=[]

    def __call__(self, kind, command, **payload):
        assert kind in trial.SOURCES and command=='batch'
        assert payload['gradient'] and payload['source_model']==kind
        q=payload['texture']
        target=0.1 if kind=='pi' else 0.9
        delta=q.astype(np.float64)-target
        loss=float(0.01+np.mean(delta**2))
        gradient=(2*delta/q.size).astype(np.float32)
        provenance=dict(source_model=kind,server_model=kind,iteration=payload['iteration'],
            texture_sha256=trial.array_sha(q),indices=payload['indices'],seeds=payload['seeds'],
            draws_sha256=trial.digest(payload['draws']),gradient_norm=float(np.linalg.norm(gradient)))
        with (self.root/'gradient_ledgers'/(kind+'.jsonl')).open('a') as handle:
            handle.write(json.dumps(provenance)+'\n')
        self.calls.append((kind,payload['iteration']))
        return dict(losses=[loss]*8,gradient=gradient,gradient_provenance=provenance)


class TrialTests(unittest.TestCase):
    def test_budget_replay_and_cached_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trial.prepare(root,steps=2)
            servers=AnalyticServers(root)
            with patch.object(trial,'rpc',servers),patch.object(trial,'status'):
                trial.train(root)
            self.assertEqual(len(servers.calls),16)
            self.assertEqual(sum(k=='pi' for k,_ in servers.calls),8)
            self.assertEqual(sum(k=='imagewam' for k,_ in servers.calls),8)
            self.assertTrue(trial.verify_training(root)['actual_call_budget_matched'])
            with np.load(root/'training/neighborhood.npz') as z:
                history=json.loads(str(z['history']))
            self.assertTrue(all(h['inner'][k]['loss_lowered'] for h in history for k in trial.SOURCES))
            # Lost outer checkpoint after successful requests: cached, audited
            # inputs can be replayed without duplicate backend gradient calls.
            (root/'training/neighborhood.npz').unlink()
            with patch.object(trial,'rpc',servers),patch.object(trial,'status'):
                trial.train(root)
            self.assertEqual(len(servers.calls),16)
            self.assertTrue(trial.verify_training(root)['updates_and_inner_neighbors_replayed'])

    def test_corrupt_update_or_unlogged_request_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trial.prepare(root,steps=1)
            servers=AnalyticServers(root)
            with patch.object(trial,'rpc',servers),patch.object(trial,'status'):
                trial.train(root)
            path=root/'training/vanilla.npz'
            with np.load(path) as z:
                values={k:z[k].copy() for k in z.files}
            values['signs'][0,0,0,0,0]*=-1
            trial.atomic_npz(path,**values)
            with self.assertRaises(AssertionError):
                trial.verify_training(root)
            values['signs'][0,0,0,0,0]*=-1
            trial.atomic_npz(path,**values)
            ledger=root/'gradient_ledgers/pi.jsonl'
            ledger.write_text(ledger.read_text()+ledger.read_text().splitlines()[0]+'\n')
            with self.assertRaises(AssertionError):
                trial.verify_training(root)

    def test_protocol_cannot_change_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trial.prepare(root,steps=1)
            with self.assertRaises(ValueError):
                trial.prepare(root,steps=2)

    def test_orphaned_rpc_refused_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trial.prepare(root,steps=1)
            (root/'gradient_ledgers/pi.jsonl').write_text(json.dumps({'iteration':1})+'\n')
            with patch.object(trial,'rpc') as rpc:
                with self.assertRaisesRegex(ValueError,'orphaned RPCs'):
                    trial.train(root)
                rpc.assert_not_called()

    def test_native_config_tampering_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            trial.prepare(root,steps=1)
            path=root/'models/pi/protocol.json'
            cfg=trial.read(path);cfg['evaluation']['replan']=1
            trial.dump(path,cfg)
            with self.assertRaises(AssertionError):
                trial.check_prepared(root)


if __name__=='__main__':
    unittest.main()
