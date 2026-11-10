"""CPU regression checks for event timing and the actual ASR denominator."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

import stage_injection as S


class StageInjectionTests(unittest.TestCase):
    def config(self):
        return dict(approach_distance_m=.12,lift_height_m=.04,
                    above_plate_xy_m=.08,lift_consecutive_actions=3)

    def test_events_inside_chunk_wait_for_next_query(self):
        m = S.Milestones([0,0,.75],self.config())
        for step,z in [(3,.75),(17,.8),(18,.8),(19,.8)]:
            m.update(step,np.array([0,0,z]),np.array([1,0,.75]),np.array([0,0,.8]))
        m.update(66,np.array([.98,0,.8]),np.array([1,0,.75]),np.array([.98,0,.82]))
        self.assertEqual(m.events,dict(initial=0,approach=3,lifted=19,above_plate=66))
        self.assertEqual(m.schedule([0,32,64,96]),dict(initial=0,approach=1,lifted=1,above_plate=3))

    def test_terminal_event_is_not_an_injection_opportunity(self):
        m = S.Milestones([0,0,.75],self.config())
        m.events['lifted'] = 80
        self.assertIsNone(m.schedule([0,32,64])['lifted'])
        self.assertIsNone(m.schedule([0,32,64])['above_plate'])

    def test_lift_requires_consecutive_actions(self):
        m = S.Milestones([0,0,.75],self.config())
        for step,z in enumerate([.8,.8,.75,.8,.8],1):
            m.update(step,np.array([0,0,z]),np.array([1,0,.75]),np.array([0,0,1]))
        self.assertNotIn('lifted',m.events)
        m.update(6,np.array([0,0,.8]),np.array([1,0,.75]),np.array([0,0,1]))
        self.assertEqual(m.events['lifted'],6)

    def test_report_excludes_unreachable_and_clean_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'input.png'
            Image.fromarray(np.zeros((8,8,3),dtype=np.uint8)).save(source)
            args = S.parser().parse_args(['prepare','--root',str(root),'--texture',str(source),'--seeds','1,2,3'])
            S.prepare(args)
            h = S.sha(root/'protocol.json')
            for seed in [1,2,3]:
                schedule = dict.fromkeys(S.STAGES,0)
                if seed == 2:
                    schedule['above_plate'] = None
                S.dump(root/'episodes'/f'seed{seed}_clean.json',dict(seed=seed,method='clean',stage=None,
                    protocol_sha256=h,status='complete',success=seed!=3,schedule=schedule))
                for stage in S.STAGES:
                    for method in ['trained','random']:
                        skipped = schedule[stage] is None
                        S.dump(root/'episodes'/f'seed{seed}_{method}_{stage}.json',dict(seed=seed,
                            method=method,stage=stage,protocol_sha256=h,
                            status='no_injection_opportunity' if skipped else 'complete',
                            success=None if skipped else method=='random',injection=dict(query=0)))
            result = S.report(root)
            self.assertTrue(result['complete'])
            self.assertEqual(result['common_reachable_seeds'],[1])
            self.assertEqual(result['no_injection_opportunities'],2)
            rows = {(r['stage'],r['cohort']):r for r in result['rows']}
            self.assertEqual(rows['initial','stage_reachable']['n'],2)
            self.assertEqual(rows['above_plate','stage_reachable']['n'],1)
            self.assertEqual(rows['initial','all_stages_reachable']['n'],1)
            self.assertEqual(rows['initial','stage_reachable']['excess_ASR'],1)
            with self.assertRaises(AssertionError):
                args.size = .2
                S.prepare(args)
            (root/'textures/trained.png').write_bytes(b'modified')
            with self.assertRaises(AssertionError):
                S.load_protocol(root)


if __name__ == '__main__':
    unittest.main()
