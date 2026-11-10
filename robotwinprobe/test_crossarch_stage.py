"""Stage windows and ASR eligibility; simulator integration is checked by smoke rollouts."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from crossarch_common import STAGES, dump
from crossarch_stage_common import PhaseTracker, condition, exposed
import crossarch_stage_report as R


class StageTests(unittest.TestCase):
    def test_monotonic_boundaries_interrupt_even_within_a_chunk(self):
        tracker = PhaseTracker([-.2, 0, .74])
        state = np.zeros(14)
        state[6] = 1
        self.assertFalse(tracker.update(state, [-.2, 0, .74], [.2, 0, .74], 1))
        state[6] = .84
        self.assertTrue(tracker.update(state, [-.2, 0, .74], [.2, 0, .74], 2))
        self.assertEqual(tracker.phase, 'grasp')
        self.assertTrue(tracker.update(state, [-.2, 0, .78], [.2, 0, .74], 3))
        self.assertEqual(tracker.phase, 'transport')
        self.assertTrue(tracker.update(state, [.18, 0, .78], [.2, 0, .74], 4))
        state[6] = 1
        self.assertFalse(tracker.update(state, [-.2, 0, .74], [.2, 0, .74], 5))
        self.assertEqual(tracker.phase, 'place')
        self.assertEqual(tracker.events, dict(approach=0, grasp=2, transport=3, place=4))

    def test_random_and_specialist_use_identical_stage_windows(self):
        for stage in STAGES:
            specialist = condition(stage + '_stage_only')
            random = condition('random_' + stage)
            for phase in STAGES:
                self.assertEqual(exposed(specialist, phase), phase == stage)
                self.assertEqual(exposed(random, phase), exposed(specialist, phase))
                self.assertTrue(exposed(condition(stage + '_full'), phase))
                self.assertFalse(exposed(condition('clean'), phase))

    def test_asr_excludes_clean_failure_unreachable_and_unexposed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'result').mkdir()
            conditions = ['clean', 'random_full'] + ['random_' + s for s in STAGES] + [
                s + '_' + mode for s in STAGES for mode in ('full', 'stage_only')]
            self.assertEqual(len(conditions) * 2 * 10, 280)
            dump(root / 'protocol.json', dict(evaluation=dict(conditions=conditions)))
            for seed in range(1, 5):
                dump(root / f'episodes/pi/clean/seed{seed}.json', dict(seed=seed, success=seed != 3,
                    query_phases=['approach', 'transport'] if seed == 2 else ['approach', 'grasp']))
                dump(root / f'episodes/pi/grasp_stage_only/seed{seed}.json', dict(seed=seed, success=False,
                    active_queries=0 if seed in (2, 4) else 1))
                dump(root / f'episodes/pi/random_grasp/seed{seed}.json', dict(seed=seed, success=True,
                    active_queries=0 if seed == 2 else 1))
            with patch.object(R, 'ROOT', root):
                R.report(root)
            summary = json.loads((root / 'summary.json').read_text())
            row = next(r for r in summary['rows'] if r['model'] == 'pi' and
                       r['specialist_stage'] == 'grasp' and r['exposure_mode'] == 'stage_only')
            self.assertEqual(row['paired'], 4)
            self.assertEqual(row['eligible'], 1)
            self.assertEqual(row['failures'], 1)
            self.assertEqual(row['random_failures'], 0)
            self.assertEqual(row['no_exposure'], 1)


if __name__ == '__main__':
    unittest.main()
