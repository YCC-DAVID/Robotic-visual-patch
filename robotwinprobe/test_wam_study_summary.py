"""Keep final attack counts and cleanup ownership tied to actual evidence."""
from pathlib import Path
import unittest

from wam_study_summary import owned_command,paired_results


class SummaryTests(unittest.TestCase):
    def test_clean_failure_and_missing_episode_are_not_attack_successes(self):
        clean={1:{'success':True},2:{'success':True},3:{'success':False}}
        candidate={1:{'success':False},3:{'success':False}}
        result=paired_results(clean,candidate)
        self.assertEqual(result['completed'],2)
        self.assertEqual(result['eligible'],1)
        self.assertEqual(result['failure_seeds'],[1])
        self.assertEqual(result['ASR'],1.)

    def test_no_paired_clean_success_does_not_report_zero_percent(self):
        result=paired_results({1:{'success':False}},{1:{'success':False}})
        self.assertIsNone(result['ASR']);self.assertEqual(result['failures'],0)

    def test_cleanup_rejects_other_roots_and_other_scripts(self):
        root=Path('/experiment/current')
        self.assertTrue(owned_command(['python','/code/wam_steps_to_patch.py','server','--root',str(root)],root))
        self.assertFalse(owned_command(['python','/code/wam_steps_to_patch.py','server','--root','/experiment/other'],root))
        self.assertFalse(owned_command(['python','/code/vision_freeze_ft.py','--root',str(root)],root))
        self.assertFalse(owned_command(['python','/code/wam_study_summary.py','--wait','--root',str(root)],root))


if __name__=='__main__':unittest.main()
