"""Exercise failure signals and result accounting without real experiment jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import stage_joint_monitor as M


class MonitorTests(unittest.TestCase):
    def test_eta_requires_same_stage_observations(self):
        row = dict(checked_at=1000, health='running', pipeline_status={'stage':'formal_closed_loops'},
                   calibration={'train':1536,'validation':128}, training={'iteration':500},
                   diagnostic_completed_states=200, diagnostic_expected_states=200,
                   recorded=100, expected=2400)
        estimate, _ = M.estimates(row, [dict(time=400, phase='capture', completed=10)])
        self.assertIsNone(estimate['stage_remaining_s'])
        estimate, _ = M.estimates(row, [dict(time=400, phase='formal_closed_loops', completed=40)])
        self.assertEqual(estimate['stage_remaining_s'], 23000)
        self.assertEqual(estimate['stage_range_s'], [11500,46000])
        self.assertIsNone(estimate['total_remaining_s'])
        self.assertIn('审计', estimate['note'])

    def test_report_includes_goal_plan_status_and_eta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            M.write(root / 'status.json', dict(stage='complete'))
            row = M.snapshot(root)
            row['eta'], _ = M.estimates(row, [])
            message = M.message(row, '测试')
            for text in ['当前任务目标','任务计划完成进度','预估当前阶段剩余','预估整体剩余时间']:
                self.assertIn(text, message)
            self.assertIn('修复、验证后恢复实验', message)
            self.assertNotIn('不要自动重启实验', message)
            self.assertEqual(row['eta']['total_remaining_s'], 0)

    def test_missing_service_alert_and_non_gating_proxy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            M.write(root / 'status.json', dict(stage='capture', pid=123, updated_unix=100))
            M.write(root / 'audit/proxy_review.json', dict(passed=False, training_gate=False))
            with patch.object(M, 'process', side_effect=lambda pid, *args: {'pid':123, 'args':[]} if pid == 123 else None):
                row = M.snapshot(root, now=110)
            self.assertEqual(row['health'], 'service_missing')
            self.assertEqual(len(row['problems']), 1)
            self.assertIn('模型服务', row['problems'][0])
            self.assertNotIn('跨架构实验结果', M.message(row, '测试'))

    def test_complete_task_failures_are_results_runtime_errors_are_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            M.write(root / 'status.json', dict(stage='complete', pid=123, updated_unix=100))
            M.write(root / 'episodes/failed_task.json', dict(status='complete', success=False))
            M.write(root / 'episodes/absent_stage.json', dict(status='no_opportunity'))
            row = M.snapshot(root, now=110)
            self.assertEqual(row['health'], 'complete')
            self.assertEqual(row['recorded'], 2)
            self.assertEqual(row['actual_episodes'], 1)
            self.assertEqual(row['no_opportunities'], 1)
            M.write(root / 'episodes/broken.json', dict(status='runtime_error', error='CUDA OOM'))
            row = M.snapshot(root, now=110)
            self.assertEqual(row['health'], 'failed')
            self.assertEqual(row['recorded'], 2)
            self.assertEqual(row['errors'][0]['error'], 'CUDA OOM')

    def test_pipeline_failure_reports_exact_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            M.write(root / 'status.json', dict(stage='stopped_on_error', error='pilot exited 1'))
            row = M.snapshot(root)
            self.assertEqual(row['health'], 'failed')
            self.assertEqual(row['problems'], ['pilot exited 1'])

    def test_owned_process_identity_and_pid_reuse(self):
        root = Path('/experiment/root')
        def data(path):
            if path.name == 'cmdline':
                return b'python\0/a/stage_joint_backend.py\0--root\0/experiment/root\0'
            raise AssertionError(path)
        # The process name may contain spaces; /proc stat fields start after ') '.
        stat = '42 (python worker) S ' + ' '.join(['0'] * 18 + ['777'])
        with patch.object(Path, 'read_bytes', data), patch.object(Path, 'read_text', return_value=stat):
            self.assertEqual(M.process(42, root, 'stage_joint_backend.py')['start_ticks'], '777')
            self.assertIsNone(M.process(42, root, 'unrelated.py'))
            self.assertIsNone(M.process(42, Path('/different/root'), 'stage_joint_backend.py'))


if __name__ == '__main__':
    unittest.main()
