# 任务配置与邮件模板

在目标run下保存 `monitor/config.json`。所有路径使用绝对路径；邮件身份与调度信息在能力核对后填写，不要用占位ID冒充已启用。凭据只存在邮件服务的授权存储，不写入这个文件。

```json
{
  "task_id": "robotwin-crossarch-joint",
  "name": "RoboTwin π0.5 / ImageWAM 全轨迹 patch",
  "run_dir": "/absolute/path/to/run",
  "status_file": "/absolute/path/to/run/status.json",
  "stage_key": "stage",
  "pid_key": "pid",
  "process_contains": "crossarch_pipeline.py",
  "stale_after_seconds": 7200,
  "progress": [
    {"label": "patch更新次数", "file": "/absolute/path/to/run/training_progress.json", "key": "iteration", "total": 500}
  ],
  "activity_files": ["/absolute/path/to/run/pipeline.log"],
  "completion_files": ["/absolute/path/to/run/required_final_artifact"],
  "completed_values": ["complete", "completed", "finished"],
  "failed_values": ["failed", "stopped_on_error"],
  "paused_values": ["paused", "clean_pilot_complete_training_held"],
  "report": "/absolute/path/to/report.md",
  "monitor_dir": "/absolute/path/to/run/monitor",
  "recipient": "che.yang@northeastern.com",
  "timezone": "America/New_York",
  "interval_seconds": 3600,
  "enabled": false,
  "scheduler": null,
  "delivery": null
}
```

`completed_values` 应匹配整个任务真正的终态；有多阶段流水线时不能将子阶段完成填入这里。`completion_files` 只能检查文件是否存在，最终报告样本数等语义判据仍须单独核对。`stale_after_seconds` 根据实际更新频率调整，不是停止实验的超时。

检查：

```bash
python /path/to/experiment-watch/scripts/check_task.py /absolute/path/to/run/monitor/config.json
```

脚本保存 `checks/*.json`、`latest.json`、`latest_summary.txt`；它不会读取或执行任意日志命令，不会配置cron或发邮件。配好脚本并不表示监控已经运行。真实每小时执行环境必须同时能访问服务器和邮件发送通道。

摘要示例：

```text
主题：[实验监控][前置检查未通过] RoboTwin 跨架构 patch
时间：2026-10-26 10:00 America/New_York
阶段：clean 预检完成；patch 优化尚未开始。
变化：π0.5 已完成10回合，任务成功5/10；ImageWAM 未测。
原因：未达到脚本预检门槛6/10，流水线已退出。
下一步：需要核对基线；本次监控未重启实验、未修改门槛。
报告：/absolute/path/to/report.md
```

邮件发送与去重状态单独记 `delivery.json`：调度ID、本次计划时段、正文快照路径、发送服务、服务返回的message ID或错误。不要将本地生成 `.txt` / `.eml` 或邮箱草稿记为已发送。
