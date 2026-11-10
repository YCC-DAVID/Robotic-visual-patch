"""Read-only experiment evidence collector. No scheduling, email or process control."""
import argparse
import datetime as dt
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo


def read(path):
    return json.loads(Path(path).read_text())


def field(data,key):
    for part in key.split('.'):
        data=data[part]
    return data


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def collect(c,now=None):
    now=time.time() if now is None else now
    status_path=Path(c['status_file']);issues=[];times=[]
    try:
        s=read(status_path);times.append(status_path.stat().st_mtime)
        stage=str(field(s,c.get('stage_key','stage')))
    except (OSError,ValueError,KeyError,TypeError) as e:
        s={};stage='unknown';issues.append('状态文件无法解析：'+type(e).__name__)
    process='unverified';pid=s.get(c.get('pid_key','pid'))
    if isinstance(pid,int) and pid>0 and c.get('process_contains'):
        try:
            data=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            process='alive' if c['process_contains'] in data else 'pid_mismatch'
        except FileNotFoundError:process='absent'
        except PermissionError:process='unknown'
    progress=[]
    for spec in c.get('progress',[]):
        try:
            value=field(read(spec['file']),spec['key']);times.append(Path(spec['file']).stat().st_mtime)
        except (OSError,ValueError,KeyError,TypeError):value=None
        progress.append(dict(label=spec['label'],value=value,total=spec.get('total')))
    for path in c.get('activity_files',[]):
        if Path(path).is_file():times.append(Path(path).stat().st_mtime)
    age=max(0,now-max(times)) if times else None
    completed=c.get('completed_values',['complete','completed','finished'])
    failed=c.get('failed_values',['failed','stopped_on_error'])
    paused=c.get('paused_values',['paused'])
    artifacts=c.get('completion_files',[])
    missing=[p for p in artifacts if not Path(p).is_file()]
    if stage in failed:state='失败'
    elif stage in paused:state='暂停／等待处理'
    elif stage in completed:
        state='完成标记待核实' if not artifacts or missing else '完成标记和产物存在（需核对结果）'
    elif stage=='unknown':state='未知'
    elif stage.startswith(('waiting','pending','blocked')):state='等待'
    elif process in ('absent','pid_mismatch'):state='意外退出／无匹配进程'
    elif age is not None and age>c.get('stale_after_seconds',7200):state='疑似停滞'
    elif process=='alive':state='运行中'
    else:state='运行状态待核实'
    return dict(task_id=c['task_id'],name=c['name'],checked_at=dt.datetime.fromtimestamp(now,ZoneInfo(c.get('timezone','America/New_York'))).isoformat(),
        state=state,stage=stage,process=process,pid=pid,activity_age_seconds=age,progress=progress,
        missing_completion_files=missing,issues=issues,report=c.get('report'),mail_sent=False)


def main(path):
    c=read(path);out=Path(c['monitor_dir']);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (out/'check.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(out/'latest.json') if (out/'latest.json').exists() else {}
        r=collect(c)
        r['changed_since_previous']=any(old.get(k)!=r[k] for k in ('state','stage','progress'))
        atomic(out/'checks'/f'{time.time_ns()}.json',r);atomic(out/'latest.json',r)
        counts='；'.join(f'{v["label"]}：{v["value"] if v["value"] is not None else "未测"}'+(f'/{v["total"]}' if v['total'] is not None else '') for v in r['progress'])
        lines=[f'[实验监控][{r["state"]}] {r["name"]}',f'时间：{r["checked_at"]}',f'阶段：{r["stage"]}',
            '变化：'+('状态或进度发生变化' if r['changed_since_previous'] else '与上次检查无变化'),
            counts or '指标：未配置可读取的进度字段',f'进程核对：{r["process"]}',f'报告：{r["report"]}',
            '这是本地检查摘要，尚未发送邮件。']
        (out/'latest_summary.txt').write_text('\n'.join(lines)+'\n')
        print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('config',type=Path);main(parser.parse_args().config)
