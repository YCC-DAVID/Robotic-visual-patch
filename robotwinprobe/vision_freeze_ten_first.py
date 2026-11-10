"""Evaluate ten shared initial states per condition before resuming all 100."""
import argparse
import fcntl
import json
import os
import shutil
from pathlib import Path

import checkpoint_study_common as S
from vision_freeze_resume_evaluation import ResumePipeline


def ten_first(pipe):
    root = pipe.root
    quick = root/'preexperiment10'
    quick.mkdir(exist_ok=True)
    protocol = json.loads((root/'protocol.json').read_text())
    seeds = protocol['evaluation']['seeds'][:10]
    protocol['evaluation']['seeds'] = seeds
    S.dump(quick/'protocol.json', protocol)
    for name in ['models', 'texture_manifest.json']:
        path = quick/name
        if not path.exists():
            path.symlink_to(root/name, target_is_directory=(root/name).is_dir())
    conditions = ['clean'] + [t['id'] for t in json.loads((root/'texture_manifest.json').read_text())]
    summary = dict(complete=False, seeds=seeds, models={})
    for arm in ['F', 'U']:
        model = arm+'_s10000'
        pending = []
        for condition in conditions:
            dest = quick/'closedloop'/model/condition
            dest.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                for suffix in ['json', 'npz']:
                    source = root/'closedloop'/model/condition/f'seed{seed}.{suffix}'
                    if source.exists() and not (dest/source.name).exists():
                        shutil.copy2(source, dest/source.name)
            rows = [json.loads((dest/f'seed{s}.json').read_text()) for s in seeds if (dest/f'seed{s}.json').exists()]
            assert all(not r.get('error') for r in rows), 'Inspect existing episode errors'
            if len(rows) < 10:
                pending.append(condition)
        if pending:
            child, log = pipe.server(model, '0', 8479)
            try:
                for condition in pending:
                    pipe.stage('preexperiment10', model=model, condition=condition)
                    argv = ['bash', 'robotwinprobe/run_sim.sh', 'robotwinprobe/vision_freeze_rollout.py',
                            '--root', str(quick), '--model', model, '--condition', condition, '--port', '8479']
                    pipe.run('preexperiment10_'+model+'_'+condition, argv, '1')
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=60)
                log.close()
        clean = {s: json.loads((quick/'closedloop'/model/'clean'/f'seed{s}.json').read_text()) for s in seeds}
        clean_success = sum(r['success'] for r in clean.values())
        summary['models'][model] = {}
        for condition in conditions:
            dest = quick/'closedloop'/model/condition
            rows = [json.loads((dest/f'seed{s}.json').read_text()) for s in seeds]
            assert all(not r.get('error') and r['initial_clean_sha256'] == clean[r['seed']]['initial_clean_sha256'] for r in rows)
            failures = sum(clean[r['seed']]['success'] and not r['success'] for r in rows)
            summary['models'][model][condition] = dict(successes=sum(r['success'] for r in rows),
                total=10, clean_successes=clean_success, ASR=failures/clean_success if clean_success else None)
            S.dump(dest/'results.json', dict(complete=True, target=10, episodes=rows,
                   task_successes=sum(r['success'] for r in rows)))
            # Identical seeds, model, texture and noise rule: retain new raw
            # episodes for the subsequent 100-state run without changing its protocol.
            full = root/'closedloop'/model/condition
            full.mkdir(parents=True, exist_ok=True)
            for seed in seeds:
                for suffix in ['json', 'npz']:
                    src = dest/f'seed{seed}.{suffix}'
                    if src.exists() and not (full/src.name).exists():
                        shutil.copy2(src, full/src.name)
        S.dump(quick/'summary.json', summary)
    summary['complete'] = True
    S.dump(quick/'summary.json', summary)
    pipe.stage('preexperiment10_complete', summary=str(quick/'summary.json'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--gpus', default='6,7')
    args = p.parse_args()
    args.model_steps = 10000
    args.train_only = False
    with (args.root/'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        pipe = ResumePipeline(args)
        try:
            pipe.check_resources(['0', '1'], all_hosts=True)
            ten_first(pipe)
            pipe.main()
        except Exception as exc:
            pipe.stage('ten_first_stopped', error=repr(exc))
            raise
