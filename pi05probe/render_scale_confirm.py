"""Locked texture confirmation on init45-49; no retraining or selection."""
import argparse
import json
import time
from render_scale_evaluate_queue import BASE, MODEL, RENDER, command, complete


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--port',type=int,default=8144)
    ap.add_argument('--gpu',type=int,default=4);args=ap.parse_args();name=args.name
    pilot=BASE/('pilot_'+name)/'episodes.json'
    while True:
        try:
            d=json.loads(pilot.read_text())
            if all(d['summary'][c]['n']==5 for c in ['random','optimized']):break
        except (FileNotFoundError,json.JSONDecodeError):pass
        time.sleep(3)
    run=BASE/('train_'+name);out=BASE/('confirm_manifest_'+name)
    assert complete(run/'config.json')
    command([MODEL,'render_scale_manifest.py','--run',str(run),'--out',str(out),'--stage','confirmation'],
        'confirm_manifest_'+name+'.log',{'OMP_NUM_THREADS':'4'})
    command([RENDER,'render_scale_rollout.py','--manifest',str(out/'manifest.json'),'--conditions','random,optimized',
        '--clean-reference',str(BASE/'clean_baseline/episodes.json'),'--indices','45,46,47,48,49',
        '--port',str(args.port),'--out','render_proxy_scale/confirm_'+name], 'confirm_'+name+'.log',
        {'MUJOCO_EGL_DEVICE_ID':str(args.gpu),'OMP_NUM_THREADS':'2'})
    print('[confirmation done]',name,flush=True)


if __name__=='__main__':main()
