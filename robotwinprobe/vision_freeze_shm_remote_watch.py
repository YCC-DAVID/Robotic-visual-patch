"""Mirror this experiment's remote validation status without controlling training."""
import argparse,json,subprocess,time
from pathlib import Path
import checkpoint_study_common as S

def main(root):
    out=root/'shm_validation_20261027';remote=str(out/'validation_status.json')
    while True:
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','nnmc63','cat',remote],capture_output=True,text=True,timeout=20)
        if result.returncode:
            S.dump(out/'mirror_status.json',dict(stage='unreachable',error=result.stderr,updated_unix=time.time()))
            time.sleep(30);continue
        status=json.loads(result.stdout);status['execution_host']='nnmc63';status['mirrored_unix']=time.time()
        pid=status.get('pid')
        check=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','nnmc63',
            'cat',f'/proc/{pid}/cmdline'],capture_output=True,timeout=20)
        alive=b'vision_freeze_shm_validation.py' in check.stdout
        status['remote_process_alive']=alive
        if not alive and not status.get('complete') and status.get('stage')!='failed':
            status.update(stage='remote_process_exited',error='Remote validation process no longer exists')
        S.dump(out/'validation_status.json',status)
        S.dump(out/'mirror_status.json',dict(stage='synchronized',remote_pid=pid,alive=alive,updated_unix=time.time()))
        if status.get('complete') or status.get('stage') in ('failed','remote_process_exited'):
            subprocess.run(['rsync','-az','--exclude=checkpoints','--exclude=jax_cache','--exclude=validation_status.json',
                '--exclude=mirror_status.json','--exclude=process.json','--exclude=protocol.json',
                'nnmc63:'+str(out)+'/',str(out)+'/'],check=True)
            return
        time.sleep(30)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
