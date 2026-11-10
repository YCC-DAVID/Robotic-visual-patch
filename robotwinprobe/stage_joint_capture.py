"""Collect exact site07 calibration, reusing complete successful expert replays."""
import argparse
from pathlib import Path

from stage_joint_common import DEFAULT,load,prepare,dump


def run(root):
    p=prepare(root)
    import bootstrap as B
    B.simulator_paths()
    from crossarch_capture import capture
    for split in ('validation','train'):
        seeds=p['data'][split+'_seeds']
        for seed in seeds:
            dump(root/'capture_status.json',dict(stage='capture',split=split,seed=seed))
            capture(root,seed)
    dump(root/'capture_status.json',dict(stage='complete'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    run(ap.parse_args().root.resolve())
