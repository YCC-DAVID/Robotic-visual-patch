"""Isolated native pi0.5 service; proxy fidelity is recorded, never a training gate."""
import argparse
import traceback
from multiprocessing.connection import Listener
from pathlib import Path

from stage_joint_common import DEFAULT,PORT,AUTH,dump


def main(root):
    from crossarch_backend import Backend
    backend=Backend('pi',root)
    backend.audit()
    with Listener(('127.0.0.1',PORT),authkey=AUTH) as listener:
        dump(root/'backend_ready.json',dict(ready=True,model='pi',port=PORT))
        while True:
            with listener.accept() as conn:
                try:
                    req=conn.recv();cmd=req.pop('command')
                    result={'ready':True,'model':'pi'} if cmd=='ping' else getattr(backend,cmd)(req)
                    conn.send(result)
                except Exception as e:
                    conn.send(dict(error=repr(e),traceback=traceback.format_exc()))
                    traceback.print_exc()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT)
    main(ap.parse_args().root.resolve())
