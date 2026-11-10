"""Explicit deterministic noise server for a study checkpoint."""
import argparse
from pathlib import Path
import numpy as np
import checkpoint_study_common as S
import checkpoint_study_model as M


def main(args):
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    policy=M.load(args.checkpoint)
    class Wrapped:
        metadata=dict(model=args.checkpoint.name,action_horizon=32,action_dim=14,
                      views=list(S.WIRE),checkpoint_sha256=S.sha(args.checkpoint/'model.safetensors'))
        def infer(self,element):
            element=dict(element);noise=np.asarray(element.pop('noise'),np.float32)
            assert noise.shape==(32,32) and np.isfinite(noise).all()
            assert np.asarray(element['state']).shape==(14,)
            assert set(element['images'])==set(S.WIRE)
            element['images']={k:np.asarray(v).copy().transpose(2,0,1) for k,v in element['images'].items()}
            return policy.infer(element,noise=noise)
    wrapped=Wrapped()
    request=dict(state=np.zeros(14,np.float32),prompt='Place the bowl on the plate.',
        images={k:np.zeros((240,320,3),np.uint8) for k in S.WIRE},
        noise=np.random.default_rng(0).standard_normal((32,32)).astype(np.float32))
    a=wrapped.infer(request)['actions'];b=wrapped.infer(request)['actions']
    assert a.shape==(32,14) and np.isfinite(a).all() and np.array_equal(a,b)
    S.dump(args.ready,dict(**wrapped.metadata,ready=True,port=args.port,deterministic=True))
    WebsocketPolicyServer(wrapped,host='127.0.0.1',port=args.port,metadata=wrapped.metadata).serve_forever()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--port',type=int,required=True);ap.add_argument('--ready',type=Path,required=True)
    main(ap.parse_args())
