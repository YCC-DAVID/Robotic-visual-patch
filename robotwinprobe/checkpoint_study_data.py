"""Fixed expert data cache, train-only normalization, exact upstream transforms."""
import argparse
import json
import os
from pathlib import Path
import time

if __name__ == '__main__':
    os.environ.setdefault('JAX_PLATFORMS','cpu')
import numpy as np
from PIL import Image
import checkpoint_study_common as S
S.training_paths()
from openpi import transforms as T
from openpi.models import model as M
from openpi.policies import aloha_policy as A
from openpi.shared import normalize


def resize224(rgb):
    image=Image.fromarray(rgb).resize((224,168),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(224,224))
    canvas.paste(image,(0,28))
    return np.asarray(canvas)


def encode_states(states):
    # Upstream _decode_state indexes a single state with state[[6, 13]].
    # Calling it on a batch would transform rows rather than gripper columns.
    return np.stack([A._decode_state(row.copy(), adapt_to_pi=True) for row in states])


def prepare(root, smoke=False):
    root=Path(root); S.prepare_protocol(root)
    cache=root/('data_smoke' if smoke else 'data_cache'); cache.mkdir(parents=True,exist_ok=True)
    if (cache/'manifest.json').exists():
        assert json.loads((cache/'manifest.json').read_text())['complete']
        return
    splits=['smoke'] if smoke else ['train','validation']
    records=[]; start=0
    for split in splits:
        manifest=json.loads((root/'expert'/f'{split}_manifest.json').read_text())
        assert manifest['complete']
        for row in manifest['selected']:
            source=root/'expert'/f'seed{row["seed"]}'/'demonstration.npz'
            assert S.sha(source)==row['sha256']
            records.append(dict(seed=row['seed'],split=split,start=start,frames=row['frames'],
                prompt=row['prompt'],path=str(source),sha256=row['sha256']))
            start+=row['frames']
    assert len({r['seed'] for r in records})==len(records)
    images={name:np.lib.format.open_memmap(cache/f'{name}.npy',mode='w+',dtype=np.uint8,
                shape=(start,224,224,3)) for name in S.CAMERAS}
    state=np.zeros((start,14),np.float32); ends=np.zeros(start,np.int32)
    train_mask=np.zeros(start,bool); episodes=np.zeros(start,np.int32); prompts=[]
    for r in records:
        i=r['start'];n=r['frames']; sl=slice(i,i+n)
        with np.load(r['path']) as z:
            state[sl]=z['state14']; ends[sl]=i+n-1; episodes[sl]=r['seed']
            train_mask[sl]=r['split'] in ('train','smoke')
            prompts.extend([r['prompt']]*n)
            for name in S.CAMERAS:
                rgb=z[name]
                for j in range(n):images[name][i+j]=resize224(rgb[j])
                images[name].flush()
        print('[cache]',r['seed'],n,flush=True)
    np.savez(cache/'index.npz',state14=state,end=ends,episode=episodes,train_mask=train_mask,prompt=prompts)
    # Encode exactly as AlohaInputs, then form per-query delta action chunks.
    encoded_state=encode_states(state)
    encoded_action=A._encode_actions_inv(state.copy(),adapt_to_pi=True)
    ids=np.flatnonzero(train_mask)
    future=np.minimum(ids[:,None]+np.arange(32)[None],ends[ids,None])
    delta=encoded_action[future].copy()
    mask=np.array(T.make_bool_mask(6,-1,6,-1))
    delta[...,mask]-=encoded_state[ids,None,:][...,mask]
    stats={}
    for key,values in [('state',encoded_state[ids]),('actions',delta.reshape(-1,14))]:
        values=values.astype(np.float64)
        stats[key]=normalize.NormStats(mean=values.mean(0),std=values.std(0),
            q01=np.quantile(values,.01,axis=0),q99=np.quantile(values,.99,axis=0))
    assets=root/('smoke_assets' if smoke else 'assets')/S.ASSET
    assets.mkdir(parents=True,exist_ok=True)
    (assets/'norm_stats.json').write_text(normalize.serialize_json(stats)+'\n')
    # A smoke cache lives separately and cannot replace the full-study stats.
    S.dump(cache/'manifest.json',dict(complete=True,frames=start,train_frames=int(train_mask.sum()),
        validation_frames=int((~train_mask).sum()),records=records,smoke=smoke,
        norm_stats=str(assets/'norm_stats.json'),norm_sha256=S.sha(assets/'norm_stats.json'),
        normalization_fit_split='train only',action_horizon=32,
        resize='native 320x240 -> PIL BILINEAR 224x168 -> black letterbox 224x224'))
    print('[data ready]',cache,flush=True)


class Data:
    def __init__(self,root,cfg,smoke=False):
        self.root=Path(root); self.cache=self.root/('data_smoke' if smoke else 'data_cache')
        self.manifest=json.loads((self.cache/'manifest.json').read_text())
        assert self.manifest['complete']
        self.index=dict(np.load(self.cache/'index.npz'))
        self.raw_states=self.index['state14']
        self.images={key:np.load(self.cache/f'{name}.npy',mmap_mode='r') for key,name in zip(S.MODEL,S.CAMERAS)}
        self.config=cfg.data.create(cfg.assets_dirs,cfg.model)
        self.norm=T.Normalize(self.config.norm_stats,use_quantiles=True)
        self.transform=T.compose([*self.config.data_transforms.inputs,self.norm,*self.config.model_transforms.inputs])
        self.encoded_state=encode_states(self.raw_states)
        self.encoded_action=A._encode_actions_inv(self.raw_states.copy(),adapt_to_pi=True)
        self.state_norm=self.norm({'state':self.encoded_state})['state']
        self.states=np.pad(self.state_norm,((0,0),(0,18))).astype(np.float32)
        self.delta_mask=np.array(T.make_bool_mask(6,-1,6,-1))
        self.train_ids=np.flatnonzero(self.index['train_mask'])
        self.validation_ids=np.flatnonzero(~self.index['train_mask'])
        if smoke:self.validation_ids=self.train_ids[:16]
        # Execute the actual model transform chain with tiny placeholder images;
        # resize/tokenization/pad ordering remains owned by upstream OpenPI.
        tokenize=next(t for t in self.config.model_transforms.inputs if isinstance(t,T.TokenizePrompt))
        tokens=[];masks=[]
        for state,prompt in zip(self.state_norm,self.index['prompt']):
            encoded=tokenize(dict(state=state.copy(),prompt=str(prompt)))
            tokens.append(encoded['tokenized_prompt']);masks.append(encoded['tokenized_prompt_mask'])
        self.tokens=np.asarray(tokens);self.token_masks=np.asarray(masks)
        self.verify()

    def full_sample(self,idx):
        future=np.minimum(np.arange(idx,idx+32),self.index['end'][idx])
        raw=dict(state=self.raw_states[idx].copy(),actions=self.raw_states[future].copy(),
            prompt=str(self.index['prompt'][idx]),
            images={w:self.images[k][idx].copy().transpose(2,0,1) for w,k in zip(S.WIRE,S.MODEL)})
        return self.transform(raw)

    def batch(self,indices):
        ids=np.asarray(indices)
        future=np.minimum(ids[:,None]+np.arange(32)[None],self.index['end'][ids,None])
        actions=self.encoded_action[future].copy()
        actions[...,self.delta_mask]-=self.encoded_state[ids,None,:][...,self.delta_mask]
        actions=self.norm({'actions':actions})['actions']
        actions=np.pad(actions,((0,0),(0,0),(0,18))).astype(np.float32)
        obs=M.Observation(images={k:v[ids].astype(np.float32)/255*2-1 for k,v in self.images.items()},
            image_masks={k:np.ones(len(ids),bool) for k in self.images},state=self.states[ids],
            tokenized_prompt=self.tokens[ids],tokenized_prompt_mask=self.token_masks[ids])
        return obs,actions

    def verify(self):
        indices=np.unique(np.r_[0,len(self.states)-1,self.train_ids[len(self.train_ids)//2],self.validation_ids[:1]])
        for idx in indices:
            exact=self.full_sample(int(idx));obs,action=self.batch([idx])
            assert np.array_equal(obs.state[0],np.asarray(exact['state'],np.float32))
            assert np.array_equal(action[0],np.asarray(exact['actions'],np.float32))
            assert np.array_equal(obs.tokenized_prompt[0],exact['tokenized_prompt'])
            assert np.array_equal(obs.tokenized_prompt_mask[0],exact['tokenized_prompt_mask'])
            for key in S.MODEL:
                assert np.array_equal(obs.images[key][0],np.asarray(exact['image'][key],np.float32)/255*2-1)
        print('[data] exact state/action/image/prompt transforms verified',flush=True)

    def data_config(self):
        return self.config


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=S.DEFAULT)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    prepare(args.root.resolve(),args.smoke)
