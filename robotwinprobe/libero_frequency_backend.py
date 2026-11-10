"""Frozen pi/FastWAM frequency-replacement diagnostic in 224 RGB input space."""
import argparse,dataclasses,json,time,traceback
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from crossarch_common import AUTH,dump,sha
from libero_frequency_mix import variants,common_affine,swap_metrics

MODE_MAP={'matched_clean':'clean','matched_attack':'full','low_attack_high_clean':'low_attack_high_clean','high_attack_low_clean':'high_attack_low_clean'}

class Diagnostic:
    def __init__(self,root):
        self.root=root;self.p=json.loads((root/'protocol.json').read_text());self.kind=self.p['model_kind']
        torch.set_num_threads(4);torch.cuda.init()
        if self.kind=='pi':
            from libero_pi_adapter import Backend
            import os
            os.environ['OPENPI_DATA_HOME']='/shared/user65/.cache/openpi'
        else:from fastwam_adapter import Backend
        self.b=Backend(root);self.native_checked=False

    def build(self,req):
        pair={}
        for condition,key in [('clean','images_clean'),('full','images_attack')]:
            arrays=[np.array(Image.fromarray(np.asarray(req[key][c],np.uint8)).resize((224,224),Image.Resampling.BILINEAR)) for c in ['base','wrist']]
            views=[torch.from_numpy(x.copy()).to('cuda').permute(2,0,1)[None].float()/255 for x in arrays]
            pair[condition]=dict(rgb=[x.astype(np.float64)/255 for x in arrays],prepared=self.b.prepare(req['state'],req['prompt'],views))
        return pair

    def mixed(self,pair,cutoff):
        controls={};audits={}
        for camera in range(2):
            raw=variants(pair['full']['rgb'][camera],pair['clean']['rgb'][camera],cutoff)
            mapped,affine=common_affine(raw)
            for name,(image,meta) in mapped.items():controls.setdefault(name,[]).append(image)
            audits[str(camera)]={'common_affine':affine}
        return controls,audits

    def inject(self,prepared,images):
        views=[torch.tensor(x.copy(),device='cuda',dtype=torch.float32).permute(2,0,1)[None]*2-1 for x in images]
        if self.kind=='pi':
            fields=dict(prepared.images)
            for key,x in zip(['base_0_rgb','left_wrist_0_rgb'],views):fields[key]=x
            out=dataclasses.replace(prepared,images=fields);actual=[fields[k] for k in ['base_0_rgb','left_wrist_0_rgb']]
        else:
            image=torch.cat(views,-1).to(torch.bfloat16);out=(image,*prepared[1:]);actual=[image[:,:,:,:224],image[:,:,:,224:]]
        recovered=[(x[0].float().cpu().permute(1,2,0).numpy().astype(np.float64)+1)/2 for x in actual]
        return out,recovered

    def physical(self,x,prepared):
        if self.kind=='pi':return np.asarray(self.b.policy._output_transform(dict(actions=x[0].float().cpu().numpy(),state=prepared.state[0].float().cpu().numpy()))['actions'])
        return self.b.denorm(self.b.proc,x[0]).copy()

    @torch.no_grad()
    def evaluate(self,pair,mode,seed,cutoff,controls=None,range_audit=None):
        if mode in ('native_clean','native_attack'):
            prepared=pair['clean' if mode=='native_clean' else 'full']['prepared'];audit={}
        else:
            if controls is None:controls,range_audit=self.mixed(pair,cutoff)
            m=MODE_MAP[mode];prepared,recovered=self.inject(pair['clean']['prepared'],controls[m]);audit={}
            for i,rgb in enumerate(recovered):
                audit[str(i)]=dict(**range_audit[str(i)],model_input_swap=swap_metrics(rgb,controls['full'][i],controls['clean'][i],cutoff,m))
        v,x=self.b.response(prepared,seed,full=True);actions=self.physical(x,prepared)
        if not self.native_checked and mode.startswith('native_'):
            if self.kind=='pi':native=self.b.native_full(prepared,seed)
            else:
                image,proprio,(context,mask)=prepared
                native=self.b.model.infer_action(prompt=None,input_image=image,proprio=proprio,context=context,context_mask=mask,action_horizon=32,num_inference_steps=10,seed=seed,rand_device='cpu')['action'][None].float().cpu()
            error=float((x.float().cpu()-native).abs().max());assert error==0,error
            dump(self.root/'audit/native_flow.json',dict(passed=True,max_abs=error));self.native_checked=True
        assert all(p.grad is None for p in self.b.model.parameters())
        return dict(actions=actions,v=v.float().cpu().numpy(),x=x.float().cpu().numpy(),image_audit=audit)

    @torch.no_grad()
    def infer_pair(self,req):
        pair=self.build(req);r=self.evaluate(pair,req['mode'],int(req['seed']),float(req['cutoff']))
        return dict(actions=r['actions'],image_audit=r['image_audit'])

    @torch.no_grad()
    def probe(self,req):
        index=int(req['index']);source=Path(self.p['frequency_restore']['source_root'])/'models'/self.kind/'episodes'
        arrays={};outcomes={}
        for name,cond in [('clean','clean'),('full',self.kind)]:
            path=source/cond/f'init{index}.npz'
            with np.load(path) as z:arrays[name]={k:z[k].copy() for k in ['states','actions','base','wrist','initial_state']}
            outcomes[name]=json.loads(path.with_suffix('.json').read_text())['success']
        assert np.array_equal(arrays['clean']['initial_state'],arrays['full']['initial_state'])
        assert np.array_equal(arrays['clean']['states'][0],arrays['full']['states'][0])
        req_pair=dict(state=arrays['clean']['states'][0],prompt=self.p['prompt_tasks'][0],
            images_clean={c:arrays['clean'][c][0] for c in ['base','wrist']},images_attack={c:arrays['full'][c][0] for c in ['base','wrist']})
        pair=self.build(req_pair);seed=20261006+index*1000;prefix=self.p['evaluation']['replan'];records=[];saved={}
        for mode,source_name in [('native_clean','clean'),('native_attack','full')]:
            result=self.evaluate(pair,mode,seed,.125);a=result['actions'].copy()
            if self.kind=='fast':a[:,-1]=np.sign(-(2*a[:,-1]-1))
            error=float(np.max(np.abs(a-arrays[source_name]['actions'][0])));assert error<1e-5,(mode,error)
            saved[mode]=result;records.append(dict(mode=mode,native_saved_action_max_error=error))
        for cutoff in self.p['frequency_restore']['probe_cutoffs']:
            controls,range_audit=self.mixed(pair,cutoff);group={}
            for mode in MODE_MAP:group[mode]=self.evaluate(pair,mode,seed,cutoff,controls,range_audit)
            ref=group['matched_clean']['actions'][:prefix].copy();full=group['matched_attack']['actions'][:prefix].copy()
            if self.kind=='fast':ref[:,6]=-(2*ref[:,6]-1);full[:,6]=-(2*full[:,6]-1)
            full_delta=full-ref
            for mode,result in group.items():
                a=result['actions'][:prefix].copy()
                if self.kind=='fast':a[:,6]=-(2*a[:,6]-1)
                delta=a-ref;norm=float(np.linalg.norm(delta));den=float(np.linalg.norm(full_delta))
                row=dict(mode=mode,cutoff=cutoff,execution_mse=float(np.mean(delta**2)),arm6_mse=float(np.mean(delta[:,:6]**2)),
                    gripper_continuous_mse=float(np.mean(delta[:,6]**2)),
                    relative_action_norm=norm/den if den>1e-12 else None,
                    action_direction_cosine=float(np.sum(delta*full_delta)/(norm*den)) if norm*den>1e-12 else None,
                    image_audit=result['image_audit'])
                if self.kind=='fast':row['gripper_binary_change_fraction']=float(np.mean(np.sign(a[:,6])!=np.sign(ref[:,6])))
                records.append(row);saved[f'{cutoff}_{mode}']=result
        out=self.root/'probe';out.mkdir(exist_ok=True)
        np.savez_compressed(out/f'init{index}.npz',**{name+'_'+k:v for name,r in saved.items() for k,v in r.items() if k in ['v','x','actions']})
        dump(out/f'init{index}.json',dict(index=index,seed=seed,outcomes=outcomes,rows=records,scope='same initial state and native noise; input-domain diagnostic, not a new physical attack'))
        return dict(index=index,complete=True)

def main(root):
    d=Diagnostic(root);p=d.p
    with Listener(('127.0.0.1',p['control_port']),authkey=AUTH) as listener:
        dump(root/'ready.json',dict(ready=True,pid=__import__('os').getpid()))
        while True:
            with listener.accept() as c:
                try:
                    req=c.recv();cmd=req.pop('command')
                    if cmd=='shutdown':c.send(dict(stopped=True));break
                    if cmd=='ping':result=dict(ready=True)
                    else:result=getattr(d,cmd)(req)
                    c.send(result)
                except Exception as e:traceback.print_exc();c.send(dict(error=repr(e),traceback=traceback.format_exc()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root.resolve())
