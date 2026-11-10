"""Train matched spectral bands, then fixed32 development real-render evaluation."""
import argparse,json,os,socket,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image
from e_method_common import load,dump,sha,draw_plan,request,cohort
BASE=Path(__file__).resolve().parent
SOURCE=BASE/'out/e_method_20261027'
DEFAULT=BASE/'out/frequency_bands_20261029'
BANDS={'band00_04':(0,4),'band04_08':(4,8),'band08_12':(8,12),'band12_16':(12,16),'lowpass00_08':(0,8)}
PY='/shared/user65/miniconda3/envs/openpi-server/bin/python'

def project(q,lo,hi,rms=.08):
    f=np.fft.fftfreq(q.shape[-1])*q.shape[-1];r2=f[:,None]**2+f[None,:]**2
    mask=(r2>lo**2)&(r2<=hi**2)
    y=np.fft.ifft2(np.fft.fft2(q-.5,axes=(-2,-1))*mask,axes=(-2,-1)).real
    actual=float(np.sqrt(np.mean(y*y)));peak=float(np.max(np.abs(y)))
    if actual<1e-20:return np.full_like(q,.5)
    scale=min(rms/actual,.499999/max(peak,1e-20))
    return (.5+y*scale).astype(np.float32)

def spectral_audit(q,lo,hi):
    f=np.fft.fftfreq(q.shape[-1])*q.shape[-1];r2=f[:,None]**2+f[None,:]**2
    outside=~((r2>lo**2)&(r2<=hi**2));p=abs(np.fft.fft2(q-.5,axes=(-2,-1)))**2
    return dict(mean_per_channel=q.mean(axis=(-2,-1)).ravel().astype(float).tolist(),
                rms=float(np.sqrt(np.mean((q-.5)**2))),minimum=float(q.min()),maximum=float(q.max()),
                outside_band_power_fraction=float(p[...,outside].sum()/max(p.sum(),1e-30)))

def write_texture(root,name,q):
    d=root/'patches'/name;d.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)).save(d/'texture_0500.png')

def setup(root,size_m=.10):
    root.mkdir(parents=True,exist_ok=True);p=load(SOURCE)
    p.update(experiment='matched_frequency_bands',backend_port=18774)
    p['frequency_study']=dict(bands=BANDS,mean_rgb=[.5,.5,.5],target_rms=.08,
        contrast='same RMS target; reduce by single global scale if [0,1] range would be violated; report achieved RMS',
        data='same policy approach-state bank and fixed batch/noise draws as original E velocity14',
        objective='velocity14',steps=500,batch=8,lr=.01,initialization='gray',training_seed=202610270,
        evaluation='32 fixed development states, persistent real render; matched random per band; no independent test used',
        selection='paired ASR excess above its own same-band random; then ASR; predeclared band order for ties',
        inference_limit='exploratory single training seed; no claim of universally strongest band',
        coordinate='radial FFT cycles per 10cm patch; DC excluded, low < radius <= high')
    if size_m != .10:
        p['patch']['size_m']=size_m
        p['backend_port']={.13:18775,.20:18776}[size_m]
        p['frequency_study'].update(coordinate=f'radial FFT cycles per {size_m*100:g}cm patch; DC excluded, low < radius <= high',
            size_comparison='same cycles per patch, mean/RMS, training noise and cohorts; fresh size-specific real-render calibration',
            calibration='regenerate own clean approach rollouts at original seeds; require exact source state/RGB matches before training')
    if (root/'protocol.json').exists():assert load(root)==json.loads(json.dumps(p))
    else:dump(root/'protocol.json',p)
    # Source paths are mapped to the shared mount for cross-node reproducibility.
    src=json.loads((SOURCE/'collection_manifest.json').read_text())
    for rec in src['records']:rec['path']=str(BASE/('out/'+rec['path'].split('/robotwinprobe/out/',1)[1]))
    if size_m == .10 and not (root/'collection_manifest.json').exists():dump(root/'collection_manifest.json',src)
    rng=np.random.default_rng(202610290);base=rng.normal(size=(1,3,128,128)).astype(np.float32)+.5
    audits={}
    for name,(lo,hi) in BANDS.items():
        q=project(base,lo,hi);write_texture(root,'random_'+name,q);audits[name]=spectral_audit(q,lo,hi)
    dump(root/'random_spectral_audit.json',audits)
    if not (root/'collection_manifest.json').exists():return None
    if size_m != .10:
        verify_size_bank(root,size_m)
    plan=draw_plan(root,202610270)
    if (root/'draws.json').exists():assert json.loads((root/'draws.json').read_text())==plan
    else:dump(root/'draws.json',plan)
    return plan

def verify_size_bank(root,size_m):
    original=json.loads((SOURCE/'collection_manifest.json').read_text())['records']
    current=json.loads((root/'collection_manifest.json').read_text())['records']
    key=lambda row:(row['seed'],row['query'],row['draw'])
    original={key(r):r for r in original};current={key(r):r for r in current}
    assert original.keys()==current.keys(),'Training state cohort changed with patch size'
    for ident,row in current.items():
        old_path=BASE/('out/'+original[ident]['path'].split('/robotwinprobe/out/',1)[1])
        assert sha(row['path'])==row['sha256']
        with np.load(row['path']) as new,np.load(old_path) as old:
            assert float(new['size_m'])==size_m
            for field in ['state14','center_m','yaw_deg','light','prompt']+[c+'_clean' for c in ('head_camera','left_camera','right_camera')]:
                assert np.array_equal(new[field],old[field]),(ident,field)
    dump(root/'size_calibration_audit.json',dict(passed=True,size_m=size_m,records=len(current),
        checks='identical state/clean RGB/prompt/pose/light and cohort; fresh size-specific LUT, masks and geometry'))

def train(root,name,band,draws):
    lo,hi=band;dest=root/'training'/name;dest.mkdir(parents=True,exist_ok=True)
    ckpt=dest/'state.npz';history=[];q=np.full((1,3,128,128),.5,np.float32);start=0
    if ckpt.exists():
        with np.load(ckpt) as z:
            assert str(z['protocol_sha256'])==sha(root/'protocol.json')
            assert str(z['draws_sha256'])==sha(root/'draws.json')
            q=z['texture'].copy();history=json.loads(str(z['history']));start=int(z['iteration'])
    for it in range(start,500):
        grad=np.zeros_like(q);losses=[]
        for draw in draws[it]:
            r=request(dict(command='method_gradient',kind='velocity14',texture=q,**draw),root)
            grad+=r['gradient']/8;losses.append(r['loss'])
        assert np.isfinite(grad).all() and np.linalg.norm(grad)>0
        q=project(q+.01*np.sign(grad),lo,hi)
        row=dict(iteration=it+1,total=500,loss=float(np.mean(losses)),**spectral_audit(q,lo,hi));history.append(row)
        assert row['outside_band_power_fraction']<1e-10
        temp=ckpt.with_suffix('.tmp.npz')
        np.savez_compressed(temp,texture=q,iteration=it+1,history=json.dumps(history),protocol_sha256=sha(root/'protocol.json'),draws_sha256=sha(root/'draws.json'));temp.replace(ckpt)
        dump(dest/'progress.json',row)
        if it==0 or (it+1)%25==0:print('[train]',name,row,flush=True)
    write_texture(root,name,q)
    exported=np.asarray(Image.open(root/'patches'/name/'texture_0500.png')).transpose(2,0,1)[None].astype(np.float32)/255
    dump(dest/'complete.json',dict(complete=True,protocol_sha256=sha(root/'protocol.json'),texture_sha256=sha(root/'patches'/name/'texture_0500.png'),float_spectrum=spectral_audit(q,lo,hi),png_spectrum=spectral_audit(exported,lo,hi)))

def summarize(root):
    from stage_joint_report import wilson,interval
    folder=root/'cohorts/development/episodes';seeds=load(root)['method_study']['development_seeds'];rows=[]
    def get(seed,c):return json.loads((folder/f'native_full_approach_{c}_s{seed}.json').read_text())
    for name in BANDS:
        pairs=[(get(s,'clean'),get(s,name),get(s,'random_'+name)) for s in seeds]
        assert all(x['status']=='complete' and not x.get('error') for p in pairs for x in p)
        pairs=[p for p in pairs if p[0]['success']]
        attack=[int(not p[1]['success']) for p in pairs];random=[int(not p[2]['success']) for p in pairs]
        diff=[a-b for a,b in zip(attack,random)]
        rows.append(dict(condition=name,n=len(pairs),failures=sum(attack),ASR=float(np.mean(attack)),ASR_CI95=wilson(attack),random_failures=sum(random),random_ASR=float(np.mean(random)),excess_ASR=float(np.mean(diff)),excess_CI95=interval(diff)))
    dump(root/'development_results.json',dict(complete=True,rows=rows,test_used=False))

def main(a):
    root=a.root.resolve();draws=setup(root,a.size_m);logs=root/'logs';logs.mkdir(exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.gpu,JAX_PLATFORMS='cpu',XLA_PYTHON_CLIENT_PREALLOCATE='false',HF_HUB_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',OPENPI_DATA_HOME='/shared/user65/.cache/openpi',ROBOTWIN_PYTHON='/shared/user65/miniconda3/envs/RoboTwin/bin/python')
    env.pop('PYTHONPATH',None)
    status=lambda stage,**kw:dump(root/'run_status.json',dict(stage=stage,host=socket.gethostname(),gpu=a.gpu,pid=os.getpid(),**kw))
    backend=None
    try:
        status('loading_backend');(root/'backend_ready.json').unlink(missing_ok=True)
        with (logs/'backend.log').open('a') as f:backend=subprocess.Popen([PY,'-u',str(BASE/'e_method_backend.py'),'--root',str(root)],cwd=BASE.parent,env=env,stdout=f,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+600
        while not (root/'backend_ready.json').exists():
            if backend.poll() is not None:raise RuntimeError('backend failed')
            if time.monotonic()>deadline:raise TimeoutError('backend loading')
            time.sleep(2)
        if draws is None:
            status('collecting_size_specific_bank',size_m=a.size_m)
            with (logs/'collection.log').open('a') as f:subprocess.run(['bash',str(BASE/'run_sim.sh'),str(BASE/'e_method_sim.py'),'--root',str(root),'--mode','collect'],cwd=BASE.parent,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            draws=setup(root,a.size_m)
            assert draws is not None
        for name,band in BANDS.items():status('training',condition=name);train(root,name,band,draws)
        conditions=['clean']+[c for name in BANDS for c in (name,'random_'+name)]
        for seed in load(root)['method_study']['development_seeds']:
            status('evaluating',seed=seed)
            with (logs/f'seed{seed}.log').open('a') as f:subprocess.run(['bash',str(BASE/'run_sim.sh'),str(BASE/'e_method_sim.py'),'--root',str(root),'--mode','evaluate','--split','development','--seed',str(seed),'--conditions',','.join(conditions)],cwd=BASE.parent,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        summarize(root);status('development_complete')
    except BaseException as e:status('failed',error=repr(e));raise
    finally:
        if backend is not None and backend.poll() is None:backend.terminate();backend.wait(timeout=30)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--root',type=Path,default=DEFAULT);p.add_argument('--size-m',type=float,choices=[.10,.13,.20],default=.10);main(p.parse_args())
