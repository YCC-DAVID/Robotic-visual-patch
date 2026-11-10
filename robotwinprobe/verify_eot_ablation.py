"""CPU-only final verification of the frozen experiment #54 artifacts."""
import hashlib,json,math
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from PIL import Image

BASE=Path('/shared/user65/workspace/chence/WAMattack')
ROOT=BASE/'robotwinprobe/out/libero13_eot_ablation_20261030'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def outcomes(root,kind,condition):
    return {i:read(root/'models'/kind/'episodes'/condition/f'init{i}.json')['success'] for i in range(40,50)}

def main():
    protocol=read(ROOT/'protocol.json');ref=Path(protocol['reference'])
    old=BASE/'robotwinprobe/out/libero_pi_fastwam_crossarch_13cm_20261028'
    require(sha(ref/'completion_verification.json')==protocol['reference_verification_sha256'],'reference verification hash')
    require(read(ROOT/'reference_summary.json')==read(ref/'completion_verification.json'),'frozen reference summary')
    full=read(Path(protocol['base_plan']))
    require(sha(Path(protocol['base_plan']))==protocol['base_plan_sha256'],'full plan hash')
    batchpath=old/'training_plan.json';batches=read(batchpath)
    require(sha(batchpath)==protocol['training']['source_batch_noise_plan_sha256'],'batch/noise plan hash')
    require(len(full)==len(batches)==500 and all(len(x)==8 for x in full),'500 x 8 plans')
    identity={'angle_deg':0.0,'dx_fraction':0.0,'dy_fraction':0.0,'scale':1.0,'gain':1.0}
    result={'passed':False,'verified_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(ROOT/'protocol.json'),
        'reference_verification_sha256':sha(ref/'completion_verification.json'),'variants':{},'scores':{},'effects':{}}
    for host in ['nnmc60','nnmc63']:
        a=read(ROOT/'audit'/f'remote_terminal_{host}.json')
        require(a['passed'] and not a['experiment_process_matches'] and all(v=='available' for v in a['locks'].values()),host+' terminal')
    result['remote_terminal_checks']={h:sha(ROOT/'audit'/f'remote_terminal_{h}.json') for h in ['nnmc60','nnmc63']}
    for endpoint,suffix in [('none','control'),('full','eot')]:
        result['scores'][endpoint]={}
        for target in ['pi','fast']:
            clean=outcomes(ref,target,'clean');eligible=[i for i in clean if clean[i]]
            result['scores'][endpoint][target]={}
            for source in ['pi','fast']:
                attack=outcomes(ref,target,source+'_'+suffix);fail=[i for i in eligible if not attack[i]]
                score={'failures':len(fail),'denominator':len(eligible),'ASR':len(fail)/len(eligible),'failed_indices':fail}
                require(score==read(ref/'completion_verification.json')['models'][target]['conditions'][source+'_'+suffix],'endpoint '+endpoint+target+source)
                result['scores'][endpoint][target][source]=score
    total=0
    for variant,active in protocol['variants'].items():
        root=ROOT/'variants'/variant;vp=read(root/'protocol.json');sealed=read(root/'sealed_textures.json')
        require(read(root/'status.json')['stage']=='complete',variant+' complete')
        require(vp['active_factors']==active and vp['training']['no_target_gradients'] and vp['training']['no_gradient_fusion'],variant+' isolation')
        require(vp['training']['batch_noise_plan_sha256']==sha(batchpath),variant+' batch hash')
        require(vp['reference_verification_sha256']==protocol['reference_verification_sha256'],variant+' reference hash')
        require(sealed['protocol_sha256']==sha(root/'protocol.json'),variant+' sealed protocol')
        draws=read(root/'eot_plan.json');require(len(draws)==500,variant+' draws length')
        enabled={key for f in active for key in protocol['factors'][f]}
        for it,(row,base) in enumerate(zip(draws,full)):
            require(len(row)==8,variant+' draw batch')
            for d,f in zip(row,base):
                require(d=={k:f[k] if k in enabled else identity[k] for k in identity},variant+' mask '+str(it))
        for name,h in sealed['textures'].items():require(sha(root/'textures'/name)==h,variant+' texture '+name)
        require(set(sealed['textures'])=={'pi_eot.png','fast_eot.png'},variant+' two textures')
        info={'masked_draws_verified':4000,'sources':{},'models':{},'sealed_manifest_sha256':sha(root/'sealed_textures.json')}
        result['scores'][variant]={}
        for kind in ['pi','fast']:
            state=root/'training'/kind/'state.npz';ledgerpath=root/'gradient_ledgers'/(kind+'.jsonl')
            require(sha(state)==sealed['training_state_sha256'][kind],variant+kind+' state hash')
            require(sha(ledgerpath)==sealed['source_gradient_checks'][kind]['ledger_sha256'],variant+kind+' ledger hash')
            with np.load(state,allow_pickle=False) as z:q=z['texture'];history=json.loads(str(z['history']))
            ledger=[json.loads(x) for x in ledgerpath.read_text().splitlines() if x.strip()]
            require(len(history)==len(ledger)==500,variant+kind+' history/ledger 500')
            require(q.shape==(1,3,128,128) and q.dtype==np.float32 and np.isfinite(q).all() and q.min()>=0 and q.max()<=1,variant+kind+' texture array')
            png=root/'textures'/(kind+'_eot.png')
            require(sha(png)==sha(root/'training'/kind/'texture_0500.png'),variant+kind+' fixed final500 PNG')
            require(np.array_equal(np.asarray(Image.open(png)),np.rint(q[0].transpose(1,2,0)*255).astype(np.uint8)),variant+kind+' final quantization')
            grayhash=hashlib.sha256(np.full((1,3,128,128),.5,np.float32).tobytes()).hexdigest()
            for it,(h,l,b,d) in enumerate(zip(history,ledger,batches,draws),1):
                require(h['iteration']==l['iteration']==it and h['source_model']==l['source_model']==l['server_model']==kind,variant+kind+' source iteration')
                require(h['gradient_provenance']==l,variant+kind+' history ledger match')
                require(l['indices']==b['indices'] and l['seeds']==b['seeds'],variant+kind+' batch/noise')
                require(l['draws_sha256']==digest(d),variant+kind+' masked draw SHA')
                require(math.isfinite(h['loss']) and math.isfinite(l['gradient_norm']) and l['gradient_norm']>0,variant+kind+' finite positive gradient')
                require(len(l['texture_sha256'])==64,variant+kind+' input SHA')
            require(ledger[0]['texture_sha256']==grayhash,variant+kind+' initial gray')
            info['sources'][kind]={'steps':500,'source_only_records':500,'history_ledger_match':True,'plan_and_noise_match':True,'gray_start':True,'final500_quantization_match':True,'checkpoint_sha256':sha(state),'ledger_sha256':sha(ledgerpath),'png_sha256':sha(png)}
            model=root/'models'/kind;cfg=read(model/'protocol.json');native=read(model/'native_audit.json')
            require(cfg['training']['isolated_source_model']==kind and cfg['training']['active_eot_factors']==active,variant+kind+' backend isolation')
            require(cfg['training']['eot_masked_plan_sha256']==sha(root/'eot_plan.json'),variant+kind+' backend masked plan')
            require(native['passed'] and all(x['max_abs']==0 for x in native['rows']) and native['nominal_gray_loss']['relative_error']==0,variant+kind+' native gray exact')
            clean=outcomes(root,kind,'clean');require(clean==outcomes(old,kind,'clean'),variant+kind+' clean matches46')
            eligible=[i for i in clean if clean[i]];require(len(eligible)==(9 if kind=='pi' else 10),variant+kind+' denominator')
            scores={}
            expected_eval=read(ref/'models'/kind/'protocol.json')['evaluation']
            for key in ['indices','steps','wait','replan','horizon','flow_steps','env_seed','noise_seed']:
                require(cfg['evaluation'][key]==expected_eval[key],variant+kind+' native eval '+key)
            files=list((model/'episodes').glob('*/init*.json'));require(len(files)==30,variant+kind+' 30 episodes')
            for condition in ['clean','pi_eot','fast_eot']:
                entries={}
                for i in range(40,50):
                    f=model/'episodes'/condition/f'init{i}.json';e=read(f)
                    require('error' not in e and type(e['success']) is bool and e['condition']==condition and e['init_index']==i,variant+kind+' valid episode')
                    require(e['protocol_sha256']==sha(model/'protocol.json'),variant+kind+' episode protocol')
                    cleanmeta=read(model/'episodes/clean'/f'init{i}.json');oldmeta=read(old/'models'/kind/'episodes/clean'/f'init{i}.json')
                    require(e['initial_state_sha256']==cleanmeta['initial_state_sha256']==oldmeta['initial_state_sha256'],variant+kind+' paired initial hash')
                    with np.load(f.with_suffix('.npz'),allow_pickle=False) as z:
                        require(hashlib.sha256(z['initial_state'].tobytes()).hexdigest()==e['initial_state_sha256'],variant+kind+' raw initial hash')
                        require(z['actions'].shape==(e['queries'],cfg['evaluation']['horizon'],7) and np.isfinite(z['actions']).all(),variant+kind+' action shape/finite')
                        require(z['states'].shape==(e['queries'],8) and np.isfinite(z['states']).all(),variant+kind+' state shape/finite')
                    require(1<=e['steps']<=cfg['evaluation']['steps'],variant+kind+' step bounds')
                    require((root/'sealed_textures.json').stat().st_mtime<=f.stat().st_mtime,variant+kind+' sealed before evaluation')
                    entries[i]=e['success'];total+=1
                if condition!='clean':
                    source=condition.removesuffix('_eot');fail=[i for i in eligible if not entries[i]]
                    require(cfg['evaluation']['texture_paths'][condition]==str(root/'textures'/(condition+'.png')),variant+kind+' evaluation frozen routing')
                    scores[source]={'failures':len(fail),'denominator':len(eligible),'ASR':len(fail)/len(eligible),'failed_indices':fail}
            result['scores'][variant][kind]=scores
            info['models'][kind]={'episodes':30,'clean_matches_46_per_init':True,'clean_success_indices':eligible,'native_audit':native,'protocol_and_initial_hashes_verified':True,'valid_arrays_verified':True}
        manifests=list((root/'source_snapshots').glob('*/manifest.json'))
        require(bool(manifests),variant+' snapshots')
        for manifest in manifests:
            for name,h in read(manifest).items():require(sha(manifest.parent/name)==h,variant+' snapshot manifest '+name)
        info['source_snapshot_manifests_verified']=len(manifests)
        result['variants'][variant]=info
    require(total==480,'480 episodes')
    for factor in protocol['factors']:
        result['effects'][factor]={}
        for target in ['pi','fast']:
            result['effects'][factor][target]={}
            for source in ['pi','fast']:
                s=result['scores'];a=s['only_'+factor][target][source];d=s['drop_'+factor][target][source]
                result['effects'][factor][target][source]={'single_gain':a['ASR']-s['none'][target][source]['ASR'],'conditional_contribution':s['full'][target][source]['ASR']-d['ASR']}
    aggregate=read(ROOT/'analysis_latest.json')
    for name,targets in result['scores'].items():
        for target,sources in targets.items():
            for source,score in sources.items():
                for key in ['failures','denominator','ASR']:require(aggregate['scores'][name][target][source][key]==score[key],'aggregate '+name+target+source)
    require(aggregate['effects']==result['effects'],'effect aggregate')
    require(read(ROOT.parent/'imagewam_libero_eot_transfer_20261030/completion_verification.json')['completed_episodes']==70,'53 terminal verification')
    result.update(passed=True,completed_episodes=480,source_textures=16,training_updates=8000,all_source_isolation_and_masks_verified=True)
    (ROOT/'completion_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'passed':True,'completed_episodes':480,'source_textures':16,'training_updates':8000,'scores':result['scores']},ensure_ascii=False))
if __name__=='__main__':main()
