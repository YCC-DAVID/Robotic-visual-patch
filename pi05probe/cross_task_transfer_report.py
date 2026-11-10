"""Audit frozen texture task transfer; paired ASR and per-episode visibility."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out';DEST=OUT/'cross_task_transfer'
TASKS=['bowl_cabinet','bottle_rack','stove']
CONDS=['clean','random_full','full_both','routed','random_block','block_base']


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main():
    assert read(DEST/'progress.json')['stage']=='complete'
    protocol=read(DEST/'protocol.json');manifest=read(DEST/'manifest.json')
    assert sha(DEST/'manifest.json')==protocol['manifest_sha256']
    assert all(sha(ROOT/name)==v for name,v in protocol['source_sha256'].items())
    assert all((None if path is None else sha(path))==manifest['texture_sha256'][name]
               for name,path in manifest['textures'].items())
    definitions=read(DEST/'task_definitions.json');assert definitions['physical_scene_specs_equal']
    summary={};audit_count=0;initial_clean={};visibility={}
    for task in TASKS:
        folder=DEST/task/'rollout';data=read(folder/'episodes.json');meta=data['metadata']
        assert read(DEST/task/'complete.json')['complete'] and (DEST/task/'service_stopped.json').exists()
        assert meta['task']==next(x['stem'] for x in protocol['tasks'] if x['name']==task)
        assert meta['bddl_sha256']==definitions['tasks'][meta['task']]['bddl_sha256']
        assert meta['geometry']==manifest['geometry'] and meta['manifest_sha256']==sha(DEST/'manifest.json')
        rows=data['per_episode'];clean={x['init_index']:x for x in rows['clean']}
        good={ep for ep,row in clean.items() if row['success']}
        initial_clean[task]={}
        for ep,row in clean.items():
            with np.load(folder/row['replay_record']) as d:initial_clean[task][ep]=d['initial_state'].copy()
        result={};taskvis={}
        for condition in CONDS:
            records=rows[condition];assert len(records)==10 and {r['init_index'] for r in records}==set(range(40,50))
            assert (None if meta['textures'][condition] is None else meta['textures'][condition]['sha256'])==manifest['texture_sha256'][condition]
            vmed=[];vmax=[];vfrac=[];delays=[]
            for row in records:
                ep=row['init_index']
                a=np.load(folder/row['replay_record']);b=np.load(folder/clean[ep]['replay_record'])
                assert np.array_equal(a['initial_state'],b['initial_state'])
                n=min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
                for key in ['fixture_body_pos','fixture_body_quat']:assert np.array_equal(a[key][:n],b[key][:n])
                calls=row['inferences'];assert calls==len(a['actions'])==len(a['sim_states'])==len(a['base224'])==len(a['wrist224'])
                assert a['actions'].shape==(calls,10,7) and np.isfinite(a['actions']).all()
                assert np.array_equal(a['noise_seeds'],20261006+ep*1000+np.arange(calls))
                assert sha_bytes(a['initial_state'])==row['initial_state_sha256']
                vis=a['visible_pixels224'];assert vis.shape==(calls,2)
                if condition=='clean':assert np.all(vis==-1)
                else:
                    assert np.all((vis>=0)&(vis<=224*224))
                    assert np.array_equal(np.median(vis,axis=0),row['visible_pixels224_median'])
                    assert np.array_equal(np.max(vis,axis=0),row['visible_pixels224_max'])
                    vmed.append(np.median(vis,axis=0));vmax.append(vis.max(axis=0));vfrac.append((vis>0).mean(axis=0))
                if ep in good and row['success']:delays.append(row['success_step']-clean[ep]['success_step'])
                audit_count+=1
            failed=[r['init_index'] for r in records if r['init_index'] in good and not r['success']]
            result[condition]={'task_successes':sum(r['success'] for r in records),'episodes':10,
                               'attack_successes':len(failed),'clean_success_denominator':len(good),
                               'ASR':len(failed)/len(good) if good else None,'failed_clean_success_indices':failed,
                               'completion_step_delta_median_joint_success':float(np.median(delays)) if delays else None,
                               'completion_step_delta_n':len(delays)}
            check=data['summary'][condition]
            assert check['successes']==result[condition]['task_successes'] and check['paired_ASR_failures']==len(failed)
            if vmed:
                taskvis[condition]={'median_of_episode_median_pixels_base_wrist':np.median(vmed,axis=0).tolist(),
                                    'median_of_episode_peak_pixels_base_wrist':np.median(vmax,axis=0).tolist(),
                                    'median_of_episode_visible_fraction_base_wrist':np.median(vfrac,axis=0).tolist(),
                                    'per_episode_median_pixels':np.array(vmed).tolist(),
                                    'per_episode_visible_fraction':np.array(vfrac).tolist()}
        summary[task]={'task':meta['task'],'prompt':meta['prompt'],'conditions':result,
                       'clean_failed_indices':sorted(set(range(40,50))-good),'init_states_sha256':meta['init_states_sha256']}
        visibility[task]=taskvis
    assert audit_count==180
    old=read(OUT/'dual_view/summary.json')['closed_loop'];low=read(OUT/'dual_view_lowfreq/summary.json')['closed_loop']
    source={}
    for name,key in [('random_full','random'),('full_both','both'),('routed','routed')]:
        r=old[key];source[name]={'task_successes':r['robot_task_successes'],'attack_successes':r['attack_successes'],'clean_success_denominator':r['clean_success_denominator']}
    for name,key in [('random_block','random_block'),('block_base','block_base')]:
        r=low[key];source[name]={'task_successes':r['task_successes'],'attack_successes':r['attack_successes'],'clean_success_denominator':r['denominator']}
    source_clean=read(OUT/'render_proxy_scale/clean_baseline/episodes.json')['per_episode']['clean']
    assert len(source_clean)==10
    source_successes=sum(r['success'] for r in source_clean)
    source['clean']={'task_successes':source_successes,'attack_successes':0,'clean_success_denominator':source_successes}
    assert all(r['clean_success_denominator']==source_successes for r in source.values())
    equality={task:{str(ep):bool(np.array_equal(initial_clean[task][ep],initial_clean['bowl_cabinet'][ep]))
                    for ep in range(40,50)} for task in TASKS}
    audit={'complete':True,'new_paired_episode_records_checked':audit_count,
           'manifest_texture_source_hashes_exact':True,'physics_fixtures_noise_paired':True,
           'all_task_local_conditions_complete':True,'physical_scene_definitions_equal':True,
           'own_initial_state_comparison_to_bowl_cabinet':equality,'services_stopped':True,
           'new_target_task_training':False,'unique_target_task_initial_pairs':30}
    full={'complete':True,'source_task_reference':source,'targets':summary,'visibility':visibility,'audit':audit,
          'notes':'Frozen source-trained textures. Each target own clean denominator, official states and goal. No task-specific target training. Same physical scene; not cross-scene generalization. Visibility is per-episode summarized to avoid overweighting long failed rollouts.'}
    (DEST/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (DEST/'summary.json').write_text(json.dumps(full,indent=2)+'\n')
    figures(full)
    lines=['# Frozen texture task transfer','','Each entry = attack successes / task-local clean successes; source reference reused.','',
           '| Task | Clean task successes /10 | Random full | Full both | Routed | Random block | Block base |','|---|---:|---:|---:|---:|---:|---:|']
    for task,counts in [('source_bowl_plate',source)]+[(t,summary[t]['conditions']) for t in TASKS]:
        entries=[str(counts['clean']['task_successes'])+'/10']+[f"{counts[c]['attack_successes']}/{counts[c]['clean_success_denominator']}" for c in CONDS[1:]]
        lines.append('| '+task+' | '+' | '.join(entries)+' |')
    lines+=['',full['notes']];(DEST/'results.md').write_text('\n'.join(lines)+'\n')
    snap=DEST/'source_snapshot';snap.mkdir(exist_ok=True)
    for name in list(protocol['source_sha256'])+['cross_task_transfer_report.py']:shutil.copy2(ROOT/name,snap/name)
    files={str(p.relative_to(DEST)):{'sha256':sha(p),'bytes':p.stat().st_size}
           for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (DEST/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'targets':summary,'audit':audit}),flush=True)


def sha_bytes(a):return hashlib.sha256(a.tobytes()).hexdigest()


def figures(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[summary['source_task_reference']]+[summary['targets'][t]['conditions'] for t in TASKS]
    z=np.array([[r[c]['attack_successes']/max(r[c]['clean_success_denominator'],1) for c in CONDS[1:]] for r in rows])
    fig,ax=plt.subplots(figsize=(10,4.7),constrained_layout=True)
    im=ax.imshow(z,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto')
    for i,r in enumerate(rows):
        for j,c in enumerate(CONDS[1:]):
            text=f"{r[c]['attack_successes']}/{r[c]['clean_success_denominator']}" if r[c]['clean_success_denominator'] else 'N/A'
            ax.text(j,i,text,ha='center',va='center',color='white' if z[i,j]>.55 else 'black',fontsize=13)
    ax.set_xticks(range(5),['Random full','Full both','Base-low / wrist-full','Random block','Block base'])
    ax.set_yticks(range(4),['Source: bowl -> plate','Bowl -> cabinet','Bottle -> rack','Turn on stove'])
    ax.set_title('Frozen13cm texture transfer: task failures among each task clean successes')
    fig.colorbar(im,ax=ax,label='Conditional attack success rate')
    fig.savefig(DEST/'transfer_matrix.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    x=np.arange(3);modes=['full_both','routed','block_base'];width=.24
    for j,c in enumerate(modes):
        for view in range(2):
            vals=[summary['visibility'][t][c]['median_of_episode_median_pixels_base_wrist'][view] for t in TASKS]
            ax[view].bar(x+(j-1)*width,vals,width,label=c)
    for view in range(2):
        ax[view].set_xticks(x,TASKS);ax[view].set_ylabel('Median of episode median visible pixels /224 input')
        ax[view].set_title(['Base camera','Wrist camera'][view]);ax[view].legend()
    fig.suptitle('Visibility along each attacked rollout; not matched trajectory causality')
    fig.savefig(DEST/'visibility.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(3,2,figsize=(8,11),constrained_layout=True)
    for row,t in enumerate(TASKS):
        d=np.load(DEST/t/'rollout/init40_full_both.npz')
        for col,key in enumerate(['base224','wrist224']):
            ax[row,col].imshow(d[key][0],interpolation='nearest');ax[row,col].set_xticks([]);ax[row,col].set_yticks([])
            ax[row,col].set_title(f"{t}, {'base' if col==0 else 'wrist'}: {int(d['visible_pixels224'][0,col])} px")
    fig.suptitle('Same frozen full-both texture; first inference on each target own init40')
    fig.savefig(DEST/'target_inputs.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
