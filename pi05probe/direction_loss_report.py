"""Independent E29 numeric audit, angle diagnostics, figures and provenance."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'out'
DEST = OUT/'direction_loss'
MODES = ['distance', 'orthogonal', 'reverse']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def angles(a, c, vector_ndim=2):
    """Per-sample angle; never assign a right angle to a zero vector."""
    a, c = np.asarray(a, dtype=float), np.asarray(c, dtype=float)
    shape = a.shape[:-vector_ndim] + (-1,)
    a, c = a.reshape(shape), c.reshape(shape)
    na, nc = np.linalg.norm(a, axis=-1), np.linalg.norm(c, axis=-1)
    valid = (nc > 1e-6) & (na > np.maximum(1e-6, .001*nc))
    cosine = np.sum(a*c, axis=-1) / np.maximum(na*nc, 1e-30)
    deg = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    assert valid.any(), 'all directions undefined'
    delta = a-c
    parallel = np.sum(delta*c, axis=-1) / np.maximum(nc, 1e-30)
    energy = np.sum(delta*delta, axis=-1)
    return {'angle_deg_mean': float(deg[valid].mean()),
            'angle_deg_median': float(np.median(deg[valid])),
            'angle_deg_min': float(deg[valid].min()), 'angle_deg_max': float(deg[valid].max()),
            'cosine_mean': float(cosine[valid].mean()),
            'norm_ratio_mean': float((na[valid]/nc[valid]).mean()),
            'norm_ratio_min': float((na[valid]/nc[valid]).min()),
            'norm_ratio_max': float((na[valid]/nc[valid]).max()),
            'valid_count': int(valid.sum()), 'total_count': int(valid.size),
            'near_orthogonal_80_100_fraction': float(((deg[valid]>=80)&(deg[valid]<=100)).mean()),
            'obtuse_fraction': float((deg[valid]>90).mean()),
            'squared_distance_radial_fraction': float(np.square(na[valid]-nc[valid]).sum()/max(energy[valid].sum(),1e-30)),
            'delta_perpendicular_energy_fraction': float(1-np.square(parallel[valid]).sum()/max(energy[valid].sum(),1e-30))}


def main():
    from PIL import Image
    assert read(DEST/'progress.json')['stage'] in ['measurements_complete', 'complete']
    protocol = read(DEST/'protocol.json')
    for key in ['source_sha256', 'calibration_fit_sha256']:
        assert all(sha(ROOT/n)==v for n,v in protocol[key].items()), key
    old = np.load(OUT/'dual_view/train_both/checkpoint_0500.npz')
    checkpoints = {}
    monitor_summary = {}
    training_ranges = {}
    for mode in MODES:
        run = DEST/('train_'+mode); cfg = read(run/'config.json')
        ck = np.load(run/'checkpoint_0500.npz'); checkpoints[mode] = ck
        assert cfg['complete'] and cfg['objective']==mode and ck['batches'].shape==(500,8)
        assert np.array_equal(ck['batches'],old['batches']) and np.array_equal(ck['noise'],old['noise'])
        assert all(sha(ROOT/n)==v for n,v in cfg['source_sha256'].items())
        assert cfg['mode']=='both' and cfg['parameter_shape']==[1,3,128,128]
        image = np.rint(ck['Q'].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)
        assert np.array_equal(image,np.array(Image.open(run/'texture_0500.png')))
        hist = read(run/'training_metrics.json')
        assert len(hist)==500 and all(len(r['per_frame'])==8 for r in hist)
        assert min(m['norm_ratio'] for r in hist for m in r['per_frame'])>.001
        training_ranges[mode] = {
            'minimum_cosine':min(m['cosine'] for r in hist for m in r['per_frame']),
            'minimum_norm_ratio':min(m['norm_ratio'] for r in hist for m in r['per_frame']),
            'maximum_norm_ratio':max(m['norm_ratio'] for r in hist for m in r['per_frame']),
            'all_sampled_training_cosines_positive':all(m['cosine']>0 for r in hist for m in r['per_frame'])}
        rows = read(run/'monitoring.json')
        assert [r['iteration'] for r in rows]==[0,100,200,300,400,500]
        monitor_summary[mode] = []
        for r in rows:
            row = {'iteration': r['iteration']}
            for split, n in [('train_diag',8),('validation',32)]:
                vals = r[split]; assert len(vals)==n
                cos = np.array([v['cosine'] for v in vals])
                row[split] = {key: float(np.mean([v[key] for v in vals]))
                              for key in ['objective','distance','cosine','norm_ratio']}
                row[split].update(angle_deg_mean=float(np.degrees(np.arccos(cos.clip(-1,1))).mean()),
                                  cosine_min=float(cos.min()))
            monitor_summary[mode].append(row)
    baseline = {'Q_exact_E26': bool(np.array_equal(checkpoints['distance']['Q'],old['Q'])),
                'Q_max_abs_error_E26': float(np.max(np.abs(checkpoints['distance']['Q']-old['Q']))),
                'PNG_exact_E26': sha(DEST/'train_distance/texture_0500.png')==sha(OUT/'dual_view/train_both/texture_0500.png')}
    clean_path = OUT/'render_proxy_scale/clean_baseline'
    refs = {r['init_index']:r for r in read(clean_path/'episodes.json')['per_episode']['clean']}
    good = {i for i,r in refs.items() if r['success']}; assert len(good)==9
    paired = 0
    def count(rows, path):
        nonlocal paired
        assert len(rows)==10 and {r['init_index'] for r in rows}==set(range(40,50))
        for r in rows:
            a = np.load(path/r['replay_record']); b = np.load(clean_path/refs[r['init_index']]['replay_record'])
            assert np.array_equal(a['initial_state'],b['initial_state'])
            n = min(len(a['fixture_body_pos']),len(b['fixture_body_pos']))
            for k in ['fixture_body_pos','fixture_body_quat']:
                assert np.array_equal(a[k][:n],b[k][:n])
            assert np.array_equal(a['noise_seeds'],20261006+r['init_index']*1000+np.arange(len(a['noise_seeds'])))
            paired += 1
        return {'task_successes': sum(r['success'] for r in rows), 'episodes':10,
                'attack_successes':sum(not r['success'] for r in rows if r['init_index'] in good),
                'denominator':9, 'failed_clean_success_indices':[r['init_index'] for r in rows if not r['success'] and r['init_index'] in good]}
    ep = read(DEST/'rollout/episodes.json')
    counts = {mode:count(ep['per_episode'][mode],DEST/'rollout') for mode in MODES}
    for mode in MODES:
        assert ep['metadata']['textures'][mode]['sha256']==sha(DEST/('train_'+mode)/'texture_0500.png')
    counts['random_reused'] = count(read(OUT/'dual_view/rollout_random/episodes.json')['per_episode']['random'],OUT/'dual_view/rollout_random')
    counts['distance_E26_reused'] = count(read(OUT/'dual_view/rollout_both/episodes.json')['per_episode']['optimized'],OUT/'dual_view/rollout_both')
    baseline['closedloop_success_flags_exact_E26'] = all(
        r['success']==oldr['success'] for r,oldr in zip(ep['per_episode']['distance'],read(OUT/'dual_view/rollout_both/episodes.json')['per_episode']['optimized']))
    old_episodes = {r['init_index']:r for r in read(OUT/'dual_view/rollout_both/episodes.json')['per_episode']['optimized']}
    baseline['closedloop_actions_and_states_exact_E26'] = True
    for row in ep['per_episode']['distance']:
        new_replay = np.load(DEST/'rollout'/row['replay_record'])
        old_replay = np.load(OUT/'dual_view/rollout_both'/old_episodes[row['init_index']]['replay_record'])
        baseline['closedloop_actions_and_states_exact_E26'] &= all(
            np.array_equal(new_replay[key],old_replay[key]) for key in ['actions','sim_states','noise_seeds'])
    data = np.load(DEST/'validation/inputs.npz')
    original = np.load(OUT/'dual_view/validation/inputs.npz')
    for k in ['frames','clean_base','clean_wrist','state8']:
        assert np.array_equal(data[k], original[k]), k
    baseline['both_camera_rendered_inputs_exact_E26'] = all(
        np.array_equal(data['distance__'+view], original['both__'+view]) for view in ['base','wrist'])
    assert baseline['both_camera_rendered_inputs_exact_E26']
    cap = read(DEST/'validation/capture.json')
    assert cap['both_camera_native_anchors_exact'] and cap['fixture_positions_quaternions_exact']
    probe = read(DEST/'validation/summary.json')
    v = np.load(DEST/'validation/velocities.npz')
    numeric_errors = []
    early = {}
    for j,name in enumerate(v['conditions']):
        a,c = v['velocity'][:,j].astype(float),v['clean_velocity'].astype(float)
        value = float(np.square(a-c).sum(axis=(-1,-2,-3)).mean())
        numeric_errors.append(abs(value-probe[str(name)]['L0']))
        early[str(name)] = {'all10x32':angles(a,c), 'real10x7':angles(a[...,:7],c[...,:7]),
                            'exec5x7':angles(a[...,:5,:7],c[...,:5,:7]),
                            'all10x32_by_noise_seed':{str(seed):angles(a[:,ni],c[:,ni])
                                for ni,seed in enumerate([20260901,20260902,20260903])}}
    a = np.load(DEST/'actions/actions.npz')
    assert a['actions'].shape==(32,4,3,10,7)
    action_summary = read(DEST/'actions/summary.json')
    assert action_summary['complete'] and action_summary['verification']['standard_policy_max_abs_error'] < 1e-6
    final = {}
    for j,name in enumerate(a['conditions'][1:],1):
        raw,clean = a['raw_actions'][:,j].astype(float),a['raw_actions'][:,0].astype(float)
        value = float(np.square(raw[:,:,:5,:7]-clean[:,:,:5,:7]).sum(axis=(-1,-2)).mean())
        numeric_errors.append(abs(value-action_summary['conditions'][str(name)]['final_raw_exec5_real7_squared']))
        final[str(name)] = {
            'raw_exec5x7':angles(raw[:,:,:5,:7],clean[:,:,:5,:7]),
            'output_translation_exec5x3':angles(a['actions'][:,j,:,:5,:3],a['actions'][:,0,:,:5,:3]),
            'output_translation_each_step':angles(a['actions'][:,j,:,:5,:3],a['actions'][:,0,:,:5,:3],1),
            'velocity_all10x32_by_denoise_step':[angles(a['velocity'][:,j,:,t],a['velocity'][:,0,:,t]) for t in range(10)],
            'early_x0_estimate_exec5x7':angles(a['noise'][None,:,:5,:7]-a['velocity'][:,j,:,0,:5,:7],
                                             a['noise'][None,:,:5,:7]-a['velocity'][:,0,:,0,:5,:7])}
    assert max(numeric_errors)<1e-4
    assert (DEST/'service_stopped.json').exists()
    pairwise = {}
    for i,m in enumerate(MODES):
        for n in MODES[i+1:]:
            x,y = checkpoints[m]['Q'],checkpoints[n]['Q']
            pairwise[m+'__'+n] = {'texture_MAE':float(np.abs(x-y).mean()),
                                  'texture_exact':bool(np.array_equal(x,y))}
    audit = {'complete':True,'new_training_runs':3,'gradient_frame_forwards':12000,
             'batch_and_noise_match_E26':True,'source_and_calibration_hashes_match':True,
             'PNG_matches_checkpoint':True,'new_closedloop_episodes':30,
             'new_and_reused_paired_records_checked':paired,
             'both_native256_anchors_exact':True,'validation_records':32*3*2,
             'full_denoise_count':32*4*3,'numeric_recompute_max_error':max(numeric_errors),
             'standard_policy_check':action_summary['verification'], 'service_stopped':True,
             'distance_baseline':baseline}
    clean_v = v['clean_velocity'].astype(float)
    channel_energy = {'clean_unused25_energy_fraction':float(np.square(clean_v[...,7:]).sum()/np.square(clean_v).sum()),
                      'render_delta_real7_energy_fraction':{}}
    for mode in MODES:
        j = list(v['conditions']).index(mode+'/both/render')
        dv = v['velocity'][:,j].astype(float)-clean_v
        channel_energy['render_delta_real7_energy_fraction'][mode] = float(np.square(dv[...,:7]).sum()/np.square(dv).sum())
    summary = {'complete':True,'closed_loop':counts,'early_angles':early,'final_angles':final,
               'monitoring':monitor_summary,'validation':probe,'actions':action_summary,
               'texture_pairwise':pairwise,'channel_energy':channel_energy,
               'training_ranges':training_ranges,'audit':audit,
               'notes':'Single seed/task, fixed500 not convergence proof. Test init40–49 previously used. Angles average per matched state/noise; 32 states from2 trajectories, not32 independent episodes.'}
    for filename,content in [('audit.json',audit),('summary.json',summary)]:
        (DEST/filename).write_text(json.dumps(content,indent=2,allow_nan=False)+'\n')
    figures(summary,data)
    progress = read(DEST/'progress.json')
    progress.update(stage='complete', audit_complete=True, figures_generated=True)
    (DEST/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    snap = DEST/'source_snapshot'; snap.mkdir(exist_ok=True)
    for name in list(protocol['source_sha256'])+['direction_loss_report.py']:
        shutil.copy2(ROOT/name,snap/name)
    files = {str(p.relative_to(DEST)):{'sha256':sha(p),'bytes':p.stat().st_size}
             for p in sorted(DEST.rglob('*')) if p.is_file() and p.name!='provenance.json' and p.suffix!='.log'}
    (DEST/'provenance.json').write_text(json.dumps({'complete':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
    print(json.dumps({'counts':counts,'audit':audit,'early_angles':early,'final_angles':final}),flush=True)


def figures(s,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(MODES,['#31688e','#d07025','#21918c']))
    fig,axes = plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for mode in MODES:
        rows = s['monitoring'][mode]; it = [r['iteration'] for r in rows]
        axes[0].plot(it,[r['validation']['distance'] for r in rows],'-o',label=mode,color=colors[mode])
        axes[1].plot(it,[r['validation']['angle_deg_mean'] for r in rows],'-o',label=mode,color=colors[mode])
    for i,mode in enumerate(MODES):
        n = s['closed_loop'][mode]['attack_successes']
        axes[2].bar(i,n,color=colors[mode]); axes[2].text(i,n+.15,str(n)+'/9',ha='center')
    axes[0].set_ylabel('Early squared velocity difference'); axes[1].set_ylabel('Angle(v_adv, v_clean), degrees')
    axes[0].legend(); axes[0].set_xlabel('PGD update'); axes[1].set_xlabel('PGD update')
    axes[2].set_xticks(range(3),MODES); axes[2].set_ylim(0,10); axes[2].set_ylabel('Task failures among9 clean successes')
    fig.suptitle('Fixed500 x8 loss ablation. Left/middle: same32 proxy states, one fixed noise. Right: true render.')
    fig.savefig(DEST/'loss_comparison.png',dpi=160); plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(14,4.3),constrained_layout=True)
    for mode in MODES:
        rows = s['final_angles'][mode]['velocity_all10x32_by_denoise_step']
        axes[0].plot(range(10),[r['angle_deg_mean'] for r in rows],'-o',label=mode,color=colors[mode])
    metrics = [('early_angles','all10x32','Early flow10x32'),('final_angles','raw_exec5x7','Final normalized action5x7'),
               ('final_angles','output_translation_exec5x3','Final translation command5x3')]
    for i,mode in enumerate(MODES):
        vals = [s[group][mode+'/both/render' if group=='early_angles' else mode][key]['angle_deg_mean'] for group,key,_ in metrics]
        axes[1].bar(np.arange(3)+(i-1)*.25,vals,.25,label=mode,color=colors[mode])
        energy = s['early_angles'][mode+'/both/render']['all10x32']['delta_perpendicular_energy_fraction']
        axes[2].bar(i,100*energy,color=colors[mode])
    axes[0].set_xlabel('Denoising step (0 = t1)'); axes[0].set_ylabel('Angle(v_adv, v_clean), degrees'); axes[0].legend()
    axes[1].set_xticks(range(3),['Early flow','Final raw action','Translation']); axes[1].set_ylabel('Mean angle, degrees'); axes[1].legend()
    axes[2].set_xticks(range(3),MODES); axes[2].set_ylabel('Early delta-v energy perpendicular to clean v (%)'); axes[2].set_ylim(0,100)
    fig.suptitle('True-render diagnostics: common states/noise. Perpendicular delta-v is NOT orthogonal v_adv.')
    fig.savefig(DEST/'direction_diagnostics.png',dpi=160); plt.close(fig)
    k = next(i for i in range(len(data['frames'])) if data['ep_id'][i]==10 and data['env_step'][i]==45)
    fig,axes = plt.subplots(3,3,figsize=(10,9),constrained_layout=True)
    for r,mode in enumerate(MODES):
        for c,img in enumerate([data[mode+'__texture'],data[mode+'__base'][k],data[mode+'__wrist'][k]]):
            axes[r,c].imshow(img,interpolation='nearest'); axes[r,c].set_xticks([]); axes[r,c].set_yticks([])
            if r==0: axes[r,c].set_title(['Shared physical Q128','Actual base224','Actual wrist224'][c])
        axes[r,0].set_ylabel(mode)
    fig.suptitle('Same state: init10, step45; full calibrated training, actual MuJoCo rendered inputs')
    fig.savefig(DEST/'textures_views.png',dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
