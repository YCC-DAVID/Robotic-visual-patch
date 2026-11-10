"""CPU-only independent verification of #57 saved six-point diagnostic evidence."""
from pathlib import Path
import json,hashlib,time
import numpy as np
ROOT=Path('/shared/user65/workspace/chence/WAMattack')
ORIG=ROOT/'robotwinprobe/out/pi_imagewam_joint_eot_20261031'
D=ORIG/'audit/gradient_replay'
STEPS=[1,100,200,300,400,500]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ash(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
plan=read(D/'replay_inputs.json');assert plan['steps']==STEPS
original_checks={n:sha(ORIG/n)==v for n,v in plan['original_hashes'].items()};assert all(original_checks.values())
source_checks={n:sha(ROOT/('pi05probe' if n in ['dual_view_proxy.py','render_world_proxy.py'] else 'robotwinprobe')/n)==v for n,v in plan['source_hashes'].items()};assert all(source_checks.values())
batches=read(ORIG/'training_plan.json');draws=read(ORIG/'eot_plan.json')
ledgers={k:[json.loads(line) for line in (ORIG/'gradient_ledgers'/f'{k}.jsonl').read_text().splitlines()] for k in ['pi','imagewam']}
with np.load(ORIG/'training/state.npz') as z:
 history=json.loads(str(z['history']));signs=z['update_signs'].copy();final=z['texture'].copy()
q=np.full_like(final,.5);pre={};assert len(history)==len(signs)==500
for i,h in enumerate(history):
 assert ash(q)==h['before_sha256'];assert ash(signs[i])==h['sign_sha256']
 if i+1 in STEPS:pre[i+1]=q.copy()
 q=np.clip(q+.01*signs[i].astype(np.float32),0,1).astype(np.float32);assert ash(q)==h['after_sha256']
 for k in ['pi','imagewam']:assert h['provenance'][k]==ledgers[k][i]
assert np.array_equal(q,final)
rows=[]
for point in plan['points']:
 step=point['step'];h=history[step-1];assert point['history']==h;assert point['batch']==batches[step-1] and point['draws']==draws[step-1]
 assert sha(Path(point['input_npz']))==point['input_npz_sha256']
 with np.load(point['input_npz']) as z:assert np.array_equal(z['q'],pre[step]) and np.array_equal(z['original_update_sign'],signs[step-1])
 f=D/'collected'/f'step_{step:04d}.npz';row=read(f.with_suffix('.json'));assert sha(f)==row['arrays_sha256'];assert row['collection_complete'] and row['step']==step
 with np.load(f) as z:a={k:z[k].copy() for k in z.files}
 assert np.array_equal(a['q'],pre[step]);assert row['input_npz_sha256']==point['input_npz_sha256'];assert all(np.isfinite(v).all() for v in a.values())
 caches={};metas={}
 for suffix in ['pi','imagewam','cross_pi']:
  cf=D/'serial_staging'/f'step_{step:04d}_{suffix}.npz';m=read(cf.with_suffix('.json'));assert sha(cf)==m['arrays_sha256'] and m['input_sha256']==point['input_npz_sha256']
  with np.load(cf) as z:caches[suffix]={k:z[k].copy() for k in z.files}
  metas[suffix]=m;assert np.array_equal(caches[suffix]['q'],a['q'])
 for k in ['pi','imagewam']:
  g=a['gradient_'+k];m=metas[k];c=caches[k];actual=row['reproduction'][k]['provenance'];expected=h['provenance'][k]
  assert g.shape==q.shape and np.linalg.norm(g)>0 and ash(g)==row['gradient_sha256'][k]
  assert np.array_equal(g,c['gradient']) and np.array_equal(a['losses_before_'+k],c['losses'])
  assert len(a['losses_before_'+k])==8 and float(np.mean(a['losses_before_'+k]))==h['raw'][k]==row['reproduction'][k]['raw_loss']
  assert actual==m['provenance'];assert float(np.linalg.norm(g))==actual['gradient_norm']==row['reproduction'][k]['gradient_norm']
  assert set(actual)==set(expected);assert {n:v for n,v in actual.items() if n!='gradient_norm'}=={n:v for n,v in expected.items() if n!='gradient_norm'}
  assert actual['indices']==point['batch']['indices'] and actual['seeds']==point['batch']['seeds']
  assert actual['draws_sha256']==hashlib.sha256(json.dumps(point['draws'],sort_keys=True).encode()).hexdigest()
  delta=np.clip(a['q']+.01*np.sign(g),0,1).astype(np.float32)-a['q'];assert np.array_equal(delta,a['delta_'+k]) and np.array_equal(delta,c['delta'])
 mappings=[('pi','pi','pi','losses_after_pi'),('imagewam','imagewam','imagewam','losses_after_imagewam'),('pi','imagewam','imagewam','losses_after_pi'),('imagewam','pi','cross_pi','losses_after_imagewam')]
 for candidate,k,cache,key in mappings:
  assert np.array_equal(a[f'losses_after_{candidate}_on_{k}'],caches[cache][key])
  effect=row['effects'][candidate][k]
  assert effect['first_order_projection']==float(np.sum(a['gradient_'+k].astype(np.float64)*a['delta_'+candidate].astype(np.float64)))
  assert effect['finite_step_loss_difference']==float(np.mean(a[f'losses_after_{candidate}_on_{k}'])-np.mean(a['losses_before_'+k]))
  assert effect['loss_before']==float(np.mean(a['losses_before_'+k])) and effect['loss_after']==float(np.mean(a[f'losses_after_{candidate}_on_{k}']))
 cosine=float(np.sum(a['gradient_pi']*a['gradient_imagewam'])/(np.linalg.norm(a['gradient_pi'])*np.linalg.norm(a['gradient_imagewam'])))
 agreement=float(np.mean(np.sign(a['gradient_pi'])==np.sign(a['gradient_imagewam'])));assert cosine==row['cosine'] and agreement==row['sign_agreement']
 selected_agreement=float(np.mean(np.sign(a['gradient_'+h['chosen']]).astype(np.int8)==signs[step-1]))
 checks=dict(provenance={k:row['reproduction'][k]['provenance']==h['provenance'][k] for k in ['pi','imagewam']},raw_loss={k:float(np.mean(a['losses_before_'+k]))==h['raw'][k] for k in ['pi','imagewam']},cosine=cosine==h['cosine'],selected_sign=selected_agreement==1.)
 exact=all(checks['provenance'].values()) and all(checks['raw_loss'].values()) and checks['cosine'] and checks['selected_sign'];assert checks==row['exact_checks'] and exact==row['passed']==row['exact_historical_gradient_gate_passed']
 for suffix,names in [('pi',['models/pi/native_audit.json']),('imagewam',['audit/native_and_gradient.json','audit/imagewam_gray_match.json']),('cross_pi',['models/pi/native_audit.json'])]:
  for name in names:assert read(Path(metas[suffix]['work'])/name)['passed']
 for suffix in ['pi','imagewam','cross_pi']:
  w=Path(metas[suffix]['work']);assert w.is_relative_to(D/'work');cfg=read(w/'protocol.json');assert cfg['diagnostic_only']['training'] is False and cfg['diagnostic_only']['closed_loops']==0
  for k in ['pi','imagewam']:
   model=read(w/f'models/{k}/protocol.json');assert model['parent_root']==str(w) and model['training']['isolated_source_model']==k and model['evaluation']['conditions']==[] and model['control_port'] in [18962,18963]
 rows.append(dict(step=step,exact_historical_passed=bool(exact),exact_checks=checks,selected_source=h['chosen'],selected_sign_agreement=selected_agreement,cosine=cosine,original_cosine=h['cosine'],sign_agreement=agreement,effects=row['effects'],norm_relative_difference={k:row['reproduction'][k]['gradient_norm']/h['provenance'][k]['gradient_norm']-1 for k in ['pi','imagewam']},gradient_sha256=row['gradient_sha256'],npz_sha256=sha(f),row_sha256=sha(f.with_suffix('.json')),input_sha256=point['input_npz_sha256'],original_pre_q_sha256=ash(a['q'])))
assert len(rows)==6
out=dict(diagnostic_evidence_integrity_passed=True,collected_diagnostic_points=6,fixed_steps=STEPS,exact_historical_replay_passed=False,exact_historical_passed_points=sum(x['exact_historical_passed'] for x in rows),historical_tolerance=0,all_500_before_after_sha_verified=True,all_original_hashes_unchanged=original_checks,all_source_hashes_match_original=source_checks,original_batch_noise_draw_verified=True,both_forward_losses_exact_at_all_six_points=True,full_gradients_and_hashes_verified=True,actual_clipped_deltas_and_projections_verified=True,finite_step_loss_arrays_and_differences_verified=True,all_three_serial_phase_native_audits_passed=True,work_protocols_isolated=True,new_training_steps=0,new_closed_loop_episodes=0,rows=rows,limitation='Native recomputation diagnostic values, not recovered historical gradient tensors. All exact mismatches remain failures. Six fixed points cannot prove all-500-step causality or identify the mechanism of zero pi ASR.',verified_at=time.time(),verifier_sha256=sha(Path(__file__)))
dump(D/'independent_evidence_verification.json',out)
print(json.dumps(dict(verified=True,points=6,exact_historical_passed_points=out['exact_historical_passed_points'],rows=rows)))
