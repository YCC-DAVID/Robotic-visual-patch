"""Backbone-only attention, spatial KV changes and KV-direction Grad-CAM.

Reuses the paired E15 observations. No denoising, optimization or rollout.
The same digital-full V direction defines the scalar CAM target for all five
conditions, including clean (whose distance-loss gradient would be zero).
"""
import argparse
import dataclasses
import hashlib
import json
import time

import drift_patch_train as T
from instructions import match_words
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', default='kv_render_probe')
    ap.add_argument('--out', default='backbone_patch_cam')
    ap.add_argument('--frames', type=int, default=0)
    args = ap.parse_args()
    import torch
    import jax
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    from openpi.models import model as M
    from openpi.models.tokenizer import PaligemmaTokenizer

    source = T.OUT / args.inputs
    dest = T.OUT / args.out
    dest.mkdir(parents=True, exist_ok=True)
    pool = np.load(source/'inputs.npz', allow_pickle=False)
    assert np.array_equal(pool['drift__mask'],pool['band32__mask'])
    names = ['clean','optimized_digital','optimized_render','random_digital','random_render']
    cfg = C.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg,model=dataclasses.replace(cfg.model,pytorch_compile_mode=None))
    policy = PC.create_trained_policy(cfg,T.TORCH_CKPT)
    model,device = policy._model,policy._pytorch_device
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    transformer = model.paligemma_with_expert.paligemma.language_model
    projector = model.paligemma_with_expert.paligemma.model.multi_modal_projector
    captured,mode = {},{'attention':False}
    expert_calls = [0]

    def feature_hook(module,inputs,output):
        captured.setdefault('features',[]).append(output)

    def attention_hook(i):
        def hook(module,inputs,output):
            if mode['attention']:
                captured.setdefault('attention',{})[i] = output[1]
        return hook

    def expert_hook(module,inputs,output):
        expert_calls[0] += 1

    handles = [projector.register_forward_hook(feature_hook)]
    handles += [layer.self_attn.register_forward_hook(attention_hook(i))
                for i,layer in enumerate(transformer.layers)]
    # The wrapper invokes GemmaModel.forward directly, bypassing top-level
    # module hooks; its attention modules are called through __call__.
    handles.append(model.paligemma_with_expert.gemma_expert.model.layers[0].self_attn.register_forward_hook(expert_hook))

    def observation(base,wrist,k,grad=False):
        x = policy._input_transform({'observation/image':base,'observation/wrist_image':wrist,
                                     'observation/state':pool['state8'][k],'prompt':str(pool['prompt'])})
        x = jax.tree.map(lambda a:torch.from_numpy(np.array(a)).to(device)[None],x)
        obs = M.Observation.from_dict(x)
        if grad:
            for view in ['base_0_rgb','left_wrist_0_rgb']:
                obs.images[view] = obs.images[view].detach().clone().requires_grad_(True)
        return obs

    def forward(obs,attention=False):
        captured.clear()
        mode['attention'] = attention
        pack = T.prefix_and_cache(model,device,obs)
        return pack,list(captured['features']),dict(captured.get('attention',{}))

    def token_energy(x):
        return x.float().square().sum((0,1,3))

    def fraction(h,roi):
        h = np.maximum(np.asarray(h,dtype=np.float64),0)
        den = h.sum()
        return float((h*roi).sum()/den) if den>1e-20 else None

    arrays,records,layout,checks = {},[],None,{}
    steps = pool['steps'] if not args.frames else pool['steps'][:args.frames]
    maps = {key:[] for key in ['attention_nouns','attention_language','attention_rollout',
                               'kv_delta_K_energy','kv_delta_V_energy','feature_delta_energy',
                               'gradcam_positive','gradcam_signed','feature_grad_x_activation',
                               'pixel_gradient_norm']}
    tokenizer = PaligemmaTokenizer(cfg.model.max_token_len)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for k,step in enumerate(steps):
            clean_base,clean_wrist = pool['clean_base'][k],pool['clean_wrist'][k]
            ref_obs = observation(clean_base,clean_wrist,k)
            with torch.no_grad():
                clean_pack,clean_features,_ = forward(ref_obs)
                clean_features = [f.detach().clone() for f in clean_features]
                digital_pack,digital_features,_ = forward(observation(pool['drift__digital'][k],clean_wrist,k))
                digital_features = [f.detach().clone() for f in digital_features]
            valid = clean_pack[1][0].bool()
            valid_ids = valid.nonzero().flatten()
            image_names = list(ref_obs.images)
            lang_len = ref_obs.tokenized_prompt.shape[1]
            image_len = (len(valid)-lang_len)//len(image_names)
            assert image_len==256 and len(captured['features'])==len(image_names)
            views = ['base_0_rgb','left_wrist_0_rgb']
            view_slices = [slice(image_names.index(v)*image_len,(image_names.index(v)+1)*image_len)
                           for v in views]
            language_start = image_len*len(image_names)
            prompt_ids = ref_obs.tokenized_prompt[0][ref_obs.tokenized_prompt_mask[0].bool()].cpu().tolist()
            pieces = [tokenizer._tokenizer.id_to_piece(int(i)) for i in prompt_ids]
            noun_local = [match_words(pieces,['bowl']),match_words(pieces,['plate'])]
            noun_global = [[language_start+i for i in indices] for indices in noun_local]
            language_global = valid_ids[valid_ids>=language_start]
            valid_inverse = {int(v):i for i,v in enumerate(valid_ids.tolist())}
            noun_valid = [valid_inverse[i] for sub in noun_global for i in sub]
            base_mask = pool['drift__mask'][k]
            coverage = base_mask.reshape(16,14,16,14).mean((1,3))
            cv = clean_pack[2][-1][1].detach().float()
            direction = (digital_pack[2][-1][1].detach().float()-cv)[:,:,valid]
            direction_energy = direction.square().sum()
            assert float(direction_energy)>0
            if layout is None:
                layout = {'layers':len(clean_pack[2]),'image_tokens':image_len,'views':views,
                          'valid_tokens':int(valid.sum()),'pieces':pieces,'noun_piece_indices':noun_local,
                          'cam_layer':'PaliGemma multi_modal_projector output (per view, 256 tokens)',
                          'target':'last prefix V projection onto same-state optimized_digital minus clean',
                          'expert_counter_hook':'gemma_expert.model.layers[0].self_attn',
                          'expert_executed':False,'render_condition':'base rendered, wrist clean',
                          'cache_dtype':str(clean_pack[2][0][0].dtype)}
                print('[layout]',json.dumps(layout),flush=True)
            images = {'clean':clean_base,'optimized_digital':pool['drift__digital'][k],
                      'optimized_render':pool['drift__optimized_render_base'][k],
                      'random_digital':pool['drift__random_digital'][k],
                      'random_render':pool['drift__random_render_base'][k]}
            frame_maps = {key:[] for key in maps}
            for name in names:
                obs = observation(images[name],clean_wrist,k,grad=True)
                pack,features,attentions = forward(obs,attention=True)
                assert torch.equal(pack[0],clean_pack[0]) and torch.equal(pack[1],clean_pack[1])
                assert len(attentions)==len(transformer.layers)
                if k==0 and name in ['clean','optimized_digital']:
                    ref = clean_pack if name=='clean' else digital_pack
                    assert all(torch.equal(a,b) for x,y in zip(pack[2],ref[2]) for a,b in zip(x,y))
                    checks[name+'_grad_nograd_KV_equal'] = True
                # Same linear representation target for every condition. Clean
                # has score 0 but nonzero sensitivity, unlike squared distance.
                score = ((pack[2][-1][1].float()-cv)[:,:,valid]*direction).sum()/direction_energy
                selected_features = [features[image_names.index(v)] for v in views]
                leaves = [obs.images[v] for v in views]
                grads = torch.autograd.grad(score,selected_features+leaves)
                cams,signed_cams,gxa,pixel_grad,feature_changes = [],[],[],[],[]
                for j,(activation,grad) in enumerate(zip(selected_features,grads[:2])):
                    act,g = activation.detach().float(),grad.detach().float()
                    signed = (act*g.mean(1,keepdim=True)).sum(-1)[0].reshape(16,16)
                    signed_cams.append(signed.cpu().numpy())
                    cams.append(signed.clamp_min(0).cpu().numpy())
                    gxa.append((act*g).sum(-1)[0].reshape(16,16).cpu().numpy())
                    pg = grads[2+j].detach().float().square().sum(1).sqrt()[0]
                    pixel_grad.append(pg.cpu().numpy())
                    feature_changes.append((act-clean_features[image_names.index(views[j])].float())
                                           .square().sum(-1)[0].reshape(16,16).cpu().numpy())
                # Wrist pixels and visual encoding are identical; deeper prefix
                # wrist KV may change through cross-view attention.
                assert np.max(feature_changes[1])==0
                raw_nouns,raw_language,rollouts,kenergy,venergy = [],[],[],[],[]
                routing = torch.eye(len(valid_ids),device=device)
                per_layer = []
                for layer in range(len(pack[2])):
                    attention = attentions[layer].detach().float()[0].mean(0)
                    na = torch.stack([attention[indices].mean(0) for indices in noun_global])
                    la = attention[language_global].mean(0)
                    raw_nouns.append(torch.stack([na[:,s].reshape(2,16,16) for s in view_slices],1).cpu().numpy())
                    raw_language.append(torch.stack([la[s].reshape(16,16) for s in view_slices]).cpu().numpy())
                    transition = attention[valid_ids][:,valid_ids]
                    transition = transition+torch.eye(len(valid_ids),device=device)
                    transition = transition/transition.sum(-1,keepdim=True)
                    routing = transition@routing
                    r = routing[noun_valid].mean(0)
                    full_r = torch.zeros(len(valid),device=device)
                    full_r[valid_ids] = r
                    rollouts.append(torch.stack([full_r[s].reshape(16,16) for s in view_slices]).cpu().numpy())
                    layer_maps,layer_metrics = [],{}
                    for pos,label in enumerate(['K','V']):
                        a,b = clean_pack[2][layer][pos].detach(),pack[2][layer][pos].detach()
                        delta = b.float()-a.float()
                        energy = token_energy(delta)
                        base_energy = token_energy(a)
                        layer_maps.append(torch.stack([energy[s].reshape(16,16) for s in view_slices]).cpu().numpy())
                        group = {}
                        for view,s in zip(views,view_slices):
                            group[view] = float((energy[s].sum()/base_energy[s].sum()).sqrt())
                        layer_metrics[label] = {'relative_norm_all_valid':float(
                            (energy[valid].sum()/base_energy[valid].sum()).sqrt()),'views':group}
                    kenergy.append(layer_maps[0]);venergy.append(layer_maps[1])
                    attnbase = raw_nouns[-1][:,0].mean(0)
                    per_layer.append({'layer':layer+1,'kv':layer_metrics,
                        'noun_attention_base_mass':float(attnbase.sum()),
                        'noun_attention_patch_mass':float((attnbase*coverage).sum()),
                        'noun_attention_patch_fraction_within_base':fraction(attnbase,coverage),
                        'V_delta_patch_fraction_within_base':fraction(layer_maps[1][0],coverage)})
                values = {'attention_nouns':raw_nouns,'attention_language':raw_language,
                          'attention_rollout':rollouts,'kv_delta_K_energy':kenergy,'kv_delta_V_energy':venergy,
                          'feature_delta_energy':feature_changes,'gradcam_positive':cams,
                          'gradcam_signed':signed_cams,'feature_grad_x_activation':gxa,
                          'pixel_gradient_norm':pixel_grad}
                for key,value in values.items():
                    a = np.asarray(value,dtype=np.float32)
                    assert np.isfinite(a).all(),key
                    frame_maps[key].append(a)
                metrics = {'step':int(step),'condition':name,'V_direction_score':float(score.detach()),
                           'mask_area_fraction':float(base_mask.mean()),'layers':per_layer,
                           'CAM_patch_fraction_within_base':fraction(cams[0],coverage),
                           'pixel_gradient_patch_fraction_within_base':fraction(pixel_grad[0],base_mask),
                           'CAM_positive_base_sum':float(np.sum(cams[0])),
                           'CAM_positive_wrist_sum':float(np.sum(cams[1])),
                           'pixel_gradient_norm':float(np.linalg.norm(pixel_grad)),
                           'feature_base_relative_norm':float(np.sqrt(np.sum(feature_changes[0])/
                               float(clean_features[image_names.index(views[0])].float().square().sum())))}
                base_feature_id = image_names.index(views[0])
                fdelta = (selected_features[0].detach().float()-clean_features[base_feature_id].float()).flatten()
                fdigital = (digital_features[base_feature_id].float()-clean_features[base_feature_id].float()).flatten()
                den = fdelta.norm()*fdigital.norm()
                metrics['feature_delta_cosine_to_digital'] = float(torch.dot(fdelta,fdigital)/den) if float(den)>0 else None
                if k==0 and name=='optimized_digital':
                    # Verify the new scalar's pixel derivative with central
                    # differences. Perturb only interior-range normalized pixels.
                    leaf = leaves[0].detach()
                    probe_gradient = grads[2].detach().float()
                    test_direction = probe_gradient*((leaf>-.95)&(leaf<.95))
                    test_direction = test_direction/test_direction.abs().max()
                    analytic = float((probe_gradient*test_direction).sum())
                    fd_checks = []
                    for h in [.001,.003,.01]:
                        scores = []
                        for sign in [1,-1]:
                            fd_obs = observation(images[name],clean_wrist,k)
                            fd_obs.images[views[0]] = leaf+sign*h*test_direction
                            with torch.no_grad():
                                fd_pack,_,_ = forward(fd_obs)
                                fd_score = ((fd_pack[2][-1][1].float()-cv)[:,:,valid]*direction).sum()/direction_energy
                            scores.append(float(fd_score));del fd_pack,fd_obs,fd_score
                        numeric = (scores[0]-scores[1])/(2*h)
                        relative_error = abs(numeric-analytic)/max(abs(numeric),abs(analytic),1e-12)
                        fd_checks.append({'h':h,'analytic':analytic,'numeric':numeric,'relative_error':relative_error})
                    assert min(x['relative_error'] for x in fd_checks)<.25,fd_checks
                    checks['direction_score_pixel_gradient_FD'] = fd_checks
                if name=='clean':
                    assert abs(metrics['V_direction_score'])<1e-7
                if name=='optimized_digital':
                    assert abs(metrics['V_direction_score']-1)<1e-5
                records.append(metrics)
                print('[backbone]',int(step),name,'score',round(metrics['V_direction_score'],4),
                      'attn_patch',round(per_layer[-1]['noun_attention_patch_fraction_within_base'],4),
                      'cam_patch',metrics['CAM_patch_fraction_within_base'],flush=True)
                del pack,features,attentions,grads,selected_features,leaves,obs,score
                captured.clear()
            for key,value in frame_maps.items():
                maps[key].append(np.asarray(value,dtype=np.float32))
            del clean_pack,digital_pack,clean_features,digital_features,ref_obs
            # Checkpoint compact maps and records only, never dense attention.
            np.savez_compressed(dest/'maps.npz',steps=steps[:k+1],conditions=np.array(names),
                views=np.array(views),mask=pool['drift__mask'][:k+1],
                **{key:np.asarray(value,dtype=np.float32) for key,value in maps.items()})
            (dest/'metrics.json').write_text(json.dumps({'complete':False,'layout':layout,'records':records},allow_nan=False))
    finally:
        for handle in handles:
            handle.remove()
    assert expert_calls[0]==0,'Action expert unexpectedly ran'
    checks['no_action_expert_calls'] = True
    checks['unchanged_wrist_visual_features_equal'] = True
    checks['clean_score_zero_digital_score_one_all_frames'] = True
    metadata = {'complete':True,'layout':layout,'records':records,'checks':checks,
                'frames':len(steps),'inputs_source':str(source/'inputs.npz'),
                'inputs_sha256':hashlib.sha256((source/'inputs.npz').read_bytes()).hexdigest(),
                'patch_size_m':.26,'position':101,'checkpoint':str(T.TORCH_CKPT),
                'elapsed_s':time.monotonic()-started,
                'peak_gpu_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
                'notes':'Backbone only. Gradient target is a digital V direction, not action loss. '
                        'Grad-CAM averages feature gradients over 256 image tokens before channel weighting. '
                        'Raw attention and rollout are task-noun-query heuristics, not causal position scores. '
                        'Spatial KV deltas label output token positions, not isolated input pixels.'}
    (dest/'metrics.json').write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    print('[done]',len(records),'records',round(metadata['elapsed_s'],1),'seconds',flush=True)


if __name__=='__main__':
    main()
