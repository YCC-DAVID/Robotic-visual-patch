"""Strict native ImageWAM FLUX.2 4B LIBERO inference for an unseen target.

No gradient/optimization API. Use the release's seven action dimensions,
eight proprio dimensions, 512 text positions, native AE and native 10-step flow.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT=Path(os.environ.get('WAMATTACK_PROJECT_ROOT','/shared/user65/workspace/chence/WAMattack'))
SOURCE=ROOT/'third_party/ImageWAM'
FLUX=ROOT/'third_party/flux2'
WEIGHTS=ROOT/'checkpoints/imagewam_release/libero/flux2_klein_4b'
QWEN=ROOT/'checkpoints/imagewam_flux2/qwen3_4b'
AE=Path('/shared/user64/workspace/yuhao/imagewam/checkpoints/flux2/FLUX.2-dev/ae.safetensors')
for path in [SOURCE/'src',FLUX/'src',FLUX]:sys.path.insert(0,str(path))


def processor():
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    raw=(WEIGHTS/'config.yaml').read_text().replace('fastwam.','imagewam.').replace(
        'fastwam_processor.FastWAMProcessor','imagewam_processor.ImageWAMProcessor')
    cfg=OmegaConf.create(raw)
    proc=instantiate(cfg.data.train.processor).eval()
    proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(WEIGHTS/'dataset_stats.json')))
    return cfg,proc


def crop_resize(rgb,width=224,height=224):
    image=Image.fromarray(np.asarray(rgb,np.uint8))
    w,h=image.size;scale=max(width/w,height/h)
    image=image.resize((round(w*scale),round(h*scale)),Image.Resampling.BILINEAR)
    w,h=image.size;left=max((w-width)//2,0);top=max((h-height)//2,0)
    return np.asarray(image.crop((left,top,left+width,top+height)),dtype=np.uint8)


def compose(raw):
    return np.concatenate([crop_resize(raw['base']),crop_resize(raw['wrist'])],axis=1)


def encode_context(root,task='put the bowl on the plate'):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from imagewam.models.backbones.imagewam import ImageWAM
    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    torch.set_num_threads(4)
    cfg,_=processor()
    assert cfg.data.train.qwen_context_len==512
    prompt=DEFAULT_PROMPT.format(task=task)
    encoder=AutoModelForCausalLM.from_pretrained(str(QWEN),torch_dtype=torch.bfloat16,
        local_files_only=True).to('cuda').eval().requires_grad_(False)
    owner=SimpleNamespace(text_encoder=SimpleNamespace(model=encoder,
        tokenizer=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True),max_length=512),
        device=torch.device('cuda'),torch_dtype=torch.bfloat16,model_paths={'flux2_src':str(FLUX)})
    with torch.no_grad():hidden,mask=ImageWAM._encode_flux2_prompts(owner,[prompt])
    assert hidden.shape==(1,512,7680) and mask.shape==(1,512)
    assert torch.isfinite(hidden).all() and mask.any()
    temporary=root/'contexts.tmp.pt'
    torch.save(dict(task=task,prompt=prompt,context=hidden.cpu(),mask=mask.cpu()),temporary)
    temporary.replace(root/'contexts.pt')
    print('Native Qwen3 context encoded on GPU:',tuple(hidden.shape),flush=True)


class Backend:
    def __init__(self,root):
        import torch
        from omegaconf import OmegaConf
        from flux2.model import Flux2,Klein4BParams
        from flux2.autoencoder import AutoEncoder,AutoEncoderParams
        from safetensors.torch import load_file
        from imagewam.models.backbones.imagewam import ImageWAM
        from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
        from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
        from imagewam.models.backbones.mot import MoT
        from imagewam.models.backbones.lora import merge_lora_state_dict_to_plain
        torch.set_num_threads(4)
        cfg,self.proc=processor()
        assert cfg.model.proprio_dim==8 and cfg.model.action_dit_config.action_dim==7
        assert cfg.data.train.qwen_context_len==512
        self.cfg=cfg;self.device='cuda';self.dtype=torch.bfloat16
        with torch.device('meta'):
            video=Flux2VideoExpert(Flux2(Klein4BParams()),'klein-base-4b')
            action=ActionDiTFlux2(**OmegaConf.to_container(cfg.model.action_dit_config,resolve=True))
            mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=True,
                gqa_implementation='repeat',force_flash_attention=False)
            ae=AutoEncoder(AutoEncoderParams())
        payload=torch.load(WEIGHTS/'model.pt',map_location='cpu',weights_only=True,mmap=True)
        mot.load_state_dict(merge_lora_state_dict_to_plain(payload['mot']),strict=True,assign=True)
        mot.to(device='cuda',dtype=self.dtype)
        ae.load_state_dict(load_file(str(AE)),strict=True,assign=True)
        ae.to(device='cuda',dtype=self.dtype).eval()
        scheduler={f'{kind}_{name}':int(cfg.model[f'{kind}_scheduler'][name])
            for kind in ['video','action'] for name in ['train_shift','infer_shift','num_train_timesteps']}
        self.model=ImageWAM(video_expert=video,action_expert=action,mot=mot,vae=ae,
            text_encoder=None,tokenizer=None,text_dim=7680,proprio_dim=8,device='cuda',
            torch_dtype=self.dtype,stack='flux2',qwen_context_len=512,pack_proprio_after_text=True,
            loss_lambda_video=.5,loss_lambda_action=1.,**scheduler)
        self.model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
        self.model.to('cuda').eval().requires_grad_(False)
        self.context=torch.load(root/'contexts.pt',map_location='cuda',weights_only=True)

    def prepare(self,state,raw):
        import torch
        rgb=compose(raw)
        image=torch.from_numpy(rgb.copy()).to(device='cuda',dtype=self.dtype).permute(2,0,1)[None]*(2./255)-1
        batch={'state':{'default':torch.tensor(state,dtype=torch.float32)[None]}}
        proprio=self.proc.normalizer.forward(self.proc.action_state_transform(batch))['state']['default']
        assert image.shape==(1,3,224,448) and proprio.shape==(1,8)
        return image,proprio.to(device='cuda',dtype=self.dtype)

    def native(self,image,proprio,seed):
        return self.model.infer_action(prompt=None,input_image=image,proprio=proprio,
            context=self.context['context'],context_mask=self.context['mask'],
            action_horizon=16,num_inference_steps=10,seed=seed,rand_device='cpu',tiled=False)['action'].float().cpu()

    def infer(self,request):
        import torch
        assert request['prompt']==self.context['task']
        with torch.no_grad():
            image,proprio=self.prepare(request['state'],request['images'])
            action=self.native(image,proprio,request['seed'])
        actions=self.proc.normalizer.normalizers['action']['default'].backward(action[None]).numpy()
        assert actions.shape==(1,16,7) and np.isfinite(actions).all()
        return dict(actions=actions[0])

    def manual(self,image,proprio,seed):
        import torch
        model=self.model
        text,mask=model._append_proprio_to_context_if_enabled(context=self.context['context'],
            context_mask=self.context['mask'],proprio=proprio,source='FLUX.2 action inference')
        ref,ids=model._encode_flux2_image_tokens(image,time_value=10.)
        pre=model.video_expert.pre_dit(x=ref.new_zeros(1,0,ref.shape[-1]),timestep=ref.new_zeros(1),
            context=text,context_mask=mask,ref_image_hidden_states=ref,
            target_img_ids=ids.new_zeros(1,0,ids.shape[-1]),ref_img_ids=ids)
        args=dict(batch_size=1,txt_len=int(pre['txt_len']),target_len=0,cond_len=int(pre['cond_len']),
            device=ref.device,text_attention_mask=pre['text_mask'])
        cache=model.mot.prefill_flux2_video_cache(video_tokens=pre['tokens'],video_freqs=pre['freqs'],
            video_t_mod=pre['t_mod'],attention_mask=model._build_mot_attention_mask_flux2(action_len=0,**args))
        attention=model._build_mot_attention_mask_flux2(action_len=16,**args)
        x=torch.randn((1,16,7),generator=torch.Generator(device='cpu').manual_seed(seed),
            dtype=torch.float32).to(device='cuda',dtype=self.dtype)
        times,deltas=model.infer_action_scheduler.build_inference_schedule(10,device='cuda',dtype=x.dtype)
        for t,dt in zip(times,deltas):
            a=model.action_expert.pre_dit(action_tokens=x,
                timestep=model._scheduler_timestep_to_unit(t[None],model.infer_action_scheduler))
            tokens=model.mot.forward_action_with_video_cache(action_tokens=a['tokens'],action_freqs=None,
                action_t_mod=a['t_mod'],action_context_payload={'ids':a['ids']},video_kv_cache=cache,
                attention_mask=attention,video_seq_len=args['txt_len']+args['cond_len'])
            x=model.infer_action_scheduler.step(model.action_expert.post_dit(tokens,a),dt,x)
        return x[0].float().cpu()

    def audit(self,root):
        import torch
        from crossarch_common import dump
        rows=[]
        calibration=ROOT/'pi05probe/out/dual_view/calibration/base.npz'
        with np.load(calibration) as data,torch.no_grad():
            for k in np.flatnonzero(data['validation_mask'])[:2]:
                raw=dict(base=data['clean_base'][k].copy(),wrist=data['clean_wrist'][k].copy())
                image,proprio=self.prepare(data['state8'][k],raw)
                expected=self.native(image,proprio,20261201)
                repeated=self.native(image,proprio,20261201)
                manual=self.manual(image,proprio,20261201)
                error=float((manual-expected).abs().max())
                assert torch.equal(expected,repeated) and error==0,('native parity',int(k),error)
                assert all(not p.requires_grad and p.grad is None for p in self.model.parameters())
                rows.append(dict(index=int(k),native_manual_max_abs=error,repeated_equal=True))
        dump(root/'audit/native.json',dict(passed=True,rows=rows,weights_loaded_strict=True,
            action_dim=7,proprio_dim=8,text_length=512,action_horizon=16,denoising_steps=10,
            image_shape=[1,3,224,448],no_target_gradients=True))
        return dict(passed=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['encode-context'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--task',default='put the bowl on the plate',help='LIBERO language instruction (default: #53 task)')
    args=parser.parse_args();encode_context(args.root,args.task)
