"""Strict public RoboTwin ImageWAM loading and native differentiable input."""
import sys
import numpy as np
from crossarch_common import WAM_SOURCE, WAM_WEIGHTS, AE, QWEN, WIRE
for p in (WAM_SOURCE/'src',WAM_SOURCE/'third_party/flux2/src'):
    sys.path.insert(0,str(p))
import torch
import torch.nn.functional as F
from PIL import Image

def prompts(root):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from flux2.text_encoder import OUTPUT_LAYERS_QWEN3
    from imagewam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    tasks=['Place the bowl on the plate.','Place the cup on the plate.']
    tok=AutoTokenizer.from_pretrained(str(QWEN),local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(str(QWEN),torch_dtype=torch.bfloat16,local_files_only=True).cuda().eval()
    rendered=[tok.apply_chat_template([{'role':'user','content':DEFAULT_PROMPT.format(task=t)}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False) for t in tasks]
    encoded=tok(rendered,return_tensors='pt',padding='max_length',truncation=True,max_length=128).to('cuda')
    with torch.no_grad():
        output=model(**encoded,output_hidden_states=True,use_cache=False)
        hidden=torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3],dim=1)
        hidden=hidden.permute(0,2,1,3).flatten(2)
    np.savez_compressed(root/'prompts.npz',tasks=tasks,context=hidden.float().cpu().numpy(),mask=encoded.attention_mask.cpu().numpy())

def load(root):
    from omegaconf import OmegaConf
    from hydra.utils import instantiate
    from flux2.model import Flux2,Klein4BParams
    from flux2.autoencoder import AutoEncoder,AutoEncoderParams
    from imagewam.models.backbones.imagewam import ImageWAM
    from imagewam.models.backbones.flux2_video_expert import Flux2VideoExpert
    from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
    from imagewam.models.backbones.mot import MoT
    from imagewam.models.backbones.lora import merge_lora_state_dict_to_plain
    from imagewam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    from safetensors.torch import load_file
    # Release used the old package spelling; no numerical configuration changes.
    raw=(WAM_WEIGHTS/'config.yaml').read_text().replace('fastwam.','imagewam.').replace('fastwam_processor.FastWAMProcessor','imagewam_processor.ImageWAMProcessor')
    cfg=OmegaConf.create(raw);dtype=torch.bfloat16
    with torch.device('meta'):
        video=Flux2VideoExpert(Flux2(Klein4BParams()),'klein-base-4b')
        action=ActionDiTFlux2(**OmegaConf.to_container(cfg.model.action_dit_config,resolve=True))
        mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=True,gqa_implementation='repeat',force_flash_attention=False)
        ae=AutoEncoder(AutoEncoderParams())
    payload=torch.load(WAM_WEIGHTS/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    mot.load_state_dict(merge_lora_state_dict_to_plain(payload['mot']),strict=True,assign=True)
    mot.to(device='cuda',dtype=dtype)
    ae.load_state_dict(load_file(str(AE)),strict=True,assign=True);ae.to(device='cuda',dtype=dtype)
    scheduler={f'{kind}_{name}':cfg.model[f'{kind}_scheduler'][name] for kind in ('video','action')
               for name in ('train_shift','infer_shift','num_train_timesteps')}
    model=ImageWAM(video_expert=video,action_expert=action,mot=mot,vae=ae,text_dim=7680,proprio_dim=14,
        device='cuda',torch_dtype=dtype,stack='flux2',qwen_context_len=128,pack_proprio_after_text=True,
        loss_lambda_video=.5,loss_lambda_action=1.,**scheduler)
    model.proprio_encoder.load_state_dict(payload['proprio_encoder'],strict=True)
    model.to('cuda').eval().requires_grad_(False)
    proc=instantiate(cfg.data.train.processor).eval()
    proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(WAM_WEIGHTS/'dataset_stats.json')))
    with np.load(root/'prompts.npz') as z:
        bank={str(t):dict(context=torch.tensor(z['context'][i:i+1],device='cuda',dtype=dtype),
            mask=torch.tensor(z['mask'][i:i+1],device='cuda',dtype=torch.bool)) for i,t in enumerate(z['tasks'])}
    return model,proc,bank

def compose(views):
    """Exact official PIL/BF16 forward, differentiable antialiased resize surrogate."""
    if len(views)!=3:
        raise ValueError('ImageWAM RoboTwin requires head and both wrist views.')
    resized=[]
    for x,size in zip(views,[(256,192),(128,96),(128,96)]):
        rgb=(x.detach()[0].permute(1,2,0).float().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
        oracle=np.asarray(Image.fromarray(rgb).resize(size,Image.Resampling.BILINEAR)).copy()
        y=F.interpolate(x,(size[1],size[0]),mode='bilinear',align_corners=False,antialias=True)*2-1
        target=torch.from_numpy(oracle).to(x.device,torch.bfloat16).permute(2,0,1)[None]*(2./255)-1
        resized.append(y+(target.float()-y).detach())
    return torch.cat([resized[0],torch.cat(resized[1:],dim=-1)],dim=-2).to(torch.bfloat16)

def state(proc,x):
    b={'state':{'default':torch.tensor(x,dtype=torch.float32)[None]}}
    return proc.normalizer.forward(proc.action_state_transform(b))['state']['default']

def denorm(proc,a):
    return proc.normalizer.normalizers['action']['default'].backward(a.float().cpu()).numpy()

def prefix(model,image,proprio,context):
    text,mask=model._append_proprio_to_context_if_enabled(context=context['context'],context_mask=context['mask'],proprio=proprio,source='FLUX.2 action inference')
    ref,ids=model._encode_flux2_image_tokens.__wrapped__(model,image,time_value=10.)
    pre=model.video_expert.pre_dit(x=ref.new_zeros(1,0,ref.shape[-1]),timestep=ref.new_zeros(1),
        context=text,context_mask=mask,ref_image_hidden_states=ref,target_img_ids=ids.new_zeros(1,0,4),ref_img_ids=ids)
    args=dict(batch_size=1,txt_len=int(pre['txt_len']),target_len=0,cond_len=int(pre['cond_len']),device=ref.device,text_attention_mask=pre['text_mask'])
    kv=model.mot.prefill_flux2_video_cache(video_tokens=pre['tokens'],video_freqs=pre['freqs'],video_t_mod=pre['t_mod'],
        attention_mask=model._build_mot_attention_mask_flux2(action_len=0,**args))
    return kv,model._build_mot_attention_mask_flux2(action_len=16,**args),args['txt_len']+args['cond_len']

def velocity(model,cache,noise,t):
    kv,mask,length=cache
    pre=model.action_expert.pre_dit(action_tokens=noise,timestep=model._scheduler_timestep_to_unit(t,model.infer_action_scheduler))
    x=model.mot.forward_action_with_video_cache(action_tokens=pre['tokens'],action_freqs=None,action_t_mod=pre['t_mod'],
        action_context_payload={'ids':pre['ids']},video_kv_cache=kv,attention_mask=mask,video_seq_len=length)
    return model.action_expert.post_dit(x,pre)

def noise(seed):
    return torch.randn((1,16,14),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).cuda().to(torch.bfloat16)

if __name__=='__main__':
    from crossarch_common import DEFAULT,prepare
    prepare();torch.set_num_threads(4);prompts(DEFAULT)
