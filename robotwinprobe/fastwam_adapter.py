"""Public FastWAM RoboTwin checkpoint, with native-equivalent differentiable inference."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from crossarch_common import ROOT, CAMERAS, WIRE, dump

SOURCE = ROOT / 'third_party/FastWAM'
ASSETS = Path('/shared/user64/workspace/yuhao/fastwam/checkpoints/Wan-AI')
sys.path.insert(0, str(SOURCE / 'src'))
sys.path.insert(0, str(SOURCE))


def config(domain='robotwin'):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base='1.3', config_dir=str(SOURCE / 'configs')):
        task='robotwin_uncond_3cam_384_1e-4' if domain=='robotwin' else 'libero_uncond_2cam224_1e-4'
        return compose(config_name='sim_'+domain, overrides=['task='+task])


def prompts(root):
    from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    with torch.device('meta'):
        encoder = WanTextEncoder()
    payload = torch.load(ASSETS / 'Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth',
                         map_location='cpu', weights_only=True, mmap=True)
    encoder.load_state_dict(payload, strict=True, assign=True)
    encoder.to(device='cuda', dtype=torch.bfloat16).eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(str(ASSETS / 'Wan2.1-T2V-1.3B/google/umt5-xxl'),
                                    seq_len=128, clean='whitespace', local_files_only=True)
    protocol=json.loads((root/'protocol.json').read_text()) if (root/'protocol.json').exists() else {}
    tasks = protocol.get('prompt_tasks',['Place the bowl on the plate.', 'Place the cup on the plate.'])
    # Match FastWAM.encode_prompt, including visible zero-padded text positions.
    ids, mask = tokenizer([DEFAULT_PROMPT.format(task=t) for t in tasks],
                          return_mask=True, add_special_tokens=True)
    with torch.no_grad():
        hidden = encoder(ids.cuda(), mask.cuda().bool())
        for i, length in enumerate(mask.gt(0).sum(1)):
            hidden[i, int(length):] = 0
    np.savez_compressed(root / 'prompts.npz', tasks=tasks,
                        context=hidden.float().cpu().numpy(), mask=np.ones_like(mask.numpy()))


class Backend:
    def __init__(self, root):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.models.wan22.fastwam import FastWAM
        from fastwam.models.wan22.wan_video_dit import WanVideoDiT
        from fastwam.models.wan22.action_dit import ActionDiT
        from fastwam.models.wan22.mot import MoT
        from fastwam.models.wan22.wan_video_vae import WanVideoVAE38
        from fastwam.models.wan22.helpers.state_dict_converters import wan_video_vae_state_dict_converter
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        protocol=json.loads((root/'protocol.json').read_text())
        self.domain=protocol.get('domain','robotwin')
        cfg = config(self.domain)
        self.action_dim=int(cfg.data.train.processor.action_output_dim)
        checkpoint='robotwin_uncond_3cam_384' if self.domain=='robotwin' else 'libero_uncond_2cam224'
        # Keep non-parameter RoPE tensors real; meta is used only for large weights.
        with torch.device('meta'):
            video = WanVideoDiT(**OmegaConf.to_container(cfg.model.video_dit_config, resolve=True))
            action = ActionDiT(**OmegaConf.to_container(cfg.model.action_dit_config, resolve=True))
            mot = MoT({'video': video, 'action': action}, mot_checkpoint_mixed_attn=True)
        payload = torch.load(root / ('checkpoints/'+checkpoint+'.pt'),
                             map_location='cpu', weights_only=True, mmap=True)
        mot.load_state_dict(payload['mot'], strict=True, assign=True)
        mot.to(device='cuda', dtype=torch.bfloat16)
        # Non-buffer frequency tensors were created in the meta context.
        from fastwam.models.wan22.wan_video_dit import precompute_freqs_cis_3d
        from fastwam.models.wan22.action_dit import precompute_freqs_cis
        video.freqs = precompute_freqs_cis_3d(video.attn_head_dim)
        action.freqs = precompute_freqs_cis(action.attn_head_dim)
        vae = WanVideoVAE38()
        vae.load_state_dict(wan_video_vae_state_dict_converter(torch.load(
            ASSETS / 'Wan2.2-TI2V-5B/Wan2.2_VAE.pth', map_location='cpu', weights_only=True, mmap=True)), strict=True)
        vae.to(device='cuda', dtype=torch.bfloat16)
        self.model = FastWAM(video, action, mot, vae, text_dim=4096, proprio_dim=int(cfg.data.train.processor.proprio_output_dim),
                             device='cuda', torch_dtype=torch.bfloat16)
        self.model.proprio_encoder.load_state_dict(payload['proprio_encoder'], strict=True)
        self.model.eval().requires_grad_(False)
        self.proc = instantiate(cfg.data.train.processor).eval()
        self.proc.set_normalizer_from_stats(load_dataset_stats_from_json(
            str(root / ('checkpoints/'+checkpoint+'_dataset_stats.json'))))
        with np.load(root / 'prompts.npz') as z:
            self.bank = {str(t): (torch.tensor(z['context'][i:i+1], device='cuda', dtype=torch.bfloat16),
                                  torch.tensor(z['mask'][i:i+1], device='cuda', dtype=torch.bool))
                         for i, t in enumerate(z['tasks'])}
        self.M = self  # Interface used by the existing diagnostic measurement routine.

    def denorm(self, proc, action):
        return proc.normalizer.normalizers['action']['default'].backward(action.float().cpu()).numpy()

    def prepare(self, state, prompt, views):
        resized = []
        sizes=[(320,256),(160,128),(160,128)] if self.domain=='robotwin' else [(224,224),(224,224)]
        if len(views)!=len(sizes):
            raise ValueError(f'FastWAM {self.domain} requires all {len(sizes)} native camera views, including wrist images.')
        for x, size in zip(views, sizes):
            rgb = (x.detach()[0].permute(1, 2, 0).float().cpu().numpy()*255).round().clip(0,255).astype(np.uint8)
            native = np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)).copy()
            target = torch.tensor(native, device='cuda', dtype=torch.bfloat16).permute(2,0,1)[None]*(2./255)-1
            y = F.interpolate(x, (size[1], size[0]), mode='bilinear', align_corners=False, antialias=True)*2-1
            resized.append(y + (target.float()-y).detach())
        image = (torch.cat([resized[0], torch.cat(resized[1:], -1)], -2) if self.domain=='robotwin'
                 else torch.cat(resized,-1)).to(torch.bfloat16)
        batch = {'state': {'default': torch.tensor(state, dtype=torch.float32)[None]}}
        proprio = self.proc.normalizer.forward(self.proc.action_state_transform(batch))['state']['default']
        return image, proprio.to(device='cuda', dtype=torch.bfloat16), self.bank[prompt]

    def raw_prepared(self, state, prompt, raw):
        keys=WIRE if self.domain=='robotwin' else ['base','wrist']
        views = [torch.tensor(raw[k].copy(), device='cuda').permute(2,0,1)[None].float()/255 for k in keys]
        return self.prepare(state, prompt, views)

    def response(self, prepared, seed, full=False):
        model = self.model
        image, proprio, (context, mask) = prepared
        context, mask = model._append_proprio_to_context(context, mask, proprio)
        # The public inference helper is decorated with no_grad. Training must
        # execute its identical body with autograd enabled back to the texture.
        first_latents = model._encode_input_image_latents_tensor.__wrapped__(model, input_image=image, tiled=False)
        pre = model.video_expert.pre_dit(x=first_latents, timestep=first_latents.new_zeros(1),
            context=context, context_mask=mask, action=None,
            fuse_vae_embedding_in_latents=bool(model.video_expert.fuse_vae_embedding_in_latents))
        length = pre['tokens'].shape[1]
        attention = model._build_mot_attention_mask(video_seq_len=length, action_seq_len=32,
            video_tokens_per_frame=int(pre['meta']['tokens_per_frame']), device=image.device)
        cache = model.mot.prefill_video_cache(video_tokens=pre['tokens'], video_freqs=pre['freqs'],
            video_t_mod=pre['t_mod'], video_context_payload={'context': pre['context'], 'mask': pre['context_mask']},
            video_attention_mask=attention[:length,:length])
        x = torch.randn((1,32,self.action_dim), generator=torch.Generator(device='cpu').manual_seed(seed),
                         dtype=torch.float32).to(device='cuda', dtype=torch.bfloat16)
        times, deltas = model.infer_action_scheduler.build_inference_schedule(10, device='cuda', dtype=x.dtype)
        first = None
        for t, dt in zip(times[:10 if full else 1], deltas):
            velocity = model._predict_action_noise_with_cache.__wrapped__(model, latents_action=x, timestep_action=t[None],
                context=context, context_mask=mask, video_kv_cache=cache,
                attention_mask=attention, video_seq_len=length)
            if first is None:
                first = velocity
            x = model.infer_action_scheduler.step(velocity, dt, x)
        return first, x

    def infer(self, req):
        prepared = self.raw_prepared(req['state'], req['prompt'], req['images'])
        image, proprio, (context, mask) = prepared
        with torch.no_grad():
            result = self.model.infer_action(prompt=None, input_image=image, proprio=proprio,
                action_horizon=32, context=context, context_mask=mask, num_inference_steps=10,
                seed=req['seed'], rand_device='cpu', tiled=False)
        return {'actions': self.denorm(self.proc, result['action'])}
