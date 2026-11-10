"""Native-equivalent differentiable ImageWAM LIBERO, only for experiment #55.

Policy weights stay frozen. Gradients go from first-flow 16x7 velocity MSE,
through the native video cache and AE, both camera proxies, and shared texture.
Discrete image/texture quantization uses an explicitly declared STE.
"""
import numpy as np
from imagewam_libero_adapter import Backend as InferenceBackend,crop_resize

class Backend(InferenceBackend):
    def prepare_views(self,state,views,quantized=True):
        import torch
        import torch.nn.functional as F
        images=[]
        for view in views:
            y=F.interpolate(view,(224,224),mode='bilinear',align_corners=False,antialias=True)*2-1
            if quantized:
                raw=np.rint(view.detach()[0].permute(1,2,0).float().cpu().numpy()*255).clip(0,255).astype(np.uint8)
                target=torch.tensor(crop_resize(raw).copy(),device='cuda',dtype=self.dtype).permute(2,0,1)[None]*(2./255)-1
                y=y+(target.float()-y).detach()
            images.append(y)
        image=torch.cat(images,dim=-1).to(self.dtype)
        batch={'state':{'default':torch.tensor(state,dtype=torch.float32)[None]}}
        proprio=self.proc.normalizer.forward(self.proc.action_state_transform(batch))['state']['default']
        return image,proprio.to(device='cuda',dtype=self.dtype)

    def video_cache(self,pre,args,checkpointed):
        import torch
        from torch.utils.checkpoint import checkpoint
        from flux2.model import apply_rope
        model=self.model;mot=model.mot
        masks=model._build_mot_attention_mask_flux2(action_len=0,**args)
        if not checkpointed:
            return mot.prefill_flux2_video_cache(video_tokens=pre['tokens'],video_freqs=pre['freqs'],video_t_mod=pre['t_mod'],attention_mask=masks)
        expert=model.video_expert;img=pre['tokens']['img'];txt=pre['tokens']['txt'];double=[];single=[]
        img_pe=pre['freqs']['img'];txt_pe=pre['freqs']['txt'];mods=pre['t_mod']
        for block in expert.double_blocks:
            def forward(i,t,block=block):
                q,k,v,pe,n,m=block._prepare_qkv(i,t,img_pe,txt_pe,mods['double_img'],mods['double_txt'])
                q,k=apply_rope(q,k,pe)
                q,k,v=map(mot._flux2_flatten_heads,(q,k,v))
                mixed=mot._mixed_attention(q,k,v,masks['double_joint'])
                ta,ia=torch.split(mixed,[n,i.shape[1]],dim=1)
                i,t=block._apply_residuals(i,t,ia,ta,m)
                return i,t,k,v
            img,txt,k,v=checkpoint(forward,img,txt,use_reentrant=False)
            double.append({'k':k,'v':v})
        stream=torch.cat([txt,img],dim=1);pe=torch.cat([txt_pe,img_pe],dim=2)
        for block in expert.single_blocks:
            def forward(x,block=block):
                st=mot._flux2_video_single_io(block,x,pe,mods['single'])
                mixed=mot._mixed_attention(st['q'],st['k'],st['v'],masks['single'])
                return block._out(st['residual_x'],mixed,st['mlp'],st['gate']),st['k'],st['v']
            stream,k,v=checkpoint(forward,stream,use_reentrant=False)
            single.append({'k':k,'v':v})
        return {'double':double,'single':single,'txt_len':int(txt.shape[1]),'img_len':int(img.shape[1]),
                'final_video':{'txt':stream[:,:txt.shape[1]],'img':stream[:,txt.shape[1]:]}}

    def response(self,image,proprio,seed,full=False,return_intermediates=False):
        import torch
        model=self.model
        text,mask=model._append_proprio_to_context_if_enabled(context=self.context['context'],context_mask=self.context['mask'],proprio=proprio,source='FLUX.2 action inference')
        ref,ids=model._encode_flux2_image_tokens.__wrapped__(model,image,time_value=10.)
        pre=model.video_expert.pre_dit(x=ref.new_zeros(1,0,ref.shape[-1]),timestep=ref.new_zeros(1),context=text,context_mask=mask,ref_image_hidden_states=ref,target_img_ids=ids.new_zeros(1,0,ids.shape[-1]),ref_img_ids=ids)
        args=dict(batch_size=1,txt_len=int(pre['txt_len']),target_len=0,cond_len=int(pre['cond_len']),device=ref.device,text_attention_mask=pre['text_mask'])
        cache=self.video_cache(pre,args,torch.is_grad_enabled())
        attention=model._build_mot_attention_mask_flux2(action_len=16,**args)
        x=torch.randn((1,16,7),generator=torch.Generator(device='cpu').manual_seed(seed),dtype=torch.float32).to(device='cuda',dtype=self.dtype)
        times,deltas=model.infer_action_scheduler.build_inference_schedule(10,device='cuda',dtype=x.dtype)
        first=None
        for t,dt in zip(times[:10 if full else 1],deltas):
            a=model.action_expert.pre_dit(action_tokens=x,timestep=model._scheduler_timestep_to_unit(t[None],model.infer_action_scheduler))
            tokens=model.mot.forward_action_with_video_cache(action_tokens=a['tokens'],action_freqs=None,action_t_mod=a['t_mod'],action_context_payload={'ids':a['ids']},video_kv_cache=cache,attention_mask=attention,video_seq_len=args['txt_len']+args['cond_len'])
            velocity=model.action_expert.post_dit(tokens,a)
            if first is None:first=velocity
            x=model.infer_action_scheduler.step(velocity,dt,x)
        return (first,x,ref) if return_intermediates else (first,x)
