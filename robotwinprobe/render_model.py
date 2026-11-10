"""Frozen RoboTwin pi0.5; prefix KV and first flow step are separate outputs."""
import dataclasses
import numpy as np
import bootstrap as B
B.model_paths()


def load():
    import torch
    from openpi.training import config as C
    from openpi.policies import policy_config as PC
    torch.set_num_threads(4)
    cfg = C.get_config('pi05_aloha')
    cfg = dataclasses.replace(cfg,
        model=dataclasses.replace(cfg.model, action_horizon=32, pytorch_compile_mode=None),
        data=C.LeRobotAlohaDataConfig(assets=C.AssetsConfig(asset_id='pi0.5_clean_randomize_joint_training'),
                                     adapt_to_pi=True, use_delta_joint_actions=True))
    policy = PC.create_trained_policy(cfg, B.ROOT/'checkpoints/pi05_robotwin2_pytorch')
    policy._model.eval()
    for p in policy._model.parameters(): p.requires_grad_(False)
    return policy


def observation(policy, state, prompt, images):
    import torch, jax
    from openpi.models import model as M
    from policy_server import aloha_images
    x = policy._input_transform({'state':state, 'prompt':str(prompt), 'images':aloha_images(images)})
    x = jax.tree.map(lambda a: torch.from_numpy(np.array(a)).to(policy._pytorch_device)[None], x)
    return M.Observation.from_dict(x)


def prefix(model, obs):
    import torch
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    images, masks, tokens, token_masks, state = model._preprocess_observation(obs, train=False)
    embeds, pad, att = model.embed_prefix(images, masks, tokens, token_masks)
    mask = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
    _, kv = model.paligemma_with_expert.forward(attention_mask=mask,
        position_ids=torch.cumsum(pad, dim=1)-1, past_key_values=None,
        inputs_embeds=[embeds,None], use_cache=True)
    return state, pad, kv


def velocity(model, obs, noise):
    import torch
    state, pad, kv = prefix(model, obs)
    return model.denoise_step(state, pad, kv, noise, torch.ones(noise.shape[0],device=noise.device))
