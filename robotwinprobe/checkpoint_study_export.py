"""Export controlled Aloha checkpoints with the audited upstream LoRA merger."""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import shutil

os.environ['JAX_PLATFORMS']='cpu'
import checkpoint_study_common as S
S.training_paths()
import g1_simple_pick_ft_export as E


def export(checkpoint, dest):
    checkpoint=checkpoint.resolve();dest=dest.resolve()
    audit_path=dest/'conversion_audit.json'
    if audit_path.exists():
        audit=json.loads(audit_path.read_text())
        # A shared checkpoint can have different mount aliases on each host.
        # Require the same actual directory, not merely matching path text.
        assert audit['complete'] and Path(audit['source']).samefile(checkpoint)
        assert audit['model_sha256']==S.sha(dest/'model.safetensors')
        return
    if (dest/'model.safetensors').exists():
        raise FileExistsError(f'Incomplete export requires inspection: {dest}')
    E.torch.set_num_threads(4)
    original=E.PI0Pytorch.load_state_dict; original_fold=E.converter._fold_lora_into_base_weights
    audit={}
    def checked(self, state_dict, strict=True, **kwargs):
        result=original(self,state_dict,strict=False,**kwargs)
        params=dict(self.named_parameters(remove_duplicate=False));tied=[];unused=[]
        for name in result.missing_keys:
            if name=='paligemma_with_expert.gemma_expert.lm_head.weight':
                with E.torch.no_grad():params[name].zero_()
                unused.append(name);continue
            aliases=[k for k,v in params.items() if name in params and k in state_dict
                     and v.data_ptr()==params[name].data_ptr()]
            if not aliases:raise RuntimeError(f'Unloaded parameter: {name}')
            tied.append(dict(name=name,loaded_aliases=aliases))
        assert not result.unexpected_keys,result.unexpected_keys
        audit.update(loaded_keys=len(state_dict),tied_missing=tied,unused_initialized_zero=unused)
        return result
    E.PI0Pytorch.load_state_dict=checked
    E.converter._fold_lora_into_base_weights=E.fast_fold_lora
    try:E.converter.convert_pi0_checkpoint(str(checkpoint),'bfloat16',str(dest),S.model_config())
    finally:
        E.PI0Pytorch.load_state_dict=original
        E.converter._fold_lora_into_base_weights=original_fold
    assets=checkpoint/'assets'
    assert (assets/S.ASSET/'norm_stats.json').is_file()
    shutil.copytree(assets,dest/'assets',dirs_exist_ok=True)
    config=json.loads((dest/'config.json').read_text())
    config.update(pi05=True,discrete_state_input=True,raw_action_dim=14,
                  source_jax_checkpoint=str(checkpoint),lora_folded=True)
    S.dump(dest/'config.json',config)
    audit.update(complete=True,source=str(checkpoint),lora_folded=True,
                 all_action_path_parameters_loaded=True,model_sha256=S.sha(dest/'model.safetensors'),
                 normalization_sha256=S.sha(dest/'assets'/S.ASSET/'norm_stats.json'))
    S.dump(audit_path,audit)
    print('[export]',dest,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    export(args.checkpoint,args.out)
