"""Avoid peer-copy corruption of PRNG keys and verify every replicated leaf."""
import hashlib
import jax
import numpy as np


def compare_fp32(actual,expected):
    assert actual.shape==expected.shape
    assert actual.dtype==expected.dtype==np.float32
    finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    return dict(passed=finite and bool(np.allclose(actual,expected,rtol=1e-6,atol=1e-7)),
        rtol=1e-6,atol=1e-7,max_abs=float(np.max(np.abs(actual-expected))) if finite else None,
        relative_l2=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-30)) if finite else None)


def replicate_key(key,sharding):
    """Upload identical key bytes from host to each GPU, not GPU-to-GPU."""
    raw=np.asarray(jax.random.key_data(key)).copy()
    parts=[jax.device_put(raw.copy(),device) for device in sharding.addressable_devices]
    result=jax.random.wrap_key_data(jax.make_array_from_single_device_arrays(raw.shape,sharding,parts))
    for shard in result.addressable_shards:
        assert np.array_equal(np.asarray(jax.random.key_data(shard.data)),raw),'Replicated PRNG key mismatch'
    return result


def audit_replicas(tree):
    import torch
    torch.set_num_threads(4)
    rows=[]
    for path,value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        groups={}
        for shard in value.addressable_shards:
            groups.setdefault(str(shard.index),[]).append(shard)
        for index,shards in groups.items():
            if len(shards)<2:continue
            digests=[]
            for shard in shards:
                shard.data.block_until_ready()
                tensor=torch.from_dlpack(shard.data).cpu().contiguous()
                arr=tensor.view(torch.uint16).numpy() if tensor.dtype==torch.bfloat16 else tensor.numpy()
                digests.append(dict(device=str(shard.device),sha256=hashlib.sha256(arr.tobytes()).hexdigest()))
            rows.append(dict(path=jax.tree_util.keystr(path),index=index,
                equal=len({d['sha256'] for d in digests})==1,replicas=digests))
    return dict(passed=all(r['equal'] for r in rows),replicated_leaves=len(rows),rows=rows)
