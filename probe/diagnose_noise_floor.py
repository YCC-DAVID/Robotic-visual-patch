"""Isolate where the run-to-run variation in FastWAM's action chunk comes from.

The instruction sweep measures a noise floor of ~0.31 in ||dA|| even though the
flow-matching epsilon is seeded (seed=42) and the observation is byte-identical.
Something else is varying. There are two candidates:

  (A) text-encoder batching: the sweep encodes all prompts as ONE batch of 7 and
      slices rows out. If the umT5 forward is not bitwise row-independent in
      bf16, two identical prompts at different batch positions get slightly
      different embeddings.
  (B) the policy forward itself: bf16 SDPA / matmul reductions are not
      guaranteed bitwise reproducible, and FastWAM never sets
      torch.use_deterministic_algorithms.

This script separates them:
  1. encodes the same prompt at two batch positions and compares the embeddings
  2. runs infer_action N times reusing ONE context tensor, and compares chunks

If (1) is zero and (2) is not, the floor is pure GPU nondeterminism -> fixable by
averaging or by accepting it as the FD significance threshold.
If (1) is nonzero, the sweep's own noise floor was inflated by its batching and
should be recomputed with a shared context row.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FASTWAM = REPO / "third_party" / "FastWAM"

os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))

import numpy as np
import torch

_torch_load_orig = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load_orig(*a, **kw)


torch.load = _torch_load_compat

sys.path.insert(0, str(REPO / "probe"))
from run_instruction_sweep import (  # noqa: E402
    build_env_and_obs,
    encode_instructions,
    load_model,
)

N_REPEAT = 4


class _Args:
    suite = "libero_spatial"
    task_id = 0
    init_index = 0
    settle_steps = 30
    text_device = "cuda"


def main():
    from run_instruction_sweep import build_cfg

    args = _Args()
    cfg = build_cfg(args)
    env, obs, task_description = build_env_and_obs(cfg, args)

    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    prompt = DEFAULT_PROMPT.format(task=task_description)
    filler = DEFAULT_PROMPT.format(task="open the top drawer of the cabinet")

    # --- (A) same prompt at batch position 0 and 6, exactly like the sweep did.
    batch = [prompt] + [filler] * 5 + [prompt]
    ctx, mask = encode_instructions(cfg, batch, args.text_device)
    d_ctx = (ctx[0].float() - ctx[6].float()).abs().max().item()
    same_mask = bool(torch.equal(mask[0], mask[6]))
    print(f"\n(A) text encoder, identical prompt at batch rows 0 vs 6:")
    print(f"    max |dcontext| = {d_ctx:.6e}   masks equal = {same_mask}")

    # Also encode it alone, to see if batch size itself changes the embedding.
    ctx_solo, _ = encode_instructions(cfg, [prompt], args.text_device)
    d_solo = (ctx[0].float() - ctx_solo[0].float()).abs().max().item()
    print(f"    max |dcontext| batch-of-7 row0 vs batch-of-1 = {d_solo:.6e}")

    # --- (B) policy forward determinism, reusing ONE context tensor.
    model, processor, device, dtype = load_model(cfg)
    from eval_libero_single import _denormalize_action, _obs_to_model_input

    h, w = [int(v) for v in cfg.data.train.video_size]
    image, proprio, _ = _obs_to_model_input(
        obs, cfg=cfg, processor=processor, width=w, height=h, device=device, dtype=dtype
    )
    c = ctx[0:1].to(device=device, dtype=dtype)
    m = mask[0:1].to(device=device)

    chunks = []
    for i in range(N_REPEAT):
        with torch.no_grad():
            pred = model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=int(cfg.data.train.num_frames) - 1,
                proprio=proprio,
                context=c,
                context_mask=m,
                num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
                sigma_shift=cfg.EVALUATION.get("sigma_shift"),
                seed=int(cfg.seed),
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=False,
            )
        chunks.append(_denormalize_action(pred["action"], processor)[0])
        print(f"    forward {i}: |eef| mean {np.abs(chunks[-1][:, :6]).mean():.6f}")

    print(f"\n(B) policy forward, SAME context tensor, {N_REPEAT} repeats:")
    base_norm = float(np.linalg.norm(chunks[0]))
    for i in range(1, N_REPEAT):
        d = float(np.linalg.norm(chunks[i] - chunks[0]))
        print(f"    ||A_{i} - A_0|| = {d:.6e}   ({100*d/base_norm:.3f}% of ||A_0||={base_norm:.3f})")
    allclose = all(np.array_equal(chunks[i], chunks[0]) for i in range(1, N_REPEAT))
    print(f"    bitwise identical across repeats: {allclose}")

    # --- verdict
    print("\n=== verdict ===")
    if d_ctx == 0.0 and not allclose:
        print("  text encoding is exact; the floor is POLICY-FORWARD nondeterminism (bf16 kernels).")
    elif d_ctx > 0.0:
        print(f"  text encoding is NOT row-independent (max |dctx| = {d_ctx:.3e}).")
        print("  The sweep's noise floor was inflated by batching; recompute with a shared row.")
    else:
        print("  fully deterministic: floor is ~0, any instruction difference is real signal.")
    env.close()


if __name__ == "__main__":
    main()
