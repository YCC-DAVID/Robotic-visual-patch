"""Same LIBERO scene, different instructions -> compare FastWAM's action chunks.

Holds the observation completely fixed (one LIBERO task, one init state, a fixed
number of settling steps) and varies ONLY the language prompt, so any difference
in the predicted action chunk is attributable to the instruction.

Preprocessing, model construction and (de)normalization are imported from
FastWAM's own `experiments/libero/eval_libero_single.py` rather than
reimplemented, so this cannot silently drift from the official eval path.

Memory note: the umT5-XXL text encoder is ~11 GB of the ~24 GB that loading the
whole model on one GPU would need. Since this script only needs a handful of text
embeddings, it encodes them up front (on CPU by default), frees the encoder, and
then builds the policy with `load_text_encoder=false`, passing the cached
`context`/`context_mask` into `infer_action`. That keeps GPU use to ~13 GB, which
matters on a shared node. Use `--text-device cuda` to encode on GPU instead.

Run:
    MUJOCO_GL=egl python probe/run_instruction_sweep.py
    MUJOCO_GL=egl python probe/run_instruction_sweep.py --suite libero_spatial --task-id 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FASTWAM = REPO / "third_party" / "FastWAM"

# Must be set before `libero` is imported: libero reads the config path at import
# time and would otherwise pick up the machine-global ~/.libero/config.yaml,
# which points at an unrelated LIBERO checkout.
os.environ.setdefault("LIBERO_CONFIG_PATH", str(REPO / "probe" / "config" / "libero"))
os.environ.setdefault("MUJOCO_GL", "egl")
# FastWAM's loader resolves auxiliary weights under <base>/<model_id>/<pattern>.
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(REPO / "checkpoints"))

# `eval_libero_single` does a bare `from action_ensembler import ...`, which only
# resolves when its own directory is on sys.path.
sys.path.insert(0, str(FASTWAM / "experiments" / "libero"))

import numpy as np
import torch

# torch >= 2.6 flipped torch.load's `weights_only` default to True. Both LIBERO's
# .pruned_init files (pickled numpy arrays) and FastWAM's released .pt (which
# carries non-tensor metadata) are plain pickles and fail under the new default.
# Both are local, trusted files, so restore the old behaviour process-wide.
_torch_load_orig = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load_orig(*a, **kw)


torch.load = _torch_load_compat

CKPT = REPO / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224.pt"
STATS = REPO / "checkpoints" / "fastwam_release" / "libero_uncond_2cam224_dataset_stats.json"

# (label, instruction). None => use the task's own BDDL language string.
DEFAULT_INSTRUCTIONS = [
    ("true", None),
    ("paraphrase", "grasp the black bowl in the middle and set it down on the plate"),
    ("other_object", "pick up the cookies and place them on the plate"),
    ("wrong_target", "pick up the black bowl and place it on the stove"),
    ("unrelated", "open the top drawer of the cabinet"),
    ("empty", ""),
]


def build_cfg(args):
    from hydra import compose, initialize_config_dir

    for p in (CKPT, STATS):
        if not p.exists():
            raise SystemExit(f"missing required file: {p}")

    with initialize_config_dir(config_dir=str(FASTWAM / "configs"), version_base="1.3"):
        return compose(
            config_name="sim_libero",
            overrides=[
                "task=libero_uncond_2cam224_1e-4",
                f"ckpt={CKPT}",
                f"EVALUATION.dataset_stats_path={STATS}",
                f"EVALUATION.task_suite_name={args.suite}",
                f"EVALUATION.task_id={args.task_id}",
                # The redirect targets a ModelScope-only repo (404 on HF); we
                # pre-fetched the original .pth files from the Wan repos instead.
                "model.redirect_common_files=false",
                # Text embeddings are precomputed below, so the policy itself
                # never needs the 11 GB encoder resident.
                "model.load_text_encoder=false",
            ],
        )


def encode_instructions(cfg, prompts, device, batched=False):
    """Build ONLY the umT5 encoder + tokenizer, encode prompts, then free them.

    Mirrors FastWAM.encode_prompt (fastwam.py:201-217) exactly, including the
    zero-out-past-seq_len step and the all-ones mask it returns.

    Each prompt is encoded ALONE (batch size 1) by default. This matters: umT5 in
    bf16 is NOT batch-invariant -- encoding the same prompt at different batch
    positions changes its embedding by max|d| ~= 1.3 (bf16 GEMM accumulation order
    depends on batch shape). Encoding as one batch and slicing rows out therefore
    injects a spurious ~0.31 difference into every pairwise action comparison,
    which is far larger than the real effect of a paraphrase. Batch-size-1 is also
    what the official eval does, since `infer_action(prompt=...)` calls
    `encode_prompt` with a single string. Pass batched=True only to reproduce the
    artifact (see probe/diagnose_noise_floor.py).
    """
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    dtype = torch.bfloat16
    _dit_c, text_c, _vae_c, tok_c = _resolve_configs(
        model_id=cfg.model.model_id,
        tokenizer_model_id=cfg.model.tokenizer_model_id,
        redirect_common_files=bool(cfg.model.redirect_common_files),
    )
    text_c.download_if_necessary()
    tok_c.download_if_necessary()
    print(f"[text] encoding {len(prompts)} prompts on {device}", flush=True)

    text_encoder = _load_registered_model(
        text_c.path, "wan_video_text_encoder", torch_dtype=dtype, device=device
    )
    # CRITICAL: _load_registered_model does NOT call .eval(). WanTextEncoder has
    # nn.Dropout(0.1), so without this the embeddings are randomly corrupted --
    # repeated encodings of the SAME string come out essentially uncorrelated
    # (L2 difference ~= 103% of ||context||), which swamps any real semantic
    # difference between instructions. FastWAM's own path avoids this because
    # infer_action calls self.eval() on the whole module before encode_prompt.
    text_encoder.eval().requires_grad_(False)
    tokenizer = HuggingfaceTokenizer(
        name=tok_c.path, seq_len=int(cfg.model.tokenizer_max_len), clean="whitespace"
    )

    groups = [prompts] if batched else [[p] for p in prompts]
    ctx_rows, mask_rows = [], []
    with torch.no_grad():
        for group in groups:
            ids, mask = tokenizer(group, return_mask=True, add_special_tokens=True)
            ids = ids.to(device)
            mask = mask.to(device, dtype=torch.bool)
            emb = text_encoder(ids, mask)
            seq_lens = mask.gt(0).sum(dim=1).long()
            for i, v in enumerate(seq_lens):
                emb[i, v:] = 0
            mask = torch.ones_like(mask)
            ctx_rows.append(emb.detach().to("cpu"))
            mask_rows.append(mask.detach().to("cpu"))

    context = torch.cat(ctx_rows, dim=0)
    context_mask = torch.cat(mask_rows, dim=0)
    print(f"[text] context {tuple(context.shape)} mask {tuple(context_mask.shape)} "
          f"(batched={batched})", flush=True)

    del text_encoder, tokenizer, ctx_rows, mask_rows
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return context, context_mask


def load_model(cfg):
    from hydra.utils import instantiate
    from eval_libero_single import (
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _resolve_dataset_stats_path,
        _resolve_eval_device,
    )
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    device = _resolve_eval_device(cfg)
    dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    print(f"[model] building on {device} dtype={dtype} (no text encoder)", flush=True)

    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(device).eval()
    if torch.cuda.is_available():
        print(f"[model] cuda mem allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(_resolve_dataset_stats_path(cfg)))
    )
    return model, processor, device, dtype


def build_env_and_obs(cfg, args):
    """One env, one init state, N settling steps -> a single frozen observation."""
    from libero.libero import benchmark
    from libero_utils import LIBERO_ENV_RESOLUTION, get_libero_dummy_action, get_libero_env

    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    init_states = suite.get_task_init_states(args.task_id)
    print(f"[env] {args.suite} task {args.task_id}: {task.language!r}", flush=True)
    print(f"[env] {len(init_states)} init states, using #{args.init_index}", flush=True)

    env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, int(cfg.seed))
    env.reset()
    obs = env.set_init_state(init_states[args.init_index])
    # Mirror the official eval: settle the scene with no-op actions before the
    # first policy forward.
    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(get_libero_dummy_action())
    return env, obs, task_description


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--settle-steps", type=int, default=30, help="num_steps_wait in the official eval")
    ap.add_argument("--text-device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default=str(REPO / "probe" / "out" / "instruction_sweep.json"))
    args = ap.parse_args()

    cfg = build_cfg(args)
    print(f"[cfg] seed={cfg.seed} num_inference_steps={cfg.EVALUATION.num_inference_steps} "
          f"replan_steps={cfg.EVALUATION.replan_steps}", flush=True)

    env, obs, task_description = build_env_and_obs(cfg, args)

    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

    instructions = [(lbl, task_description if txt is None else txt) for lbl, txt in DEFAULT_INSTRUCTIONS]
    # Repeat the true instruction to measure the sampling/compute noise floor,
    # which is the significance threshold for every other difference below.
    instructions.append(("true_repeat", task_description))
    prompts = [DEFAULT_PROMPT.format(task=txt) for _, txt in instructions]

    context_all, mask_all = encode_instructions(cfg, prompts, args.text_device)
    model, processor, device, dtype = load_model(cfg)

    from eval_libero_single import _denormalize_action, _obs_to_model_input

    h, w = [int(v) for v in cfg.data.train.video_size]
    image, proprio, _imgs = _obs_to_model_input(
        obs, cfg=cfg, processor=processor, width=w, height=h, device=device, dtype=dtype
    )
    print(f"[obs] model input {tuple(image.shape)} range "
          f"[{float(image.min()):.3f}, {float(image.max()):.3f}] proprio {tuple(proprio.shape)}", flush=True)

    action_horizon = int(cfg.data.train.num_frames) - 1
    H_exec = int(cfg.EVALUATION.replan_steps)

    results = {}
    for i, (label, instruction) in enumerate(instructions):
        with torch.no_grad():
            pred = model.infer_action(
                prompt=None,
                input_image=image,
                action_horizon=action_horizon,
                proprio=proprio,
                context=context_all[i : i + 1].to(device=device, dtype=dtype),
                context_mask=mask_all[i : i + 1].to(device=device),
                num_inference_steps=int(cfg.EVALUATION.num_inference_steps),
                sigma_shift=cfg.EVALUATION.get("sigma_shift"),
                seed=int(cfg.seed),
                rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
                tiled=False,
            )
        action = _denormalize_action(pred["action"], processor)[0]  # [32, 7]
        results[label] = {"instruction": instruction, "action": action}
        print(f"[infer] {label:14s} chunk {action.shape} "
              f"|eef| mean {np.abs(action[:, :6]).mean():.4f} "
              f"gripper[:5] {np.round(action[:5, -1], 3).tolist()}", flush=True)

    noise_floor = float(np.linalg.norm(results["true"]["action"] - results["true_repeat"]["action"]))
    print(f"\n=== noise floor (true vs true_repeat, same seed): {noise_floor:.6e} ===")

    labels = [lbl for lbl, _ in instructions if lbl != "true_repeat"]
    print(f"\n=== pairwise ||A_i - A_j||  (full {action_horizon}-step / first {H_exec} executed) ===")
    print("                " + "".join(f"{l[:12]:>14s}" for l in labels))
    pairwise = {}
    for a in labels:
        row = f"{a[:14]:16s}"
        for b in labels:
            full = float(np.linalg.norm(results[a]["action"] - results[b]["action"]))
            pref = float(np.linalg.norm(results[a]["action"][:H_exec] - results[b]["action"][:H_exec]))
            pairwise[f"{a}|{b}"] = {"full": full, "prefix": pref}
            row += f"{full:7.3f}/{pref:6.3f}"
        print(row)

    a_true = float(np.linalg.norm(results["true"]["action"]))
    print(f"\n=== vs true instruction  (||A_true|| = {a_true:.3f}, noise floor = {noise_floor:.3e}) ===")
    if noise_floor == 0.0:
        print("    noise floor is exactly 0 (bitwise-identical repeat), so any nonzero")
        print("    ||dA|| below is real instruction signal, not sampling noise.")
    for lbl in labels:
        if lbl == "true":
            continue
        d = pairwise[f"true|{lbl}"]["full"]
        rel = 100.0 * d / a_true
        extra = "" if noise_floor == 0.0 else f"   {d / noise_floor:6.1f}x floor"
        print(f"  {lbl:14s} ||dA|| = {d:8.4f}   {rel:6.2f}% of ||A_true||{extra}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            {
                "suite": args.suite,
                "task_id": args.task_id,
                "task_description": task_description,
                "init_index": args.init_index,
                "settle_steps": args.settle_steps,
                "seed": int(cfg.seed),
                "num_inference_steps": int(cfg.EVALUATION.num_inference_steps),
                "action_horizon": action_horizon,
                "replan_steps": H_exec,
                "noise_floor": noise_floor,
                "instructions": {k: v["instruction"] for k, v in results.items()},
                "actions": {k: v["action"].tolist() for k, v in results.items()},
                "pairwise": pairwise,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {out}")
    env.close()


if __name__ == "__main__":
    main()
