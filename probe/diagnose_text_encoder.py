"""Is the umT5 text encoder deterministic across repeated identical calls?

The instruction sweep still shows a nonzero noise floor (||dA|| ~ 0.46) after
switching to per-prompt (batch size 1) encoding, even though the policy forward
is provably bitwise deterministic for a fixed context tensor. So the variation
must originate in the text encoder.

This loads ONLY umT5 (no policy, so it's fast) and measures, at the embedding
level:
  - WITHIN-prompt spread: encode the same string R times, compare
  - BETWEEN-prompt distance: compare different strings

The ratio of those two is the embedding-level signal-to-noise. If within-prompt
spread is 0, the encoder is deterministic and the sweep's floor comes from
somewhere else. If it is nonzero and comparable to the between-prompt distance,
then instruction comparisons need the embeddings cached once and reused, and the
within-prompt spread is the true significance threshold.
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
sys.path.insert(0, str(REPO / "probe"))

import numpy as np
import torch

_torch_load_orig = torch.load


def _torch_load_compat(*a, **kw):
    kw.setdefault("weights_only", False)
    return _torch_load_orig(*a, **kw)


torch.load = _torch_load_compat

R = 3
DEVICE = "cuda"
# Set PROBE_TEXT_EVAL=0 to reproduce the dropout-corrupted behaviour.
EVAL_MODE = os.environ.get("PROBE_TEXT_EVAL", "1") != "0"

TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
VARIANTS = {
    "true": TASK,
    "paraphrase": "grasp the black bowl in the middle and set it down on the plate",
    "unrelated": "open the top drawer of the cabinet",
}


def main():
    from run_instruction_sweep import build_cfg
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
    from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

    class _A:
        suite, task_id = "libero_spatial", 0

    cfg = build_cfg(_A())

    dtype = torch.bfloat16
    _d, text_c, _v, tok_c = _resolve_configs(
        model_id=cfg.model.model_id,
        tokenizer_model_id=cfg.model.tokenizer_model_id,
        redirect_common_files=bool(cfg.model.redirect_common_files),
    )
    text_c.download_if_necessary()
    tok_c.download_if_necessary()
    enc = _load_registered_model(text_c.path, "wan_video_text_encoder", torch_dtype=dtype, device=DEVICE)
    # _load_registered_model does not call .eval(); WanTextEncoder has Dropout(0.1).
    # Toggle via EVAL_MODE to demonstrate the difference.
    if EVAL_MODE:
        enc.eval().requires_grad_(False)
    print(f"[enc] training mode = {enc.training} (EVAL_MODE={EVAL_MODE})")
    tok = HuggingfaceTokenizer(name=tok_c.path, seq_len=int(cfg.model.tokenizer_max_len), clean="whitespace")

    def encode(text):
        with torch.no_grad():
            ids, mask = tok([DEFAULT_PROMPT.format(task=text)], return_mask=True, add_special_tokens=True)
            ids = ids.to(DEVICE)
            mask = mask.to(DEVICE, dtype=torch.bool)
            emb = enc(ids, mask)
            v = int(mask.gt(0).sum(dim=1)[0])
            emb[0, v:] = 0
            return emb.detach().float().cpu()[0]

    reps = {k: [encode(t) for _ in range(R)] for k, t in VARIANTS.items()}

    print(f"\n=== WITHIN-prompt spread ({R} repeats, batch size 1 each) ===")
    within = {}
    for k, rs in reps.items():
        dmax = max(float((rs[i] - rs[0]).abs().max()) for i in range(1, R))
        dl2 = max(float(torch.linalg.norm(rs[i] - rs[0])) for i in range(1, R))
        within[k] = dl2
        ident = all(torch.equal(rs[i], rs[0]) for i in range(1, R))
        print(f"  {k:12s} max|d| = {dmax:.6e}   ||d||_2 = {dl2:.6e}   bitwise identical = {ident}")

    print("\n=== BETWEEN-prompt distance (repeat 0 of each) ===")
    keys = list(VARIANTS)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = float(torch.linalg.norm(reps[a][0] - reps[b][0]))
            print(f"  {a:12s} vs {b:12s} ||d||_2 = {d:.6e}")

    print("\n=== embedding-level signal-to-noise ===")
    worst_within = max(within.values())
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            d = float(torch.linalg.norm(reps[a][0] - reps[b][0]))
            snr = d / worst_within if worst_within > 0 else float("inf")
            print(f"  {a:12s} vs {b:12s}: {snr:8.1f}x  (worst within-prompt spread {worst_within:.3e})")

    scale = float(torch.linalg.norm(reps["true"][0]))
    print(f"\n||context_true||_2 = {scale:.3f}")
    if worst_within == 0.0:
        print("VERDICT: text encoder IS deterministic -> the sweep's floor comes from elsewhere.")
    else:
        print(f"VERDICT: text encoder is NONdeterministic ({worst_within:.3e} L2, "
              f"{100*worst_within/scale:.3f}% of ||context||).")
        print("         -> cache each instruction's embedding once and reuse it; treat the")
        print("            within-prompt spread propagated through the policy as the floor.")


if __name__ == "__main__":
    main()
