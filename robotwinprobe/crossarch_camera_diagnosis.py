"""Read-only camera counterfactuals for the failed grasp-onset proxy checks.

Mixed camera inputs isolate routes of error and are not physical rollouts.
The original proxy, gate, protocol, and training status are not changed.
"""
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from crossarch_common import DEFAULT, CAMERAS, WIRE, dump
from crossarch_backend import Backend
from crossarch_proxy import images, arrays
from render_geometry import patterns


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def cosine(a, b):
    a, b = a.flatten().astype(float), b.flatten().astype(float)
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def shapley(values):
    result = {}
    for camera in range(3):
        bit = 1 << camera
        result[CAMERAS[camera]] = sum(
            math.factorial(mask.bit_count()) * math.factorial(2 - mask.bit_count()) / 6
            * (values[mask] - values[mask | bit])
            for mask in range(8) if not mask & bit)
    return result


def diagnose(root=DEFAULT):
    dest = root / 'audit/camera_diagnosis_20261026'
    dest.mkdir(exist_ok=True)
    audit = json.loads((root / 'audit/pi_proxy.json').read_text())
    records = list(dict.fromkeys(r['record'] for r in audit['rows'] if r['stage'] == 'grasp'))
    backend = Backend('pi', root)
    rows, contributions, pixels = [], [], []
    examples = {}
    for path in records:
        with np.load(path) as z:
            state, prompt = z['state14'].copy(), str(z['prompt'])
            clean = {w: z[n + '_clean'].copy() for n, w in zip(CAMERAS, WIRE)}
            mask = {w: z[n + '_mask'].copy().astype(bool) for n, w in zip(CAMERAS, WIRE)}
            for pattern in ('random', 'checker'):
                true = {w: z[n + '_' + pattern].copy() for n, w in zip(CAMERAS, WIRE)}
                q = torch.tensor(patterns()[pattern], device='cuda').permute(2, 0, 1)[None].float() / 255
                with torch.no_grad():
                    proxy = dict(zip(WIRE, arrays(images(q, path))))
                for n, w in zip(CAMERAS, WIRE):
                    error = (proxy[w].astype(float) - true[w]) / 255
                    real = (true[w].astype(float) - clean[w]) / 255
                    pixels.append(dict(record=path, pattern=pattern, camera=n, visible_pixels=int(mask[w].sum()),
                        proxy_render_rmse=float(np.sqrt(np.mean(error ** 2))),
                        actual_patch_clean_rmse=float(np.sqrt(np.mean(real ** 2))),
                        mask_rmse=float(np.sqrt(np.mean(error[mask[w]] ** 2))) if mask[w].any() else None))
                if pattern == 'checker':
                    for label, views in [('clean', clean), ('actual', true), ('proxy', proxy)]:
                        for n, w in zip(CAMERAS, WIRE):
                            examples[Path(path).stem + '_' + label + '_' + n] = views[w]
                for seed in (42, 43, 44):
                    full = seed == 42

                    def response(raw):
                        with torch.no_grad():
                            v, a = backend.response(backend.raw_prepared(state, prompt, raw), seed, full=full)
                        return v[..., :14].float().cpu().numpy(), a[..., :14].float().cpu().numpy() if full else None

                    vc, ac = response(clean)
                    repaired = {}
                    for selection in range(8):
                        raw = {w: (true if selection & (1 << i) else proxy)[w] for i, w in enumerate(WIRE)}
                        repaired[selection] = response(raw)
                    vr, ar = repaired[7]
                    real_delta = vr - vc
                    errors = {k: mse(v, vr) for k, (v, _) in repaired.items()}
                    baseline_error = errors[0]
                    attribution = shapley(errors)
                    contributions.append(dict(record=path, pattern=pattern, noise_seed=seed,
                        real_patch_L0=mse(vr, vc), baseline_proxy_error=baseline_error,
                        error_shapley_fraction={k: v / max(baseline_error, 1e-20) for k, v in attribution.items()},
                        full_action_proxy_error=mse(repaired[0][1], ar) if full else None,
                        full_action_actual_patch_effect=mse(ar, ac) if full else None))
                    for selection, (v, a) in repaired.items():
                        row = dict(record=path, pattern=pattern, noise_seed=seed, family='repair_proxy',
                            actual_camera_mask=selection, cosine_to_real_delta=cosine(v - vc, real_delta),
                            residual_error_fraction=errors[selection] / max(baseline_error, 1e-20))
                        if full:
                            row['action_error_fraction'] = mse(a, ar) / max(mse(repaired[0][1], ar), 1e-20)
                        rows.append(row)
                    # Independent route check: actual patched cameras versus clean cameras.
                    for selection in range(8):
                        if selection == 0:
                            v, a = vc, ac
                        elif selection == 7:
                            v, a = vr, ar
                        else:
                            raw = {w: (true if selection & (1 << i) else clean)[w] for i, w in enumerate(WIRE)}
                            v, a = response(raw)
                        row = dict(record=path, pattern=pattern, noise_seed=seed, family='actual_patch_route',
                            actual_camera_mask=selection, cosine_to_real_delta=cosine(v - vc, real_delta),
                            L0_fraction_of_all_actual=mse(v, vc) / max(mse(vr, vc), 1e-20))
                        if full:
                            row['action_effect_fraction'] = mse(a, ac) / max(mse(ar, ac), 1e-20)
                        rows.append(row)
                    # Separate head-camera patch pixels from its surrounding scene.
                    for label, repair_mask in [('head_patch_pixels', mask[WIRE[0]]),
                                                ('head_outside_pixels', ~mask[WIRE[0]])]:
                        raw = dict(proxy)
                        raw[WIRE[0]] = np.where(repair_mask[..., None], true[WIRE[0]], proxy[WIRE[0]])
                        v, a = response(raw)
                        row = dict(record=path, pattern=pattern, noise_seed=seed, family=label,
                            cosine_to_real_delta=cosine(v - vc, real_delta),
                            residual_error_fraction=mse(v, vr) / max(baseline_error, 1e-20))
                        if full:
                            row['action_error_fraction'] = mse(a, ar) / max(mse(repaired[0][1], ar), 1e-20)
                        rows.append(row)
                dump(dest / 'responses.json', rows)
                dump(dest / 'contributions.json', contributions)
                dump(dest / 'pixels.json', pixels)
                print('completed', Path(path).name, pattern, flush=True)

    summaries = []
    for family in ('repair_proxy', 'actual_patch_route', 'head_patch_pixels', 'head_outside_pixels'):
        for selection in (range(8) if family in ('repair_proxy', 'actual_patch_route') else [None]):
            selected = [r for r in rows if r['family'] == family and r.get('actual_camera_mask') == selection]
            summary = dict(family=family, actual_camera_mask=selection,
                actual_cameras=[CAMERAS[i] for i in range(3) if selection is not None and selection & (1 << i)],
                count=len(selected))
            for metric in ('cosine_to_real_delta', 'residual_error_fraction', 'action_error_fraction',
                           'L0_fraction_of_all_actual', 'action_effect_fraction'):
                values = [r[metric] for r in selected if metric in r]
                if values:
                    summary[metric + '_median'] = float(np.median(values))
                    summary[metric + '_mean'] = float(np.mean(values))
            summaries.append(summary)
    summary = dict(model='pi', unique_grasp_onset_states=2, render_conditions=4, patterns=2,
        noise_seeds=[42, 43, 44], full_flow_action_noise_seed=42,
        meaning='Repair: replace proxy views with actual-render views. Route: replace clean views with actual-patch views. Camera mixtures are offline counterfactuals, not physically possible simultaneous views or task-ASR evidence.',
        shapley_error_fraction_mean={n: float(np.mean([r['error_shapley_fraction'][n] for r in contributions])) for n in CAMERAS},
        full_action_error_to_actual_patch_energy_median=float(np.median([
            r['full_action_proxy_error'] / max(r['full_action_actual_patch_effect'], 1e-20)
            for r in contributions if r['full_action_proxy_error'] is not None])),
        rows=summaries)
    dump(dest / 'summary.json', summary)
    np.savez_compressed(dest / 'examples.npz', **examples)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    diagnose()
