"""Recompute color/edge/pixel controls and paired KV/velocity summaries."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pixels-only', action='store_true')
    args = ap.parse_args()
    dest = Path(__file__).resolve().parent / 'out/color_checker_probe'
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    patterns, conditions = pool['patterns'].tolist(), pool['conditions'].tolist()
    palette = pool['palette'].astype(float)
    yy, xx = np.mgrid[:224, :224]
    inner = pool['inner_mask'].astype(bool)
    boundary = pool['mask'].astype(bool) & ~inner
    # Diagnose the existing UV/export convention without refitting on checks.
    Htex = pool['image256_to_texture_pixel_H']
    Hexport = pool['source224_to_export128_H']
    scale = 256 / 224
    A = np.array([[scale, 0, (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
    flip = np.array([[1, 0, 0], [0, -1, 127], [0, 0, 1]])
    Hback = np.linalg.inv(Hexport) @ flip @ Htex @ A
    coordinates = np.stack([xx, yy, np.ones_like(xx)], -1) @ Hback.T
    source_xy = coordinates[..., :2] / coordinates[..., 2:]
    delta = source_xy - np.stack([xx, yy], -1)
    support = inner.any(0)
    coordinate = {'mean_dx_source_px': float(delta[support, 0].mean()),
                  'mean_dy_source_px': float(delta[support, 1].mean()),
                  'p05_dx': float(np.quantile(delta[support, 0], .05)),
                  'p95_dx': float(np.quantile(delta[support, 0], .95)),
                  'p05_dy': float(np.quantile(delta[support, 1], .05)),
                  'p95_dy': float(np.quantile(delta[support, 1], .95)),
                  'notes': 'Inferred UV composition of frozen ramp homography and original exportH, '
                           'including pixel-center224-to256 mapping; not a new fit or independent exact UV measurement.'}
    records = []
    for name in patterns:
        reference = pool[name + '__digital'].astype(float)
        labels = np.argmin(((pool[name + '__pattern'][..., None, :].astype(float) - palette) ** 2).sum(-1), -1)
        # Cell cores exclude one source pixel next to a grid line and the patch boundary.
        s = int(name.replace('cell', ''))
        core = (xx % s >= 1) & (xx % s < s - 1) & (yy % s >= 1) & (yy % s < s - 1)
        for c in conditions:
            images = pool[name + '__' + c].astype(float)
            errors = images - reference
            for k, step in enumerate(pool['steps']):
                m, i, edge = pool['mask'][k].astype(bool), inner[k], boundary[k]
                colors = images[k][i]
                pd = ((colors[:, None] - palette) ** 2).mean(-1)
                label = pd.argmin(-1)
                ci = i & core
                records.append({'pattern': name, 'step': int(step), 'condition': c,
                                'mask_MAE_u8': float(np.abs(errors[k][m]).mean()),
                                'interior_MAE_u8': float(np.abs(errors[k][i]).mean()),
                                'interior_RMSE_u8': float(np.sqrt(np.square(errors[k][i]).mean())),
                                'boundary_MAE_u8': float(np.abs(errors[k][edge]).mean()),
                                'nearest_palette_label_agreement': float((label == labels[i]).mean()),
                                'fraction_more_than8u8_RMS_from_any_palette': float((np.sqrt(pd.min(-1)) > 8).mean()),
                                'flat_cell_core_pixels': int(ci.sum()),
                                'flat_cell_core_MAE_u8': float(np.abs(errors[k][ci]).mean()) if ci.any() else None})
    pixel_summary = {name: {c: {key: float(np.mean([r[key] for r in records
                      if r['pattern'] == name and r['condition'] == c and r[key] is not None]))
                      if any(r[key] is not None for r in records if r['pattern'] == name and r['condition'] == c) else None
                      for key in ['mask_MAE_u8', 'interior_MAE_u8', 'interior_RMSE_u8', 'boundary_MAE_u8',
                                  'nearest_palette_label_agreement', 'fraction_more_than8u8_RMS_from_any_palette',
                                  'flat_cell_core_pixels', 'flat_cell_core_MAE_u8']}
                      for c in conditions} for name in patterns}
    (dest / 'pixel_metrics.json').write_text(json.dumps({'records': records, 'summary': pixel_summary,
        'coordinate_diagnostic': coordinate, 'notes': 'Palette labels are descriptive, mixed pixels remain ambiguous. '
        'Distances to pure palettes for lit images include brightness change; interpret mixing in unlit rows.'},
        indent=2, allow_nan=False) + '\n')
    alignment = np.load(dest / 'alignment_inputs.npz', allow_pickle=False)
    align_summary = {}
    for name in patterns:
        align_summary[name] = {}
        for c, src in [('minimal_nearest_export', pool), ('calibrated_minimal', alignment),
                       ('render', pool), ('calibrated_render', alignment)]:
            error = src[name + '__' + c].astype(float) - pool[name + '__digital']
            values = []
            for k in range(len(inner)):
                im = src[name + '__' + c][k][inner[k]].astype(float)
                truth = pool[name + '__digital'][k][inner[k]].astype(float)
                label = lambda rgb: ((rgb[:, None] - palette) ** 2).sum(-1).argmin(-1)
                values.append((float(np.abs(error[k][inner[k]]).mean()),
                               float((label(im) == label(truth)).mean())))
            align_summary[name][c] = {'interior_MAE_u8': float(np.mean([v[0] for v in values])),
                                     'palette_agreement': float(np.mean([v[1] for v in values]))}
    (dest / 'alignment_summary.json').write_text(json.dumps(align_summary, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    m = pool['mask'][0].astype(bool)
    by, bx = np.where(m)
    sy = slice(max(0, by.min() - 4), min(224, by.max() + 5))
    sx = slice(max(0, bx.min() - 4), min(224, bx.max() + 5))
    show = ['digital', 'export_roundtrip_linear', 'minimal_linear_export',
            'minimal_nearest_export', 'unlit_default', 'render']
    titles = ['Digital224', 'OpenCV export roundtrip', 'Minimal NN, linear export',
              'Minimal NN, nearest export', 'Unlit production rendering', 'Lit production rendering']
    fig, axes = plt.subplots(4, len(show), figsize=(16, 10), constrained_layout=True)
    for row, name in enumerate(patterns):
        for col, (c, title) in enumerate(zip(show, titles)):
            axes[row, col].imshow(pool[name + '__' + c][0, sy, sx], interpolation='nearest')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(title, fontsize=10)
        axes[row, 0].set_ylabel(name + ' source px', fontsize=11)
    fig.suptitle('E23 five-color checks: same state, geometry and mask; source pixels enlarged with nearest display')
    fig.savefig(dest / 'fig_checker_inputs.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.array([2, 4, 8, 16])
    for c in ['digital_resize256_224', 'export_roundtrip_linear', 'minimal_linear_export',
              'minimal_nearest_export', 'unlit_default', 'render']:
        axes[0].plot(x, [pixel_summary[n][c]['interior_MAE_u8'] for n in patterns], 'o-', label=c)
        axes[1].plot(x, [pixel_summary[n][c]['nearest_palette_label_agreement'] for n in patterns], 'o-')
    axes[0].set_ylabel('Interior mean absolute RGB error / digital, u8')
    axes[1].set_ylabel('Interior nearest-color label agreement')
    axes[1].set_ylim(0, 1.03)
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xticks(x); ax.set_xticklabels(x)
        ax.set_xlabel('Cell width in original digital224 pixels'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('E23: 16 correlated states, patch interior excludes 4-pixel outer boundary')
    fig.savefig(dest / 'fig_checker_pixels.png', dpi=160); plt.close(fig)
    show_align = [('digital', pool), ('minimal_nearest_export', pool), ('calibrated_minimal', alignment),
                  ('render', pool), ('calibrated_render', alignment)]
    fig, axes = plt.subplots(4, 5, figsize=(14, 10), constrained_layout=True)
    for row, name in enumerate(patterns):
        for col, (c, src) in enumerate(show_align):
            axes[row, col].imshow(src[name + '__' + c][0, sy, sx], interpolation='nearest')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(c, fontsize=10)
        axes[row, 0].set_ylabel(name + ' source px')
    fig.suptitle('E23 alignment control: only export geometry changes; ramp calibration predates checkerboards')
    fig.savefig(dest / 'fig_checker_alignment.png', dpi=160); plt.close(fig)
    if args.pixels_only:
        print('[pixels]', json.dumps({'coordinate': coordinate, 'summary': pixel_summary})); return
    metadata = json.loads((dest / 'metrics.json').read_text())
    assert metadata['complete'] and len(metadata['records']) == 16 * 4 * 9
    model_summary = {}
    for name in patterns:
        model_summary[name] = {}
        for c in conditions:
            rs = [r for r in metadata['records'] if r['pattern'] == name and r['condition'] == c]
            model_summary[name][c] = {
                'KV_direct_relative_digital': float(np.mean([r['vs_digital']['all18_KV']['relative'] for r in rs])),
                'lastV_direct_relative_digital': float(np.mean([r['vs_digital']['layers'][-1]['V']['relative'] for r in rs])),
                'L0_clean': float(np.mean([r['vs_clean']['L0'] for r in rs])),
                'L0_direct_digital': float(np.mean([r['vs_digital']['L0'] for r in rs])),
                'L0_real7_clean': float(np.mean([r['vs_clean']['L0_real7'] for r in rs])),
                'L0_exec7_clean': float(np.mean([r['vs_clean']['L0_exec7'] for r in rs]))}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for c in ['digital', 'export_roundtrip_linear', 'minimal_linear_export',
              'minimal_nearest_export', 'unlit_default', 'render']:
        axes[0].plot(x, [model_summary[n][c]['KV_direct_relative_digital'] for n in patterns], 'o-', label=c)
        axes[1].plot(x, [model_summary[n][c]['L0_clean'] for n in patterns], 'o-')
    axes[0].set_ylabel('Direct all18-layer K/V distance / digital KV norm')
    axes[1].set_ylabel('Early t=1 L0 / clean, three paired noises')
    for ax in axes:
        ax.set_xscale('log', base=2); ax.set_xticks(x); ax.set_xticklabels(x)
        ax.set_xlabel('Cell width in source224 pixels'); ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=8)
    fig.suptitle('E23: backbone direct differences and early velocity separately; no learned attack or ASR')
    fig.savefig(dest / 'fig_checker_model.png', dpi=160); plt.close(fig)
    (dest / 'checker_summary.json').write_text(json.dumps({'complete': True, 'pixels': pixel_summary,
        'model': model_summary, 'coordinate': coordinate, 'alignment': align_summary}, indent=2, allow_nan=False) + '\n')
    lines = ['# E23 five-color checkerboard rendering diagnostic', '',
             '16 fixed E15 states x 4 source cell widths x 9 conditions; same mask and clean background. '
             'No training, integration or closed-loop evaluation. Three shared t=1 action noises.', '',
             '| Cell width, source px | Condition | Interior RGB MAE, u8 | Direct KV / digital | Early L0 / clean |',
             '|---|---|---:|---:|---:|']
    for name in patterns:
        for c in ['digital', 'digital_resize256_224', 'export_roundtrip_linear', 'export_roundtrip_nearest',
                  'minimal_linear_export', 'minimal_nearest_export', 'unlit_nomipmap', 'unlit_default', 'render']:
            lines.append('| {} | {} | {:.4f} | {:.5f} | {:.6f} |'.format(name, c,
                pixel_summary[name][c]['interior_MAE_u8'], model_summary[name][c]['KV_direct_relative_digital'],
                model_summary[name][c]['L0_clean']))
    lines += ['', 'Frozen UV/export composition implies mean source-coordinate offset dx={:.5f}, dy={:.5f} '
              'pixels over the union of interiors. This is inferred from a prior ramp fit, not exact new UV ground truth.'.format(
                  coordinate['mean_dx_source_px'], coordinate['mean_dy_source_px']), '',
              '| Cell width | Old minimal NN interior MAE | Calibrated minimal interior MAE | Old production MAE | Calibrated production MAE |',
              '|---|---:|---:|---:|---:|']
    for name in patterns:
        a = align_summary[name]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(name,
            a['minimal_nearest_export']['interior_MAE_u8'], a['calibrated_minimal']['interior_MAE_u8'],
            a['render']['interior_MAE_u8'], a['calibrated_render']['interior_MAE_u8']))
    full_dest = dest.parent / 'color_checker_full_alignment'
    full_inputs = np.load(full_dest / 'inputs.npz', allow_pickle=False)
    full_metrics = json.loads((full_dest / 'metrics.json').read_text())
    assert full_metrics['complete'] and len(full_metrics['records']) == 80
    full_summary = {}
    for c in full_inputs['conditions']:
        rs = [r for r in full_metrics['records'] if r['condition'] == c]
        diff = full_inputs['full__' + c].astype(float) - full_inputs['full__digital']
        full_summary[str(c)] = {'L0_clean': float(np.mean([r['vs_clean']['L0'] for r in rs])),
                               'KV_relative_digital': float(np.mean([r['vs_digital']['all18_KV']['relative'] for r in rs])),
                               'lastV_relative_digital': float(np.mean([r['vs_digital']['layers'][-1]['V']['relative'] for r in rs])),
                               'L0_direct_digital': float(np.mean([r['vs_digital']['L0'] for r in rs])),
                               'interior_MAE_u8': float(np.mean([np.abs(d[i]).mean() for d, i in zip(diff, full_inputs['inner_mask'])])),
                               'mask_MAE_u8': float(np.mean([np.abs(d[m]).mean() for d, m in zip(diff, full_inputs['mask'])]))}
    (dest / 'full_alignment_summary.json').write_text(json.dumps(full_summary, indent=2) + '\n')
    lines += ['', '## Same calibration applied to historical full', '',
              '| Condition | Interior MAE, u8 | Direct KV / digital | Early L0 / clean |',
              '|---|---:|---:|---:|']
    for c, r in full_summary.items():
        lines.append('| {} | {:.4f} | {:.5f} | {:.6f} |'.format(c, r['interior_MAE_u8'],
            r['KV_relative_digital'], r['L0_clean']))
    lines += ['', 'Calibrated minimal input equals digital in the interior. About5.16% of mask pixels '
              'still differ, all within its outermost one-pixel contour. The early learned response '
              'remains weak, so correcting UV alignment alone does not recover this historical attack. '
              'The contour includes occlusion boundaries and the shared composite mask need not equal '
              'true minimal-render coverage; unlit scene pixels can differ there. This is a controlled '
              'input sensitivity diagnostic, not a general claim about all physical boundaries. '
              'Production rendering retains filtering, resize and light changes. No rollout was run.']
    full_show = full_inputs['conditions'].tolist()
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), constrained_layout=True)
    for col, c in enumerate(full_show):
        im = full_inputs['full__' + c][0]
        axes[0, col].imshow(im[sy, sx], interpolation='nearest'); axes[0, col].set_title(c, fontsize=10)
        error = np.abs(im.astype(float) - full_inputs['full__digital'][0]).mean(-1)
        axes[1, col].imshow(error[sy, sx], cmap='magma', vmin=0, vmax=255, interpolation='nearest')
        for row in [0, 1]: axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    axes[1, 0].set_ylabel('Mean RGB error / digital, 0-255')
    fig.suptitle('E23 historical full: calibrated minimal interior is exact; a thin mask-contour difference remains')
    fig.savefig(dest / 'fig_full_alignment.png', dpi=160); plt.close(fig)
    lines += ['',
              'Colored grids are calibration probes. Their L0 need not mimic an optimized full patch; '
              'large pixel or KV changes do not establish a learned attack mechanism or success rate.', '',
              '![Input crops](fig_checker_inputs.png)', '', '![Pixels](fig_checker_pixels.png)', '',
              '![Backbone and head](fig_checker_model.png)', '', '![Fixed-ramp alignment control](fig_checker_alignment.png)', '',
              '![Historical full alignment control](fig_full_alignment.png)', '']
    (dest / 'results.md').write_text('\n'.join(lines))
    print('[done]', dest)


if __name__ == '__main__':
    main()
