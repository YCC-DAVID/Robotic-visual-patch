"""E21: standalone pixel, KV and paired early-flow figures and tables."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='white_patch_probe')
    args = ap.parse_args()
    dest = Path(__file__).resolve().parent / 'out' / args.out
    pool = np.load(dest / 'inputs.npz', allow_pickle=False)
    capture = json.loads((dest / 'capture.json').read_text())
    metrics = json.loads((dest / 'metrics.json').read_text())
    assert metrics['complete'] and capture['complete']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    labels = {'digital_white': 'Digital white', 'render_white': 'Render white (lit)',
              'render_white_unlit': 'Render white (unlit)', 'digital_matched_rgb': 'Digital matched gray',
              'render_white_raw': 'Full raw render', 'digital_white_interior': 'Digital white, interior',
              'render_white_interior': 'Render lit, interior',
              'render_white_unlit_interior': 'Render unlit, interior'}
    conditions = metrics['conditions']
    summary = {'frames': metrics['frames'], 'conditions': {}, 'pairs': [], 'pixel_frames': []}
    for condition in conditions:
        rows = [r for r in metrics['records'] if r['condition'] == condition]
        item = {'reference': rows[0]['reference']}
        for key in ['vs_clean', 'vs_digital']:
            item[key] = {
                'feature_relative': float(np.mean([r[key]['feature']['relative_to_reference'] for r in rows])),
                'all18_KV_relative': float(np.mean([r[key]['all18_KV']['relative_to_reference'] for r in rows])),
                'all18_KV_delta_rms': float(np.mean([r[key]['all18_KV']['delta_rms'] for r in rows])),
                'last_K_relative': float(np.mean([r[key]['kv'][-1]['K']['relative_to_reference'] for r in rows])),
                'last_V_relative': float(np.mean([r[key]['kv'][-1]['V']['relative_to_reference'] for r in rows])),
                **{label: float(np.mean([r[key]['velocity'][label] for r in rows]))
                   for label in ['L0', 'L0_real7', 'L0_exec7']}}
        summary['conditions'][condition] = item
    pair_names = list(dict.fromkeys((r['target'], r['reference']) for r in metrics['pairs']))
    for ca, cb in pair_names:
        rows = [r for r in metrics['pairs'] if r['target'] == ca and r['reference'] == cb]
        summary['pairs'].append({'target': ca, 'reference': cb,
            'feature_relative': float(np.mean([r['feature']['relative_to_reference'] for r in rows])),
            'all18_KV_relative': float(np.mean([r['all18_KV']['relative_to_reference'] for r in rows])),
            'last_K_relative': float(np.mean([r['kv'][-1]['K']['relative_to_reference'] for r in rows])),
            'last_V_relative': float(np.mean([r['kv'][-1]['V']['relative_to_reference'] for r in rows])),
            **{label: float(np.mean([r['velocity'][label] for r in rows]))
               for label in ['L0', 'L0_real7', 'L0_exec7']}})
    for k, step in enumerate(pool['steps']):
        m, inner = pool['mask'][k], pool['inner_mask'][k]
        boundary = m & ~inner
        item = dict(capture['frame_metrics'][k])
        item['boundary_px'] = int(boundary.sum())
        item['boundary_fraction'] = float(boundary.sum() / m.sum())
        for condition in ['render_white', 'render_white_unlit', 'digital_matched_rgb']:
            delta = np.abs(pool[condition][k].astype(float) - pool['digital_white'][k].astype(float))
            item[condition] = {region + '_MAE_u8': float(delta[mask].mean())
                              for region, mask in [('full', m), ('interior', inner), ('boundary', boundary)]}
            item[condition]['fraction_different_full_mask'] = float(np.any(delta > 0, axis=-1)[m].mean())
        delta = np.abs(pool['render_white'][k].astype(float) - pool['digital_matched_rgb'][k].astype(float))
        item['lit_vs_matched_interior_MAE_u8'] = float(delta[inner].mean())
        summary['pixel_frames'].append(item)
    keys = ['segmentation_IoU', 'boundary_fraction', 'lit_MAE_all_mask_u8', 'lit_MAE_interior_u8',
            'unlit_MAE_all_mask_u8', 'unlit_MAE_interior_u8', 'raw_outside_mask_MAE_u8',
            'lit_vs_matched_interior_MAE_u8']
    summary['pixel_means'] = {key: float(np.mean([r[key] for r in summary['pixel_frames']])) for key in keys}
    summary['pixel_means']['lit_interior_rgb_mean'] = np.mean(
        [r['lit_interior_rgb_mean'] for r in summary['pixel_frames']], axis=0).tolist()
    summary['pixel_means']['unlit_boundary_MAE_u8'] = float(np.mean(
        [r['render_white_unlit']['boundary_MAE_u8'] for r in summary['pixel_frames']]))
    summary['pixel_means']['unlit_fraction_different_full_mask'] = float(np.mean(
        [r['render_white_unlit']['fraction_different_full_mask'] for r in summary['pixel_frames']]))
    summary['notes'] = ('Each KV ratio is computed within a state then averaged over 16 correlated states. '
        'L0 is paired first-flow squared velocity difference averaged over states and three common noises. '
        'Full and eroded-interior comparisons have different masks/references. '
        'The segmentation footprint is a resized binary diagnostic, not an exact fractional coverage model. '
        'No trained-white attack or rollout. Uniform-white has no interior texture spectrum to test.')
    (dest / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')

    # Pixel views use the same crop and fixed error scales in each column.
    indices = [0, 10, 15]
    fig, axes = plt.subplots(3, 6, figsize=(16, 9), constrained_layout=True)
    titles = ['Digital white', 'Render lit', 'Render unlit', 'Lit |RGB - white|',
              'Unlit |RGB - white|', 'Mask / geometry footprint']
    for row, k in enumerate(indices):
        m, inner = pool['mask'][k], pool['inner_mask'][k]
        yy, xx = np.where(m)
        ys = slice(max(int(yy.min()) - 8, 0), min(int(yy.max()) + 9, 224))
        xs = slice(max(int(xx.min()) - 8, 0), min(int(xx.max()) + 9, 224))
        for col, c in enumerate(['digital_white', 'render_white', 'render_white_unlit']):
            axes[row, col].imshow(pool[c][k][ys, xs], interpolation='nearest')
        for col, c in [(3, 'render_white'), (4, 'render_white_unlit')]:
            error = np.abs(pool[c][k].astype(float) - pool['digital_white'][k].astype(float)).mean(-1)
            h = axes[row, col].imshow(error[ys, xs], vmin=0, vmax=80 if col == 3 else 255,
                                      cmap='magma', interpolation='nearest')
            fig.colorbar(h, ax=axes[row, col], fraction=.04, pad=.02)
        geometry = pool['coverage224'][k] >= 128
        overlay = np.zeros((224, 224, 3))
        overlay[m & geometry] = [.7, .7, .7]
        overlay[m & ~geometry] = [.2, .4, 1.]
        overlay[~m & geometry] = [.1, .9, .3]
        axes[row, 5].imshow(overlay[ys, xs], interpolation='nearest')
        axes[row, 5].contour(inner[ys, xs], levels=[.5], colors=['#ff9933'], linewidths=1)
        axes[row, 0].set_ylabel('step ' + str(int(pool['steps'][k])))
        for col in range(6):
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            if row == 0: axes[row, col].set_title(titles[col])
    fig.suptitle('E21: same full mask and clean background; orange = eroded interior\n'
                 'Footprint: gray = overlap, blue = digital mask only, green = geometry only')
    fig.savefig(dest / 'fig_white_pixels.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), constrained_layout=True)
    colors = {'render_white': '#bc4142', 'render_white_unlit': '#2878a5',
              'digital_matched_rgb': '#e59b32', 'render_white_raw': '#745ba9',
              'render_white_interior': '#bc4142', 'render_white_unlit_interior': '#2878a5'}
    for ax, label in zip(axes[:2], ['K', 'V']):
        for c in ['render_white', 'render_white_unlit', 'digital_matched_rgb', 'render_white_raw']:
            rows = [r for r in metrics['records'] if r['condition'] == c]
            y = [np.mean([r['vs_digital']['kv'][l][label]['relative_to_reference'] for r in rows])
                 for l in range(18)]
            ax.plot(range(18), y, label=labels[c], color=colors[c], linewidth=2)
        ax.set_xlabel('Backbone layer (0-based)')
        ax.set_ylabel('||' + label + ' - ' + label + '(digital white)|| / ||' + label + '(digital white)||')
        ax.set_title('Full-mask ' + label + ': direct paired difference')
        ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    for c in ['render_white_interior', 'render_white_unlit_interior']:
        rows = [r for r in metrics['records'] if r['condition'] == c]
        y = [np.mean([r['vs_digital']['kv'][l]['V']['relative_to_reference'] for r in rows])
             for l in range(18)]
        axes[2].plot(range(18), y, color=colors[c], label=labels[c], linewidth=2)
    axes[2].set_title('Interior mask V: separate digital reference')
    axes[2].set_xlabel('Backbone layer (0-based)')
    axes[2].set_ylabel('Relative difference to digital white, interior')
    axes[2].set_ylim(bottom=0); axes[2].grid(alpha=.2)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle('E21: backbone only; 16-state mean, valid prefix tokens; no action head in KV distances')
    fig.savefig(dest / 'fig_white_kv.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    groups = [conditions[:5], conditions[5:]]
    short_labels = {'digital_white': 'Digital\nwhite', 'render_white': 'Render\nlit',
                    'render_white_unlit': 'Render\nunlit', 'digital_matched_rgb': 'Digital\nmatched gray',
                    'render_white_raw': 'Raw\nrender', 'digital_white_interior': 'Digital white\ninterior',
                    'render_white_interior': 'Render lit\ninterior',
                    'render_white_unlit_interior': 'Render unlit\ninterior'}
    for col, group in enumerate(groups):
        x = np.arange(len(group))
        for row, reference in enumerate(['vs_clean', 'vs_digital']):
            ax = axes[row, col]
            for shift, quantity, label in [(-.25, 'L0', 'all 10x32'), (0, 'L0_real7', 'real 10x7'),
                                           (.25, 'L0_exec7', 'first 5x7')]:
                y = [summary['conditions'][c][reference][quantity] for c in group]
                ax.bar(x + shift, y, width=.23, label=label)
            ax.set_xticks(x); ax.set_xticklabels([short_labels[c] for c in group], fontsize=9)
            ax.set_ylabel('Mean squared first-flow velocity difference')
            ax.set_title(('Full mask' if col == 0 else 'Eroded interior mask') + ': ' +
                         ('vs clean' if row == 0 else 'vs corresponding digital white'))
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('E21: fresh common action noise for every input; t=1; no integration or rollout')
    fig.savefig(dest / 'fig_white_flow.png', dpi=160)
    plt.close(fig)

    lines = ['# E21 constant-white diagnostic', '',
        '16 states, 8 conditions, 3 paired action noises. KV distances are backbone-only. '
        'L0 is first-flow velocity squared difference, not final actions or attack success.', '',
        'Full-mask inputs share the exact E15 mask and clean background except the separate raw-render anchor. '
        'Interior inputs share a 4-pixel-eroded mask and use their own digital reference.', '',
        '| Condition | all18 KV / clean | all18 KV / digital | last V / digital | L0 / clean | L0 / digital |',
        '|---|---:|---:|---:|---:|---:|']
    for c in conditions:
        a, b = summary['conditions'][c]['vs_clean'], summary['conditions'][c]['vs_digital']
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            c, a['all18_KV_relative'], b['all18_KV_relative'], b['last_V_relative'], a['L0'], b['L0']))
    lines += ['', '## Direct paired controls', '',
              '| Target | Reference | all18 KV / reference | last V / reference | L0 | L0 real7 | L0 first5x7 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['pairs']:
        lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            r['target'], r['reference'], r['all18_KV_relative'], r['last_V_relative'],
            r['L0'], r['L0_real7'], r['L0_exec7']))
    lines += ['', '## Pixel controls', '', '```json', json.dumps(summary['pixel_means'], indent=2), '```', '',
              'Unlit interior input, KV and velocity are identical to digital white interior for all states. '
              'Full-mask boundaries retain AA/resize/coverage differences. The eroded mask is a diagnostic '
              'and removes about 39% of full-mask pixels, so raw full/interior L0 values have different support.', '',
              'One white level does not calibrate the photometric response of gray or colored textures, '
              'and does not identify the high-frequency sampling mechanism of trained full patches.', '',
              '![Pixels](fig_white_pixels.png)', '![Backbone](fig_white_kv.png)', '![Velocity](fig_white_flow.png)', '']
    (dest / 'results.md').write_text('\n'.join(lines))
    print(json.dumps({'conditions': summary['conditions'], 'pixel_means': summary['pixel_means']}, indent=2))


if __name__ == '__main__':
    main()
