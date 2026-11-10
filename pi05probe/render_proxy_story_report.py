"""Compare paired model fidelity of E17 and calibrated world-texture proxies."""
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent / 'out/render_proxy_eot'
    dest = root / 'story_report'
    dest.mkdir(exist_ok=True)
    runs = [('E17', 'validation_e17', 'e17'), ('World: four-sample approximation', 'validation_world', 'world'),
            ('World: native center', 'validation_world_center', 'world')]
    data = []
    for label, folder, condition in runs:
        metrics = json.loads((root / folder / 'metrics.json').read_text())
        assert metrics['complete'] and len(metrics['records']) == 144
        data.append((label, metrics, condition))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    patterns = ['full', 'band32', 'random']
    colors = ['#bd4545', '#dc9a3d', '#2b8291']
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, pattern in zip(axes, patterns):
        for color, (label, metrics, cond) in zip(colors, data):
            rows = [r for r in metrics['records'] if r['pattern'] == pattern and r['condition'] == cond]
            y = [np.mean([r['vs_render']['layers'][l]['V']['relative'] for r in rows]) for l in range(18)]
            ax.plot(range(18), y, label=label, color=color, linewidth=2)
        ax.set_title(pattern); ax.set_xlabel('Backbone layer (0-based)')
        ax.set_ylabel('||V(proxy) - V(render)|| / ||V(render)||')
        ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('E22 proxy fidelity: direct paired backbone differences, 16-state mean')
    fig.savefig(dest / 'fig_proxy_kv_fidelity.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)
    x = np.arange(3)
    for j, (label, metrics, cond) in enumerate(data):
        rows_by_pattern = [[r for r in metrics['records'] if r['pattern'] == p and r['condition'] == cond]
                           for p in patterns]
        for ax, key in zip(axes, ['L0', 'L0_real7', 'L0_exec7']):
            y = [np.mean([r['vs_render'][key] for r in rows]) for rows in rows_by_pattern]
            ax.bar(x + (j-1)*.25, y, width=.23, color=colors[j], label=label)
            ax.set_xticks(x); ax.set_xticklabels(patterns)
            ax.set_ylabel('Squared paired velocity difference to true render')
            ax.grid(axis='y', alpha=.2)
            ax.set_title({'L0': 'All 10x32', 'L0_real7': 'Real 10x7', 'L0_exec7': 'First 5x7'}[key])
    axes[0].legend(fontsize=8)
    fig.suptitle('E22: same three action noises at t=1; no integration or ASR in this comparison')
    fig.savefig(dest / 'fig_proxy_velocity_fidelity.png', dpi=160)
    plt.close(fig)
    lines = ['# Paired proxy fidelity', '',
             '16 correlated replay states, three patterns and three common action noises. '
             'Each candidate is frozen before optimized-pattern evaluation; same full mask and clean background.', '',
             '| Pattern | Candidate | KV / render | L0 vs render | L0 / clean | RGB RMSE |',
             '|---|---|---:|---:|---:|---:|']
    for pattern in patterns:
        for label, metrics, cond in data:
            rows = [r for r in metrics['records'] if r['pattern'] == pattern and r['condition'] == cond]
            values = [np.mean([r['vs_render']['all18_KV']['relative'] for r in rows]),
                      np.mean([r['vs_render']['L0'] for r in rows]),
                      np.mean([r['vs_clean']['L0'] for r in rows]),
                      np.mean([r['pixel_RMSE_vs_render'] for r in rows])]
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(pattern, label, *values))
    lines += ['', '![Backbone fidelity](fig_proxy_kv_fidelity.png)',
              '![Paired velocity fidelity](fig_proxy_velocity_fidelity.png)', '',
              'These validate the existing patterns only. Learned Q128 may exploit proxy errors; '
              'new texture true-render checks and closed loop remain necessary. No measured renderer '
              'gradient or physical MSAA shader frequency is claimed from the sampling-model comparison.', '']
    (dest / 'results.md').write_text('\n'.join(lines))
    print('[done]', dest)


if __name__ == '__main__':
    main()
