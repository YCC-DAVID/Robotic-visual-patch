"""Completed Q128 training, independent true-render diagnostics and paired pilot figures."""
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent / 'out/render_proxy_eot'
    final = root / 'final_report'
    final.mkdir(exist_ok=True)
    configs, results, checkpoints = {}, {}, {}
    for c, folder in [('nominal', 'train_nominal'), ('eot', 'train_eot_run1')]:
        configs[c] = json.loads((root / folder / 'config.json').read_text())
        results[c] = json.loads((root / folder / 'results.json').read_text())
        checkpoints[c] = np.load(root / folder / 'checkpoint_0500.npz', allow_pickle=False)
        assert results[c]['complete'] and results[c]['gradient_frame_forwards'] == 4000
    assert np.array_equal(checkpoints['nominal']['batches'], checkpoints['eot']['batches'])
    paired = json.loads((root / 'trained_validation/summary.json').read_text())
    early = json.loads((root / 'early_validation_0100/summary.json').read_text())
    baseline = json.loads((root / 'physical_baseline/episodes.json').read_text())
    nominal_test = json.loads((root / 'physical_nominal/episodes.json').read_text())
    eot_test = json.loads((root / 'physical_eot/episodes.json').read_text())
    conditions = {**baseline['per_episode'], **nominal_test['per_episode'], **eot_test['per_episode']}
    reference = {r['init_index']: r for r in conditions['clean']}
    clean_success = {r['init_index'] for r in conditions['clean'] if r['success']}
    physical = {}
    for c, rows in conditions.items():
        assert {r['init_index'] for r in rows} == set(range(40, 45))
        assert all(r['initial_state_sha256'] == reference[r['init_index']]['initial_state_sha256']
                   for r in rows), 'paired conditions must share initial physics'
        eligible = [r for r in rows if r['init_index'] in clean_success]
        failures = sum(not r['success'] for r in eligible)
        physical[c] = {'successes': sum(r['success'] for r in rows), 'n': len(rows),
                       'paired_ASR_failures': failures, 'paired_ASR_n': len(eligible),
                       'paired_ASR': failures / len(eligible) if eligible else None,
                       'per_episode': rows}
    summary = {'complete': True, 'training': results, 'true_render_diagnostic': paired,
               'early100_diagnostic': early, 'physical': physical,
               'checks': {'equal_4000_gradient_frames': True, 'identical_batch_schedule': True,
                          'all_conditions_initial_state_hashes_equal': True,
                          'complete_fixed_test_indices': True},
               'notes': 'One nominal and one EOT texture, one shared training seed, 10 training episodes. '
                        'Validation state photometry is calibrated; new optimizer never uses validation gradients. '
                        '32-state open-loop is base-only; actual closed loop uses both patched cameras. '
                        'Five predefined test initial states; report paired ASR only within 4 clean successes. '
                        'Historical full and random-image exports are open-loop references; uniform Q128 is '
                        'the closed-loop random control. No broad ASR or deployment robustness claim.'}
    (final / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'nominal': '#297da4', 'eot': '#c45449'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for c in ['nominal', 'eot']:
        hist = checkpoints[c]['history']
        axes[0].plot(np.arange(1, 501), hist, alpha=.15, color=colors[c])
        smooth = np.convolve(hist, np.ones(25)/25, mode='valid')
        axes[0].plot(np.arange(25, 501), smooth, label=c, color=colors[c], linewidth=2)
        y = [early[c+'_render']['L0'], paired[c+'_render']['L0']]
        axes[1].plot([100, 500], y, 'o-', label=c, color=colors[c])
        axes[1].plot([100, 500], [early[c+'_proxy']['L0'], paired[c+'_proxy']['L0']], '--',
                     color=colors[c], alpha=.5)
    axes[0].set_xlabel('PGD update'); axes[0].set_ylabel('Batch t=1 velocity squared objective')
    axes[0].set_title('Training: raw and 25-update moving mean')
    axes[1].set_title('32 validation states: solid true render, dashed proxy')
    axes[1].set_xlabel('Checkpoint iteration'); axes[1].set_ylabel('Mean L0 / clean, 3 common noises')
    for ax in axes: ax.legend(); ax.grid(alpha=.2); ax.set_ylim(bottom=0)
    fig.suptitle('E22: same Q128, same batches, 500x8 gradient budget; training inputs differ by EOT')
    fig.savefig(final / 'fig_training_and_render.png', dpi=160); plt.close(fig)

    names = ['clean', 'random_world', 'old_full', 'nominal', 'eot']
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    x = np.arange(len(names)); y = [physical[c]['successes'] for c in names]
    ax.bar(x, y, color=['#777777', '#a0a0a0', '#dfb466', colors['nominal'], colors['eot']])
    ax.set_xticks(x); ax.set_xticklabels(['Clean', 'Uniform random Q128', 'Historical full', 'Nominal proxy', 'Proxy + EOT'])
    ax.set_ylim(0, 5.8); ax.set_yticks(range(6)); ax.set_ylabel('Robot task successes / 5 initial states')
    for xi, yi, c in zip(x, y, names):
        ax.text(xi, yi+.12, str(yi)+'/5', ha='center')
    ax.set_title('E22 physical pilot: both actual rendered views; test init40-44; lower means stronger disruption')
    ax.grid(axis='y', alpha=.2)
    fig.savefig(final / 'fig_physical_pilot.png', dpi=160); plt.close(fig)

    inputs = np.load(root / 'trained_validation/inputs.npz', allow_pickle=False)
    predictions = np.load(root / 'trained_validation/model_inputs.npz', allow_pickle=False)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
    for row, c in enumerate(['nominal', 'eot']):
        axes[row, 0].imshow(inputs[c+'__texture'], interpolation='nearest')
        axes[row, 0].set_title(c+' physical Q128 PNG')
        axes[row, 1].imshow(predictions[c+'_proxy'][0]); axes[row, 1].set_title('Nominal proxy, init10 first state')
        axes[row, 2].imshow(inputs[c+'__render'][0]); axes[row, 2].set_title('Actual MuJoCo render, same state')
        err = np.abs(predictions[c+'_proxy'][0].astype(float)-inputs[c+'__render'][0].astype(float)).mean(-1)
        h = axes[row, 3].imshow(err, vmin=0, vmax=20, cmap='magma'); axes[row, 3].set_title('Mean absolute RGB error, u8')
        fig.colorbar(h, ax=axes[row, 3], fraction=.04)
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('E22 learned textures and full-image renderer validation; error scale 0-20/255')
    fig.savefig(final / 'fig_learned_textures.png', dpi=160); plt.close(fig)

    lines = ['# E22 completed same-budget world-texture experiment', '',
             '500 updates x 8 identical sampled frames for nominal and EOT; Q128 directly exported as PNG. '
             'New training init0-9; validation10-11; closed-loop test40-44. One shared training seed.', '',
             '| Mode | Train diagnostic proxy L0 | Validation proxy L0, training noise | Validation true-render L0, 3 noises | Direct proxy/render velocity L0 |',
             '|---|---:|---:|---:|---:|']
    for c in ['nominal', 'eot']:
        r = results[c]
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            c, r['final_train_diag_L0'], r['final_validation_proxy_L0'], paired[c+'_render']['L0'],
            paired[c+'_proxy']['paired_L0_vs_render']))
    lines += ['', '| Learned texture | Proxy L0, same three noises | True-render L0 / clean | Direct proxy/render KV relative distance |',
              '|---|---:|---:|---:|']
    for c in ['nominal', 'eot']:
        lines.append('| {} | {:.6f} | {:.6f} | {:.6f} |'.format(
            c, paired[c+'_proxy']['L0'], paired[c+'_render']['L0'],
            paired[c+'_proxy']['KV_relative_render']))
    lines += ['', 'The same 32 states give historical full/random-export true-render L0 '
              '{:.6f}/{:.6f}; these historical patterns are reference conditions, not equal-budget controls.'.format(
                  paired['old_full_render']['L0'], paired['old_random_render']['L0'])]
    lines += ['', '| Physical condition | Robot successes | Paired attack failures / clean successes |',
              '|---|---:|---:|']
    for c in names:
        r = physical[c]
        lines.append('| {} | {}/{} | {}/{} |'.format(c, r['successes'], r['n'],
                                                   r['paired_ASR_failures'], r['paired_ASR_n']))
    lines += ['', 'All episode outcomes are retained, including init40 where clean failed. '
              'These five episodes do not estimate broad physical ASR; EOT benefit must be read from '
              'the actual paired outcomes, not assumed from its objective. Old full uses a different '
              'historical pool and is not an equal-budget new training control.', '',
              '![Training and render](fig_training_and_render.png)',
              '![Physical pilot](fig_physical_pilot.png)', '![Textures](fig_learned_textures.png)', '']
    (final / 'results.md').write_text('\n'.join(lines))
    print('[done]', final)


if __name__ == '__main__':
    main()
