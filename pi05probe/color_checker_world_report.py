"""Plot all four world checkers with forward projection in column two.

Read-only with respect to captured inputs and pixel/model measurements. The
unprojected cut example stays in the original figure and is excluded from the
main comparison. New filenames preserve the original capture artifacts.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='color_checker_world')
    args = ap.parse_args()
    dest = Path(__file__).resolve().parent / 'out' / args.out
    source = dest / 'inputs.npz'
    data = np.load(source, allow_pickle=False)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else FontProperties()
    mask = data['mask'][0].astype(bool)
    by, bx = np.where(mask)
    sy = slice(max(0, int(by.min()) - 5), min(224, int(by.max()) + 6))
    sx = slice(max(0, int(bx.min()) - 5), min(224, int(bx.max()) + 6))
    columns = [
        ('Q128', '① 世界纹理 Q128\n规则方格'),
        ('projected_nearest224', '② 数字参考\n正向透视 / nearest / 224'),
        ('minimal_unlit_nearest224', '③ 真实 render\n无光照 / nearest / AA0 / 224'),
        ('unlit_default256_resize224', '④ 真实 render\n无光照 / 默认过滤 / AA4 / 256→224'),
        ('lit_default256_resize224', '⑤ 真实 render\n正常光照 / 默认过滤 / AA4 / 256→224'),
    ]
    rows = []
    fig, axes = plt.subplots(4, len(columns), figsize=(17, 12), constrained_layout=True)
    for row, (name, size) in enumerate(zip(data['patterns'], data['source_Q128_cell_sizes'])):
        keys = []
        for col, (condition, title) in enumerate(columns):
            key = str(name) + '__' + condition
            image = data[key] if condition == 'Q128' else data[key][0, sy, sx]
            axes[row, col].imshow(image, interpolation='nearest')
            if row == 0:
                axes[row, col].set_title(title, fontproperties=font, fontsize=11, pad=12)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
            keys.append(key)
        axes[row, 0].set_ylabel('格宽 {} texels\n{:.3f} mm'.format(
            int(size), float(data['patch_m']) * 1000 * int(size) / 128),
            fontproperties=font, fontsize=11)
        rows.append({'row_one_based': row + 1, 'pattern': str(name), 'cell_width_texels': int(size),
                     'columns': keys})
    fig.suptitle('四行均为世界纹理透视对照：同一 Q、相机、状态与可见 mask',
                 fontproperties=font, fontsize=16)
    figure = dest / 'fig_world_checker_perspective.png'
    fig.savefig(figure, dpi=160); plt.close(fig)

    # Make the large-cell projection legible without neighboring dense rows.
    name = 'uvcell16'
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    zoom_columns = [columns[1], columns[2], columns[4]]
    for ax, (condition, title) in zip(axes, zoom_columns):
        ax.imshow(data[name + '__' + condition][0, sy, sx], interpolation='nearest')
        ax.set_title(title, fontproperties=font, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('16 texels 世界方格：数字参考也经过透视变换', fontproperties=font, fontsize=15)
    zoom = dest / 'fig_world_checker_large_cells.png'
    fig.savefig(zoom, dpi=160); plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    report = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'input_sha256': source_hash, 'input_file_unchanged': True, 'frame_index': 0,
              'environment_step': int(data['steps'][0]), 'rows': rows,
              'new_main_column_two': 'projected_nearest224', 'unprojected_cut_in_main_figure': False,
              'new_images': [figure.name, zoom.name],
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'notes': 'New plot layout only. All four texture sizes were already captured with perspective. '
                       'No new render, calibration, pixel metric, model forward or closed-loop result. '
                       'Original cut-example figure remains as a historical artifact.'}
    (dest / 'presentation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'complete': True, 'figures': report['new_images'], 'rows': rows}, ensure_ascii=False))


if __name__ == '__main__':
    main()
