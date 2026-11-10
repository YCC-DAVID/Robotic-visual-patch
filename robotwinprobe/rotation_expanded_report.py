"""Independent-seed uncertainty and traceable reports for the larger yaw cohort."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from digital_transform_common import ROOT, dump
from rotation_expanded_plan import DEFAULT, ANGLES, GROUPS, prepare, records, episode_dir


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0., center - width), min(1., center + width)]


def paired_difference(a, b, rng_seed):
    """a-b, paired by independent clean-success initial seed, not by frame."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    assert a.shape == b.shape and a.ndim == 1
    if not len(a):
        return None
    delta = a - b
    rng = np.random.RandomState(rng_seed)
    boot = delta[rng.randint(len(delta), size=(20000, len(delta)))].mean(1)
    plus, minus = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
    return dict(n=len(a), difference=float(delta.mean()),
        bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
        attack_success_only_a=plus, attack_success_only_b=minus,
        both_attack_success=int(((a == 1) & (b == 1)).sum()),
        neither_attack_success=int(((a == 0) & (b == 0)).sum()))


def build_report(root, publish=True):
    protocol = prepare(root)
    cohort_path = root / 'cohort.json'
    cohort = json.loads(cohort_path.read_text()) if cohort_path.exists() else None
    seeds = cohort['seeds'] if cohort else protocol['initial_seeds']
    selected = set(seeds)
    rr = {(c, m): records(root, c, m) for c, m in GROUPS}
    clean = rr['nominal', 'clean']
    clean_ready = cohort is not None and selected <= clean.keys()
    eligible = sorted(s for s in seeds if s in clean and clean[s]['success'])
    rows, binary = [], {}
    for cname, method in GROUPS:
        data = rr[cname, method]
        row = dict(condition=cname, method=method, completed=len(selected & data.keys()), target=len(seeds),
            complete=clean_ready and selected <= data.keys())
        if row['complete']:
            row['task_successes'] = sum(data[s]['success'] for s in seeds)
            if method != 'clean':
                failures = [s for s in eligible if not data[s]['success']]
                row.update(attack_successes=len(failures), clean_successes=len(eligible),
                    ASR=len(failures) / len(eligible) if eligible else None,
                    ASR_wilson95=wilson(len(failures), len(eligible)), failed_seeds=failures)
                binary[cname, method] = [int(not data[s]['success']) for s in eligible]
        rows.append(row)
    comparisons = []
    keys = [((c, 'full'), ('nominal', 'full')) for c in ANGLES[1:]]
    keys += [((c, 'full'), (c, 'random')) for c in ANGLES]
    for i, (a, b) in enumerate(keys):
        if a in binary and b in binary:
            comparisons.append(dict(a=list(a), b=list(b), **(paired_difference(binary[a], binary[b], 20261090 + i) or {})))
    summary = dict(complete=all(r['complete'] for r in rows), cohort_frozen=cohort is not None,
        initial_states=len(seeds), clean_finished=len(selected & clean.keys()), clean_successes_so_far=len(eligible),
        target_clean_successes=50, target_reached=(len(eligible) >= 50 if clean_ready else None),
        rows=rows, paired_comparisons=comparisons)
    dump(root / 'summary.json', summary)
    with (root / 'asr.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)
    state = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {'stage': 'prepared_not_started'}
    rel = '../' + str(root.relative_to(ROOT))
    lines = ['# 15　数字patch旋转：独立初态扩样', '',
        '[总览](README.md) · [原10初态先导实验](10_digital_transform_sensitivity.md)', '',
        '原实验只有8个clean成功初态：Full原朝向3/8，−10°为2/8，+10°为1/8。一个样本即12.5个百分点，不能仅凭这些计数认定稳定的旋转鲁棒性差异。', '',
        '本轮评估RoboTwin `place_container_plate / demo_clean` 的 **π0.5原数字patch**；不是ImageWAM局部结构实验。冻结20 cm、128×128、第500步Full纹理与原Random纹理。两张纹理均在0°、−10°、+10°下测试，另有一组clean。', '',
        '## 固定方案', '',
        '- 首批100个新初态：10600–10699；与训练、旧验证和旧10初态分离。',
        '- 若首批clean成功少于50个，再加入完整的100个初态10700–10799；最多200个初态，不按攻击结果延长或筛选。',
        '- 确定初态全集后，每组跑同一全集，包括clean失败的初态。ASR仅以其中clean成功者作分母。',
        '- 世界矩形和纹理一起旋转，位置、光照不变；使用原数字透视合成与逐查询更新的可见mask。',
        '- 三相机共享同一世界patch。所有条件逐集核对初始state、原始RGB和prompt；动作noise沿用 `20261006 + seed*1000 + query`，每32步重规划。',
        '- 原8个clean成功初态保留为先导结果，不并入新统计；多帧和多份noise不计作独立初态。', '',
        '## 进度与闭环结果', '',
        f"进度：`{state['stage']}`。当前选定批次clean完成 **{summary['clean_finished']}/{len(seeds)}**，其中成功 **{len(eligible)}**。{'初态全集已冻结。' if cohort else '尚未冻结最终初态全集。'}", '',
        '| 工况 | 已完成／初态数 | 任务成功／初态数 | ASR | 95% Wilson区间 |',
        '|---|---:|---:|---:|---:|']
    labels = {'nominal': '0°', 'yaw_minus10': '−10°', 'yaw_plus10': '+10°'}
    for row in rows:
        name = 'Clean' if row['method'] == 'clean' else f"{row['method'].title()} {labels[row['condition']]}"
        rate, ci, task_rate = '待整组完成', '—', '待整组完成'
        if row['complete']:
            task_rate = f"{row['task_successes']}/{len(seeds)}"
            if row['method'] == 'clean':
                rate = '—'
            elif row['clean_successes']:
                rate = f"{row['attack_successes']}/{row['clean_successes']} ({row['ASR']:.1%})"
                ci = '–'.join(f'{v:.1%}' for v in row['ASR_wilson95'])
            else:
                rate = '无clean成功初态'
        lines.append(f"| {name} | {row['completed']}/{len(seeds)} | {task_rate} | {rate} | {ci} |")
    lines += ['', 'ASR＝原本clean成功、加patch后任务失败的比例。各条件跑齐后才发布ASR；运行中不按局部结果调整纹理或参数。']
    if comparisons:
        lines += ['', '| 配对比较（前者ASR−后者ASR） | 差值 | 95%配对bootstrap区间 | 仅前者攻击成功／仅后者攻击成功 |', '|---|---:|---:|---:|']
        for c in comparisons:
            if 'difference' not in c:
                continue
            a, b = c['a'], c['b']
            label = f'{a[1].title()} {labels[a[0]]} − {b[1].title()} {labels[b[0]]}'
            interval = ' 至 '.join(f'{v * 100:+.1f}' for v in c['bootstrap95'])
            lines.append(f"| {label} | {c['difference']*100:+.1f} pp | {interval} pp | {c['attack_success_only_a']}／{c['attack_success_only_b']} |")
    lines += ['', '每个独立初态为一个重采样单位，共20000次；同一次抽样保留其全部条件配对。区间为逐项95%区间。主比较是Full ±10°相对Full 0°，Full与Random的差为补充对照。', '',
        '这轮只扩大同一任务、同一冻结纹理的测试初态数，不代表覆盖了多任务或多次独立训练。旋转同时改变投影采样，不能仅凭ASR下降断言存在对朝向敏感的独立结构。', '',
        f'原始记录：[协议]({rel}/protocol.json) · [运行状态]({rel}/status.json) · [汇总]({rel}/summary.json) · [CSV]({rel}/asr.csv)。']
    if (root / 'audit.json').exists():
        lines += [f'[完成后的逐集配对核对]({rel}/audit.json)。']
    if publish:
        (ROOT / 'result/15_rotation_expanded.md').write_text('\n'.join(lines) + '\n')
    return summary


def audit(root):
    protocol = prepare(root)
    cohort = json.loads((root / 'cohort.json').read_text())
    checked = 0
    for seed in cohort['seeds']:
        baseline = episode_dir(root, 'nominal', 'clean') / f'seed{seed}'
        clean = json.loads(baseline.with_suffix('.json').read_text())
        with np.load(baseline.with_suffix('.npz')) as z:
            state = z['initial_state14'].copy()
            rgb = {n: z['initial_clean_' + n].copy() for n in ['head_camera', 'left_camera', 'right_camera']}
        masks = {}
        for cname, method in GROUPS:
            stem = episode_dir(root, cname, method) / f'seed{seed}'
            row = json.loads(stem.with_suffix('.json').read_text())
            assert row['prompt'] == clean['prompt']
            assert row['initial_state_sha256'] == clean['initial_state_sha256']
            assert row['initial_rgb_sha256'] == clean['initial_rgb_sha256']
            assert row['texture_sha256'] == (None if method == 'clean' else protocol['texture_sha256'][method])
            with np.load(stem.with_suffix('.npz')) as z:
                assert np.array_equal(z['initial_state14'], state)
                assert np.array_equal(z['states14'][0], state)
                for n, value in rgb.items():
                    assert np.array_equal(z['initial_clean_' + n], value)
                assert np.array_equal(z['noise_seeds'], 20261006 + seed * 1000 + np.arange(row['queries']))
                assert z['actions'].shape == (row['queries'], 32, 14)
                assert np.isfinite(z['actions']).all()
                if method != 'clean':
                    assert row['paired_initial_exact'] and row['patch_removal_restores_initial_clean']
                    if cname in masks:
                        assert np.array_equal(masks[cname], z['initial_masks']), (seed, cname, 'mask')
                    else:
                        masks[cname] = z['initial_masks'].copy()
            checked += 1
    result = dict(complete=True, initial_states=len(cohort['seeds']), checked_episodes=checked,
        raw_initial_RGB_state_prompt_exact=True, query_noise_rule_exact=True,
        first_query_full_random_masks_equal=True, frozen_texture_hashes_match=True,
        no_pilot_seeds_in_new_cohort=True)
    assert not set(cohort['seeds']) & set(protocol['old_pilot_seeds'])
    dump(root / 'audit.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(DEFAULT))
    ap.add_argument('--audit', action='store_true')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    if args.audit:
        audit(root)
    print(json.dumps(build_report(root), ensure_ascii=False, indent=2))
