"""Stage-specialist progress and same-seed full versus phase-only task failures."""
import json

from crossarch_common import ROOT, STAGES, dump
from crossarch_stage_common import DEFAULT, MODELS, condition, load


def report(root=DEFAULT):
    p = load(root)
    status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else dict(stage='prepared')
    lines = ['# RoboTwin：分阶段patch训练与闭环验证', '',
        '按用户要求先训练8张：π0.5和ImageWAM各自的接近、抓取、搬运、放置patch。联合版及跨模型迁移后续再做。', '',
        f'当前状态：**{status["stage"]}**。' + (f' 错误：`{status["error"]}`' if status.get('error') else ''), '',
        '每张纹理10cm、Q128、灰色初始化、500×8、lr .01、相同3cm位移／±15°旋转／三光照EOT。两模型同一阶段共享固定状态/noise抽样计划，只更新纹理。', '',
        '阶段窗口：开始至夹爪开度<0.85为接近；随后至容器抬升>3cm为抓取；随后至容器距盘中心<6cm为搬运；余下为放置。闭环阶段由当前真实状态在线判定并保持单调，不使用未来专家动作。', '',
        '全程组始终有patch；阶段限定组只在对应阶段出现并在离开时移除。为避免阶段短于动作段而从未观察到patch，所有组（包括clean和Random）在阶段边界立即重规划，其余时刻保持各自原生最大动作段。新clean基线独立测量，不复用旧SR。', '',
        '原方向余弦0.8和L0相对误差0.5作为诊断记录，未宣称原检验通过；本轮保留像素误差、原生推理一致性及有限非零纹理梯度检查。效果以真实render配对闭环为准。', '',
        f'训练标定：{len(list((root/"bank/train").glob("*.npz")))}/1536；验证标定：{len(list((root/"bank/validation").glob("*.npz")))}/128。', '',
        '| 模型 | 专用阶段 | patch更新 |', '|---|---|---|']
    progress = {}
    for model in MODELS:
        progress[model] = {}
        for stage in STAGES:
            path = root / 'patches' / model / stage / 'progress.json'
            row = json.loads(path.read_text()) if path.exists() else dict(iteration=0, complete=False)
            progress[model][stage] = row
            lines.append(f'| {model} | {stage} | {row["iteration"]}/500 |')
    results = {}
    for model in MODELS:
        results[model] = {}
        for spec in p['evaluation']['conditions']:
            rows = [json.loads(f.read_text()) for f in (root / 'episodes' / model / spec).glob('seed*.json')]
            results[model][spec] = {r['seed']: r for r in rows if 'success' in r}
    summary = []
    lines += ['', '先导闭环固定22000–22009，不按攻击效果扩样。每模型14条件×10初态，合计280回合；同一个Random纹理提供全程与四个阶段窗口对照。', '',
        '| 模型 | clean成功／已完成 |', '|---|---|']
    for model in MODELS:
        clean = results[model]['clean']
        lines.append(f'| {model} | {sum(r["success"] for r in clean.values())}/{len(clean)} |')
    lines += ['', '| 模型 | 专用patch | 出现窗口 | 任务成功／已完成 | 配对ASR | 相应Random ASR | 已配对初态 |', '|---|---|---|---|---|---|---|']
    for model in MODELS:
        clean = results[model]['clean']
        for stage in STAGES:
            for mode in ('full', 'stage_only'):
                key = stage + '_' + mode
                attacked = results[model][key]
                random_key = 'random_full' if mode == 'full' else 'random_' + stage
                rand = results[model][random_key]
                paired = sorted(set(clean) & set(attacked) & set(rand))
                eligible = [s for s in paired if clean[s]['success'] and
                    (mode == 'full' or stage in clean[s]['query_phases'])]
                no_exposure = sum(not attacked[s]['active_queries'] or not rand[s]['active_queries']
                    for s in eligible) if mode == 'stage_only' else 0
                eligible = [s for s in eligible if mode == 'full' or
                    (attacked[s]['active_queries'] > 0 and rand[s]['active_queries'] > 0)]
                failures = sum(not attacked[s]['success'] for s in eligible)
                random_failures = sum(not rand[s]['success'] for s in eligible)
                row = dict(model=model, specialist_stage=stage, exposure_mode=mode, paired=len(paired),
                    eligible=len(eligible), failures=failures, ASR=failures/len(eligible) if eligible else None,
                    random_failures=random_failures, random_ASR=random_failures/len(eligible) if eligible else None,
                    no_exposure=no_exposure, eligible_exposed=len(eligible),
                    completed=len(attacked), task_successes=sum(r['success'] for r in attacked.values()))
                summary.append(row)
                asr = f'{failures}/{len(eligible)}' if eligible else '待测'
                random_asr = f'{random_failures}/{len(eligible)}' if eligible else '待测'
                lines.append(f'| {model} | {stage} | {mode} | {row["task_successes"]}/{len(attacked)} | {asr} | {random_asr} | {len(paired)}/10 |')
    errors = [dict(path=str(f), error=d['error']) for f in (root / 'episodes').glob('*/*/seed*.json')
              if 'error' in (d := json.loads(f.read_text()))]
    dump(root / 'summary.json', dict(stage=status['stage'], patches=progress, rows=summary,
        complete=status['stage']=='complete', errors=errors,
        completed_episodes=sum(len(v) for groups in results.values() for v in groups.values()), expected_episodes=280))
    lines += ['', 'ASR以相同初态、目标模型clean成功为条件；阶段限定组另要求clean在该阶段有实际查询机会。缺少曝光、运行错误和未到达阶段单列，不当作攻击失败或任务失败。不同窗口的暴露时长不同，阶段限定对照不能证明纯粹的单位时间敏感性。', '',
        '这是一轮10初态探索性实验，patch有效性和阶段排序需看真实计数，不由离线L0或cosine单独推断。', '',
        f'原始目录：`{root}`。配置：`protocol.json`；日志：`logs/`；纹理：`patches/模型/阶段/`。']
    (ROOT / 'result/32_robotwin_stage_specialists.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    report()
