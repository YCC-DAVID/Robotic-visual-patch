"""Compare max-min joint training dynamics: #51 (pi+fast), #57 (pi+imagewam), #62 (pi+fast+imagewam).
Parses per-step log lines '[tag] it {raw} {ratios} chosen' and the per-step reference curves."""
import ast, json, re, glob
from collections import Counter
import numpy as np

O = '/shared/user65/workspace/chence/WAMattack/robotwinprobe/out/'
RUNS = {'51 pi+fast': 'libero13_joint_eot_20261029', '57 pi+imagewam': 'pi_imagewam_joint_eot_20261031',
        '62 three': 'three_source_joint_eot_20261106'}
pat = re.compile(r'^\[[^\]]+\] (\d+) (\{.*?\}) (\{.*?\}) (\w+)\s*$')

for name, d in RUNS.items():
    steps = {}
    for f in sorted(glob.glob(O + d + '/logs/pipeline*.log')):
        for line in open(f):
            m = pat.match(line.strip())
            if m:
                steps[int(m.group(1))] = (ast.literal_eval(m.group(2)), ast.literal_eval(m.group(3)), m.group(4))
    import os; rf = O + d + '/specialist_reference.json'; ref = json.load(open(rf if os.path.exists(rf) else O + d + '/reference.json'))
    ref = json.load(open(O + 'three_source_joint_eot_20261106/specialist_reference.json')); ks = list(steps[1][0])
    print('==', name, 'steps', len(steps), 'chosen', dict(Counter(s[2] for s in steps.values())))
    for lo in range(1, 501, 100):
        its = [i for i in range(lo, lo + 100) if i in steps]
        row = []
        for k in ks:
            raw = np.mean([steps[i][0][k] for i in its]); rr = np.mean([ref[k][i - 1] for i in its])
            row.append('%s raw %.3f ref %.3f r̄ %.2f n %d' % (k, raw, rr, raw / rr, sum(steps[i][2] == k for i in its)))
        print('  %3d-%3d  ' % (lo, lo + 99) + ' | '.join(row))
    if name.startswith('62'):
        L = [json.loads(l) for l in open(O + d + '/gradient_ledgers/pi.jsonl')]
    prog = json.load(open(O + d + '/training/progress.json'))
    print('  last cosine', prog['last'].get('cosine'))
