"""Plot saved native clean frames without rerunning the simulator or policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=Path(__file__).resolve().parent/'out/clean_pilot')
    args = parser.parse_args()
    result = json.loads((args.run/'results.json').read_text())
    assert result['complete'], 'do not present an incomplete pilot as final'
    episodes = result['episodes']
    fig, axes = plt.subplots(2, len(episodes), figsize=(3.2*len(episodes), 5.4), squeeze=False)
    for column, episode in enumerate(episodes):
        with np.load(args.run/episode['record']) as data:
            axes[0,column].imshow(data['head_camera'][0], interpolation='nearest')
            axes[1,column].imshow(data['final_head_camera'], interpolation='nearest')
        status = 'success' if episode['success'] else 'failure'
        label = f"seed {episode['seed']} / {episode['actor_name']} {episode['container_id']}"
        axes[0,column].set_title(label, fontsize=9)
        axes[1,column].set_title(f"{status} / {episode['steps']} steps", fontsize=10,
                                 color='green' if episode['success'] else 'firebrick')
        for row in range(2):
            axes[row,column].set_xticks([]); axes[row,column].set_yticks([])
    axes[0,0].set_ylabel('Initial clean RGB')
    axes[1,0].set_ylabel('Final clean RGB')
    fig.suptitle(f"RoboTwin clean integration pilot: {result['task_successes']}/{len(episodes)} task successes (no patch)")
    fig.tight_layout()
    target = args.run/'fig_clean_pilot.png'
    fig.savefig(target, dpi=120)
    plt.close(fig)
    print(target)


if __name__ == '__main__':
    main()
