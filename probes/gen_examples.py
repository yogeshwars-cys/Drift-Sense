"""Regenerate examples/success_case.png and examples/failure_case.png against
the current pipeline. Matches the existing two-panel style: reference on the
left, search image with true/predicted markers on the right.
"""
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'D:\Downloads\projects\semicon')
import localize as L

GT = json.load(open('dataset_primary/ground_truth.json'))


def render(pair_id, out_path, label):
    g = GT[pair_id]
    ref = L.load_gray(g['ref_path'])
    sea = L.load_gray(g['search_path'])
    x, y, info = L.localize(ref, sea)
    err = ((x - g['gt_x']) ** 2 + (y - g['gt_y']) ** 2) ** 0.5
    tag = 'SUCCESS' if err <= 15 else 'HONEST FAILURE'
    conf = info.get('confidence', 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    ax[0].imshow(ref, cmap='gray')
    ax[0].set_title(f"{g['style']}_{pair_id:03d} reference")
    ax[0].axis('off')

    ax[1].imshow(sea, cmap='gray')
    ax[1].plot(g['gt_x'], g['gt_y'], '+', color='lime', markersize=18, mew=3,
               label=f"true ({g['gt_x']:.0f},{g['gt_y']:.0f})")
    ax[1].plot(x, y, '+', color='red', markersize=18, mew=3,
               label=f"predicted ({x:.0f},{y:.0f})")
    ax[1].set_title(f"search — {tag}, error={err:.1f}px, conf={conf:.2f}")
    ax[1].legend(loc='upper right')
    ax[1].axis('off')

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f'{label}: pair {pair_id} ({g["style"]}) -> {out_path}  '
          f'{tag}, error={err:.2f}px, conf={conf:.2f}')


if __name__ == '__main__':
    render(34, 'examples/success_case.png', 'success')
    render(38, 'examples/failure_case.png', 'failure')
