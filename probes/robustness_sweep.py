"""Robustness of localize.py to conditions the official test set may not share.

Two axes, each a controlled experiment: the seed, the geometry, the placement
and the ground truth are identical across arms, so only the swept variable
differs.

  * NOISE   -- the statement promises the test set is MORE noisy than ours.
  * PITCH   -- the lattice sensor is the load-bearing assumption in the
               pipeline; it was tuned against DRAM 9-14px / FinFET 7-11px at
               search scale. If Applied Materials' generator uses a different
               band, this is where the pipeline would fail.

    python probes/robustness_sweep.py noise
    python probes/robustness_sweep.py pitch
"""
import json, os, subprocess, sys, shutil, time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
N = int(os.environ.get('SWEEP_N', '100'))
SEED = 11
WORK = os.path.join(HERE, '_sweep')

AXES = {
    'noise': ('--noise-scale', [1.0, 2.0, 3.0]),
    'pitch': ('--pitch-shift', [-3, 0, 4, 8]),
}


def run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f'failed: {" ".join(cmd)}')
    return r.stdout


def main():
    axis = sys.argv[1] if len(sys.argv) > 1 else 'noise'
    flag, values = AXES[axis]
    os.makedirs(WORK, exist_ok=True)
    print(f'{axis} sweep  n={N}  seed={SEED}\n')
    rows = []
    for v in values:
        d = os.path.join(WORK, f'{axis}_{v}')
        res = os.path.join(WORK, f'{axis}_{v}.json')
        t0 = time.time()
        run([PY, 'dataset_generator.py', '--n', str(N), '--out', d,
             '--seed', str(SEED), '--difficulty-mix', '0.2',
             '--placement', 'uniform', flag, str(v)])
        gen_s = time.time() - t0
        run([PY, 'benchmark.py', '--dataset', d, '--out', res])
        R = json.load(open(res))
        solv = [r for r in R if r['landmark_in_fov']]
        acc = 100 * sum(r['error_px'] <= 15 for r in R) / len(R)
        sacc = 100 * sum(r['error_px'] <= 15 for r in solv) / len(solv)
        rt = sum(r['runtime_s'] for r in R) / len(R)
        scale_ok = 100 * sum(bool(r['scale_ok']) for r in R) / len(R)
        rows.append((v, acc, sacc, rt, scale_ok, gen_s))
        print(f'  {flag} {v:>5}  <=15px {acc:5.1f}%  solvable {sacc:5.1f}%  '
              f'{rt:4.2f}s/pair  scale_ok {scale_ok:5.1f}%  (gen {gen_s:.0f}s)')
        shutil.rmtree(d, ignore_errors=True)

    print(f'\n| {flag} | <=15px | solvable <=15px | s/pair | lattice scale_ok |')
    print('|---|---|---|---|---|')
    for v, acc, sacc, rt, sok, _ in rows:
        print(f'| {v} | {acc:.1f}% | {sacc:.1f}% | {rt:.2f} | {sok:.0f}% |')
    print(f'\ndataset generation: {sum(r[5] for r in rows)/len(rows):.0f}s '
          f'per {N} pairs (mean over arms)')


if __name__ == '__main__':
    main()
