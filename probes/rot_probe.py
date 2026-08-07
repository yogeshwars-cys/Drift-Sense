"""Clean probe of the SHIPPED relative_rotation(), against recorded truth."""
import sys, os, json, time
sys.path.insert(0, r'D:\Downloads\projects\semicon')
os.chdir(r'D:\Downloads\projects\semicon')
import numpy as np, cv2
import lattice as L

ds = sys.argv[1] if len(sys.argv) > 1 else '/tmp/dsprobe'
recs = json.load(open(f'{ds}/ground_truth.json'))
print(f'{ds}  n={len(recs)}')

for max_padded in (1024, 2000, 4000):
    L._HARM_MAXPAD = max_padded
    errs, ts = [], []
    rows = []
    for m in recs:
        s = cv2.imread(m['search_path'], cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(m['ref_path'], cv2.IMREAD_GRAYSCALE)
        se, lat_s, lat_r, ok = L.estimate_scale(s, r)
        t0 = time.time()
        # call the shipped function, overriding only the pad cap
        _orig = L._harmonic_angles
        L._harmonic_angles = lambda i, b, **kw: _orig(i, b, max_padded=max_padded)
        d, conf = L.relative_rotation(s, r, lat_s['pitch'], se)
        L._harmonic_angles = _orig
        ts.append(time.time() - t0)
        truth = -float(m['rotation_ref_deg'])
        e = abs(d - truth)
        errs.append(e)
        rows.append((m['style'], truth, d, conf, e))
    e = np.array(errs)
    print(f'  max_padded={max_padded:5d}  median={np.median(e):5.2f}  '
          f'p90={np.percentile(e,90):6.2f}  max={e.max():6.2f}  '
          f'{np.mean(ts):.2f}s/img-pair')
    if max_padded == 1024:
        for st, t, d, c, er in rows:
            print(f'      {st:6s} truth={t:6.2f} est={d:6.2f} conf={c:.2f} err={er:5.2f}')
