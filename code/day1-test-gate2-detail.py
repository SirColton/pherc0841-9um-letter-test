"""day1-test-gate2-detail.py - the full fine shift surface behind sanity gate 2, beside the gate, never instead of it.

day1-render-gate2.py prints only the peak and two axis lines of its 25 x 25 scan. This recomputes the same surface
(both maps Gaussian sigma 4 px, Pearson over the render mask minus a 40 px edge, overlap of the two shifted crops,
shifts -12..+12 px) for each map given, saves it, and reports the peak, r at (0, 0), the best r within 1 px of
(0, 0), and a parabolic sub-pixel peak per axis. The first map should be the gate-2 map itself, as a check that this
reproduces day1-render-gate2.py's peak and r(0, 0) exactly.
Usage: day1-test-gate2-detail.py <out.txt> <name=map.npy> [...]    CPU only. Run from <workdir>.
"""
import json
import os
import sys
import time

import numpy as np
from scipy.ndimage import gaussian_filter

import runfile

OUT = sys.argv[1]
runfile.claim(OUT)
R = 'geom/day1-9um/render/'
mask = np.load(R + '252405-mask.npy')
ref = np.load(R + '252405-spmap-grid.npy').astype(np.float32) * 255
H, W = mask.shape
E = 40
M = mask.copy(); M[:E] = M[-E:] = False; M[:, :E] = M[:, -E:] = False
rs = gaussian_filter(ref, 4)
lines = ['day1-test-gate2-detail.py  %s   (r(dy, dx) pairs read[y, x] with key[y - dy, x - dx], day1-render-gate2.py convention)' % time.strftime('%Y-%m-%d %H:%M:%S')]
res = {}


def parab(rm, r0, rp):
    den = rm - 2 * r0 + rp
    return 0.0 if den == 0 else 0.5 * (rm - rp) / den


for arg in sys.argv[2:]:
    name, path = arg.split('=', 1)
    ps = gaussian_filter(np.load(path).astype(np.float32), 4)

    def r_at(dy, dx):
        ys, ye = max(0, dy), min(H, H + dy)
        xs, xe = max(0, dx), min(W, W + dx)
        mm = M[ys:ye, xs:xe] & M[ys - dy:ye - dy, xs - dx:xe - dx]
        return float(np.corrcoef(ps[ys:ye, xs:xe][mm], rs[ys - dy:ye - dy, xs - dx:xe - dx][mm])[0, 1])

    fine = np.array([[r_at(dy, dx) for dx in range(-12, 13)] for dy in range(-12, 13)])
    np.save(os.path.splitext(OUT)[0] + '-%s.npy' % name, fine)
    iy, ix = np.unravel_index(np.nanargmax(fine), fine.shape)
    w1 = fine[11:14, 11:14]
    jy, jx = np.unravel_index(np.argmax(w1), w1.shape)
    sub = None
    if 0 < iy < 24 and 0 < ix < 24:
        sub = (iy - 12 + parab(fine[iy - 1, ix], fine[iy, ix], fine[iy + 1, ix]), ix - 12 + parab(fine[iy, ix - 1], fine[iy, ix], fine[iy, ix + 1]))
    res[name] = {'map': path, 'peak_r': float(fine[iy, ix]), 'peak_px': [int(iy - 12), int(ix - 12)], 'r_zero': float(fine[12, 12]),
                 'best_within_1px_r': float(w1[jy, jx]), 'best_within_1px_at': [int(jy - 1), int(jx - 1)],
                 'peak_minus_best_within_1px': float(fine[iy, ix] - w1[jy, jx]), 'subpixel_peak_px': None if sub is None else [float(sub[0]), float(sub[1])]}
    v = res[name]
    lines.append('  %-28s peak r %+.4f at (%d, %d) px; r(0,0) %+.4f; best within 1 px %+.4f at (%d, %d); peak - that = %.4f; sub-pixel peak %s px'
                 % (name, v['peak_r'], v['peak_px'][0], v['peak_px'][1], v['r_zero'], v['best_within_1px_r'], v['best_within_1px_at'][0],
                    v['best_within_1px_at'][1], v['peak_minus_best_within_1px'], 'n/a' if sub is None else '(%+.2f, %+.2f)' % sub))
    print(lines[-1], flush=True)
json.dump(res, open(os.path.splitext(OUT)[0] + '.json', 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
