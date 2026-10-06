"""student-train-synth.py - a tiny SYNTHETIC render in day1-render-9um.py's on-disk format, for student-train.py's smoke test.

WHY. The training harness (student-train.py) reads each segment as day1-render-9um.py writes it:
<prefix>-stack.npy (41, H, W) uint8, <prefix>-mask.npy (H, W) bool, <prefix>-spmap-grid.npy (H, W) float32
(the teacher by index, canvas block mean / 255), <prefix>-grid.json (crop_cells, H, W, NL, stored_along, ...).
Before the real renders of w00 and 552896 exist, its multi-segment path is exercised on w00big (real) plus
this synthetic stand-in for 552896. The stack and mask are SYNTHETIC; the teacher grid is the real 552896
teacher by index (the frozen tif), so the harness's teacher re-derivation and hash checks run on it as they
will on the real render. grid.json carries "synthetic": true and student-train.py refuses it without --smoke.

Writes only under geom/student-9um/harness/. CPU only, no network, no model, no 252405 file.
Usage: student-train-synth.py [--out geom/student-9um/harness/smoke/synth] [--cells 60]
"""
import argparse
import hashlib
import json
import os

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

ap = argparse.ArgumentParser()
ap.add_argument('--out', default='geom/student-9um/harness/smoke/synth')
ap.add_argument('--cells', type=int, default=60)
ap.add_argument('--seed', type=int, default=7)
a = ap.parse_args()
assert os.path.abspath(a.out).replace('\\', '/').find('/geom/student-9um/harness/') > 0, 'write only under geom/student-9um/harness/'
os.makedirs(a.out, exist_ok=True)

SEG = '552896'
TIF = ('geom/student-9um/freeze/teacher/PHerc0841-20260220214732-2.403um-0.22m-77keV-volume-20260319124803-'
       '20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif')
allowed = np.load('geom/student-9um/freeze/geom/%s-allowed.npy' % SEG)
n = a.cells
# the n x n cell block with the most allowed cells (stride 10), so valid 128-px windows exist in it
best, rc = -1, None
for r in range(0, allowed.shape[0] - n, 10):
    for c in range(0, allowed.shape[1] - n, 10):
        s = int(allowed[r:r + n, c:c + n].sum())
        if s > best:
            best, rc = s, (r, c)
r0, c0 = rc
r1, c1 = r0 + n, c0 + n
H, W, NL, POOL = 5 * (n - 1), 5 * (n - 1), 41, 4
cv = tifffile.imread(TIF)
cv = cv.reshape(cv.shape[-2:]) if cv.ndim > 2 else cv
blk = cv[20 * r0:20 * r0 + H * POOL, 20 * c0:20 * c0 + W * POOL]
grid = blk.astype(np.float32).reshape(H, POOL, W, POOL).mean((1, 3)) / 255.0
del cv, blk
rng = np.random.default_rng(a.seed)
# synthetic papyrus: smooth noise about 90, a brighter sheet at layers 17-23, and a weak copy of the teacher
# at layers 19-21 so a model has something to fit; it is NOT a render of anything
base = gaussian_filter(rng.normal(0, 12, (NL, H, W)).astype(np.float32), (1, 2, 2)) + 90
prof = np.exp(-0.5 * ((np.arange(NL) - 20) / 3.0) ** 2).astype(np.float32)[:, None, None]
stack = base + 25 * prof + 15 * prof * grid[None]
stack = np.clip(np.rint(stack), 0, 255).astype(np.uint8)
mask = np.zeros((H, W), bool)
mask[2:-2, 2:-2] = True
stack[:, ~mask] = 0
p = os.path.join(a.out, SEG)
np.save(p + '-stack.npy', stack)
np.save(p + '-mask.npy', mask)
np.save(p + '-spmap-grid.npy', grid)
json.dump({'synthetic': True, 'mesh': 'SYNTHETIC (stack and mask); teacher grid = the real 552896 teacher by index',
           'crop_cells': [r0, r1, c0, c1], 'canvas_origin_px': [20 * r0, 20 * c0], 'pool': POOL, 'H': H, 'W': W, 'NL': NL,
           'pitch_um_in_plane': 2.403 * POOL, 'pitch_um_depth': 9.6, 'stored_along': 'plus normal', 'flipped': False,
           'orient': {}, 'centre_layer': NL // 2, 'grids': ['spmap'], 'grid_units': 'canvas value / 255, block mean',
           'stack_sha1': hashlib.sha1(stack.tobytes()).hexdigest()[:12]},
          open(p + '-grid.json', 'w'), indent=1)
print('wrote %s-{stack,mask,spmap-grid}.npy and -grid.json: cells rows [%d, %d) cols [%d, %d), %d allowed of %d; render %dx%d x %d'
      % (p, r0, r1, c0, c1, best, n * n, H, W, NL))
