"""day1-render-gate2.py - sanity gate 2 for a day1-render-9um.py frame: does released ink_9um s43-75k,
run on the native render, line up with ScrollPrize's 2.4 um map to within 1 px?

WHY. Gate 2 of the day-1 plan's step 3 (geom/day1-9um/design.json, "choice"): before the frozen recipe
reads auto_grown_20260220174252405, the registration must put the 9 um read where the 2.4 um map has
its ink. On w00big, c0841-9um.py's local shift scan of ink_9um rev peaked at (0, 0) on an 8 px step;
this scans every 1 px, so "within 1 px" is decidable.

WHAT IT RUNS, and nothing else: ONE model, the released seed43-step-075000, with c0841-9um.py's recipe
(21 layers about the stack centre, stored order reversed = 'rev', layer windows 0-17, 2-19, 4-21 run
through koine's infer.py and averaged). No other model and no other setting touches this frame here:
it is the held-out test and nothing may be tuned on it. The map is saved for the plan's step 4, which
reports this same released baseline.

WHAT IT REPORTS: the peak of Pearson r (both maps Gaussian sigma 4 px, c0841's s4) over shifts of
-12..+12 px in both axes, and whether it is within 1 px of (0, 0); plus c0841's coarse scan (+-40 px,
step 8) for comparison with w00big. The scan is over the whole rendered frame minus a 40 px edge.
It refuses to run until <baselines>/DONE exists, so it never shares the GPU with that part.

Usage: polite-run.py day1-render-gate2.py <render dir> <name> [<baselines dir>]
GPU only through polite-run.py. Reads local files; writes only into <render dir>.
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

KOINE = os.path.abspath('koine/ink-detection')
sys.path.insert(0, KOINE)
CKPT = 'geom/ink9um/seed43-step-075000.pth'
WINDOWS = ((0, 17), (2, 19), (4, 21))
CH = 128
RD, NAME = sys.argv[1], sys.argv[2]
BASE = sys.argv[3] if len(sys.argv) > 3 else 'geom/day1-9um/baselines'
OUT = os.path.join(RD, '%s-gate2-run.txt' % NAME)
import runfile
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


say('day1-render-gate2.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
done = os.path.join(BASE, 'DONE')
if not os.path.exists(done):
    say('  %s is absent: the baselines part has not finished its GPU runs. Not run.' % done)
    runfile.done(OUT, '\n'.join(lines))
    sys.exit(3)
say('  %s present, mtime %s' % (done, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(done)))))
for f in sorted(os.listdir(BASE)):
    p = os.path.join(BASE, f)
    if os.path.isfile(p) and (f.endswith('.json') or f == 'DONE'):
        say('    baselines/%s  sha1 %s  mtime %s' % (f, hashlib.sha1(open(p, 'rb').read()).hexdigest()[:12],
                                                time.strftime('%H:%M:%S', time.localtime(os.path.getmtime(p)))))

import tifffile
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter
from koine_machines.inference import infer

stack = np.load(os.path.join(RD, '%s-stack.npy' % NAME), mmap_mode='r')
mask = np.load(os.path.join(RD, '%s-mask.npy' % NAME))
ref = np.load(os.path.join(RD, '%s-spmap-grid.npy' % NAME)).astype(np.float32) * 255
c0 = stack.shape[0] // 2
H, W = mask.shape
say('  stack %s, centre index %d, checkpoint %s (sha1 %s)' % (stack.shape, c0, CKPT, hashlib.sha1(open(CKPT, 'rb').read()).hexdigest()[:12]))
cp = os.path.join(RD, '%s-ink9-s43-75k-c+0-rev.npy' % NAME)
if os.path.exists(cp):
    p = np.load(cp)
    say('  map: cached %s' % cp)
else:
    t1 = time.time()
    s = np.ascontiguousarray(np.asarray(stack[c0 - 10:c0 + 11])[::-1])
    tmp = os.path.abspath(os.path.join(RD, 'tmp-gate2'))
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    zpath = os.path.join(tmp, 'in.zarr')
    z = zarr.open(zpath, mode='w', shape=s.shape, chunks=(s.shape[0], CH, CH), dtype='u1', zarr_format=2)
    z[:] = s
    preds = []
    for (a, b) in WINDOWS:
        tif = os.path.join(tmp, 'pred-%d_%d.tif' % (a, b))
        infer.main([zpath, os.path.abspath(CKPT), tif, '--layer-start', str(a), '--layer-end', str(b),
                    '--no-compile', '--num-workers', '0'])
        q = tifffile.imread(tif).astype(np.float32)
        q = q[0] if q.ndim == 3 else q
        if q.shape != s.shape[1:]:
            q = np.asarray(Image.fromarray(q).resize(s.shape[1:][::-1], Image.LANCZOS), np.float32)
        preds.append(q)
    p = np.mean(preds, axis=0).astype(np.float32)
    np.save(cp, p)
    shutil.rmtree(tmp, ignore_errors=True)
    say('  ran ink_9um s43-75k rev c+0, 3 windows, in %.0f s; map mean %.1f sd %.1f' % (time.time() - t1, p[mask].mean(), p[mask].std()))

E = 40
M = mask.copy()
M[:E] = M[-E:] = False
M[:, :E] = M[:, -E:] = False
ps, rs = gaussian_filter(p, 4), gaussian_filter(ref, 4)


def r_at(dy, dx):
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    aa = ps[ys:ye, xs:xe]
    bb = rs[ys - dy:ye - dy, xs - dx:xe - dx]
    mm = M[ys:ye, xs:xe] & M[ys - dy:ye - dy, xs - dx:xe - dx]
    return float(np.corrcoef(aa[mm], bb[mm])[0, 1])


fine = np.array([[r_at(dy, dx) for dx in range(-12, 13)] for dy in range(-12, 13)])
iy, ix = np.unravel_index(np.nanargmax(fine), fine.shape)
pk = (iy - 12, ix - 12)
ok = abs(pk[0]) <= 1 and abs(pk[1]) <= 1
say('  fine local shift scan (s4, -12..+12 px, step 1 = 9.61 um), %d px: peak r %+.3f at (%d, %d) px; r at (0, 0) %+.3f'
    % (M.sum(), fine[iy, ix], pk[0], pk[1], fine[12, 12]))
say('    r along dy at dx=0: %s' % ' '.join('%+.3f' % v for v in fine[:, 12][6:19]))
say('    r along dx at dy=0: %s' % ' '.join('%+.3f' % v for v in fine[12, :][6:19]))
coarse = [(r_at(dy, dx), dy, dx) for dy in range(-40, 41, 8) for dx in range(-40, 41, 8)]
cpk = max(coarse)
say('  coarse scan as c0841 (s4, +-40 px, step 8): peak r %+.3f at (%d, %d) px  (w00big ink9 ours-c+0-rev: +0.481 at (0, 0))' % cpk)
say('  GATE 2 %s: peak %s 1 px of (0, 0)' % ('PASS' if ok else 'FAIL', 'within' if ok else 'NOT within'))
json.dump({'peak_r': float(fine[iy, ix]), 'peak_shift_px': [int(pk[0]), int(pk[1])], 'r_zero': float(fine[12, 12]),
           'coarse_peak': [float(cpk[0]), cpk[1], cpk[2]], 'pass': bool(ok), 'checkpoint': CKPT},
          open(os.path.join(RD, '%s-gate2.json' % NAME), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines))
