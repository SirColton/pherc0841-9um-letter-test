"""day1-render-labelcheck.py - are the labels on a render grid where the labels are? Two independent carries.

WHY. day1-render-9um.py brings ScrollPrize's republished labels (ink-labels/.../20260918, on the
segment's own 2.403 um canvas, 20 px per mesh cell) onto the render grid BY INDEX. The w00big labels
used so far for scoring came the other way: verify-nine-um-test-does-it-matter/labels-on-grid.py
carried the ink-dataset's v2 rasters by 3D nearest neighbour onto that dataset's own tifxyz and read the
raster at 20 x the cell index. That dataset's tifxyz has its own scale (meta "scale" 0.0496 for w00,
0.0498 for 252405, i.e. 20.14-20.18 and 20.07 raster px per cell), so a factor of exactly 20 would
drift with distance from the raster origin. This measures both carries against each other, on w00big
and on the 252405 frame, with a local shift scan, so a misplacement shows as a shift rather than as a
lower number nobody can interpret.

FOR EACH FRAME: grid points in the 2.403 um frame (the render's own bilinear mesh positions) ->
nearest point of the dataset's densified tifxyz (5 sub-samples per cell, labels-on-grid.py's) ->
raster px = cell x factor, factor either 20 (labels-on-grid.py's) or 1/meta scale; 4x4 mean.
Compared with the by-index republished grid: Pearson r and the shift (+-30 render px) that maximises it.

Usage: day1-render-labelcheck.py <out dir>     CPU only; local files; writes only into <out dir>.
"""
import json
import os
import sys
import time

import numpy as np
import tifffile
import zarr
from scipy.spatial import cKDTree

OUTD = sys.argv[1]
os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(OUTD, 'labelcheck-run.txt')
import runfile
runfile.claim(OUT)
lines = []


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def bil(A, ii, jj):
    i1 = np.clip(np.floor(ii).astype(int), 0, A.shape[0] - 2)
    j1 = np.clip(np.floor(jj).astype(int), 0, A.shape[1] - 2)
    di, dj = ii - i1, jj - j1
    return (A[i1, j1] * (1 - di) * (1 - dj) + A[i1 + 1, j1] * di * (1 - dj)
            + A[i1, j1 + 1] * (1 - di) * dj + A[i1 + 1, j1 + 1] * di * dj)


def grid_points(mesh, crop, pool=4):
    X, Y, Z = (tifffile.imread(os.path.join(mesh, k + '.tif')).astype(np.float64)[crop[0]:crop[1], crop[2]:crop[3]] for k in 'xyz')
    h, w = X.shape
    H, W = (h - 1) * 20 // pool, (w - 1) * 20 // pool
    ii = (np.arange(H) * pool + (pool - 1) / 2) / 20.0
    jj = (np.arange(W) * pool + (pool - 1) / 2) / 20.0
    JJ, II = np.meshgrid(jj, ii)
    return np.stack([bil(A, II, JJ) for A in (X, Y, Z)], -1), H, W


def by_index(path, crop, H, W, pool=4):
    cv = zarr.open_array(path, mode='r')[:]
    R0, C0 = crop[0] * 20, crop[2] * 20
    return cv[R0:R0 + H * pool, C0:C0 + W * pool].astype(np.float32).reshape(H, pool, W, pool).mean((1, 3)) / 255.0


def carry_nn(Q, dsdir, prefix, raster, factors, SUB=5):
    X, Y, Z = (tifffile.imread(os.path.join(dsdir, prefix + k + '.tif')).astype(np.float64) for k in 'xyz')
    lo, hi = Q.reshape(-1, 3).min(0) - 60, Q.reshape(-1, 3).max(0) + 60
    ok0 = (X > 0) & (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (Z >= lo[2]) & (Z <= hi[2])
    rr, cc = np.nonzero(ok0)
    r0, r1, c0, c1 = max(rr.min() - 2, 0), min(rr.max() + 3, X.shape[0]), max(cc.min() - 2, 0), min(cc.max() + 3, X.shape[1])
    rgr = np.arange(r0, r1 - 1, 1.0 / SUB)
    cgr = np.arange(c0, c1 - 1, 1.0 / SUB)
    CC, RR = np.meshgrid(cgr, rgr)
    i1, j1 = np.floor(RR).astype(int), np.floor(CC).astype(int)
    ok = (X[i1, j1] > 0) & (X[i1 + 1, j1] > 0) & (X[i1, j1 + 1] > 0) & (X[i1 + 1, j1 + 1] > 0)
    P = np.stack([bil(A, RR, CC)[ok] for A in (X, Y, Z)], 1)
    T = cKDTree(P)
    dist, idx = T.query(Q.reshape(-1, 3))
    dr, dc = RR[ok][idx].reshape(Q.shape[:2]), CC[ok][idx].reshape(Q.shape[:2])
    L = tifffile.imread(raster)
    L = L.reshape(L.shape[-2:]) if L.ndim > 2 else L
    out = {}
    for name, (fr, fc) in factors.items():
        R, C = dr * fr, dc * fc
        acc = np.zeros(Q.shape[:2], np.float32)
        for a in range(4):
            for b in range(4):
                ri = np.clip(np.rint(R - 1.5 + a).astype(int), 0, L.shape[0] - 1)
                ci = np.clip(np.rint(C - 1.5 + b).astype(int), 0, L.shape[1] - 1)
                acc += L[ri, ci] / 255.0
        out[name] = acc / 16
    return out, dist, L.shape, X.shape


def r_at(a, b, m, dy, dx):
    H, W = a.shape
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = m[ys:ye, xs:xe] & m[ys - dy:ye - dy, xs - dx:xe - dx]
    aa, bb = a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm]
    if aa.std() == 0 or bb.std() == 0:
        return -2.0
    return float(np.corrcoef(aa, bb)[0, 1])


def scan(a, b, m, R=30):
    """r at (0, 0), and the best shift: a step-3 pass over +-R, then step 1 within +-3 of its peak."""
    r00 = r_at(a, b, m, 0, 0)
    best = max((r_at(a, b, m, dy, dx), dy, dx) for dy in range(-R, R + 1, 3) for dx in range(-R, R + 1, 3))
    best = max((r_at(a, b, m, dy, dx), dy, dx) for dy in range(best[1] - 3, best[1] + 4) for dx in range(best[2] - 3, best[2] + 4))
    return r00, best


say('day1-render-labelcheck.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
res = {}
A = 'geom/l0-1203-render/audit/'
FR = {
    'w00big': dict(mesh='geom/c0841/mesh-l0/w00', crop=(380, 580, 500, 750),
                   rep='geom/day1-9um/render/labelcheck/w00-20260918/inklabels.zarr/0',
                   dsdir=A + 'sota-research/w00-dataset/', prefix='', raster=A + 'sota-research/w00-dataset/w00_inklabels_v2.tif',
                   meta=A + 'evidence-control-training-labels/w00_meta.json',
                   prior=A + 'verify-nine-um-test-does-it-matter/w00_inklabels_v2-grid.npy',
                   mask='geom/l0-1203-render/audit/build-deciding-test/ours-mask.npy'),
    '252405': dict(mesh='geom/day1-9um/render/src/mesh-2.403', crop=(42, 468, 210, 664),
                   rep='geom/day1-9um/render/src/labels-20260918/inklabels.zarr/0',
                   dsdir=A + 'design-use-what-exists/seg/', prefix='auto_grown_20260220174252405_',
                   raster=A + 'design-use-what-exists/seg/auto_grown_20260220174252405_inklabels_v2.tif',
                   meta=A + 'design-use-what-exists/seg/auto_grown_20260220174252405_meta.json', prior=None,
                   mask='geom/day1-9um/render/252405-mask.npy'),
}
for fn, f in FR.items():
    Q, H, W = grid_points(f['mesh'], f['crop'])
    rep = by_index(f['rep'], f['crop'], H, W)
    sc = json.load(open(f['meta']))['scale']
    facs = {'x20': (20.0, 20.0), 'x1/scale': (1 / sc[1], 1 / sc[0])}
    car, dist, lshape, dshape = carry_nn(Q, f['dsdir'], f['prefix'], f['raster'], facs)
    m = np.load(f['mask']) if os.path.exists(f['mask']) else np.ones((H, W), bool)
    say('')
    say('  %s: grid %dx%d; dataset tifxyz %s, raster %s (%.3f x %.3f per cell), meta scale %s -> 1/scale %.3f x %.3f'
        % (fn, H, W, dshape, lshape, lshape[0] / dshape[0], lshape[1] / dshape[1], sc, 1 / sc[1], 1 / sc[0]))
    say('    grid -> dataset surface, nearest distance (2.403 um voxels): median %.2f p95 %.2f max %.2f' % (np.median(dist), np.percentile(dist, 95), dist.max()))
    say('    republished by index: ink frac (>=0.75) %.4f' % (rep[m] >= 0.75).mean())
    rr = {}
    cand = dict(car)
    if f['prior'] and os.path.exists(f['prior']):
        cand['prior grid (labels-on-grid.py)'] = np.load(f['prior'])
    for k, g in cand.items():
        r00, best = scan(g, rep, m)
        rr[k] = {'r0': r00, 'best_r': best[0], 'best_shift_px': [best[1], best[2]]}
        say('    %-32s vs republished-by-index: r at (0,0) %+.3f; best %+.3f at shift (%d, %d) render px (%.0f, %.0f um)'
            % (k, r00, best[0], best[1], best[2], best[1] * 9.612, best[2] * 9.612))
    res[fn] = rr
json.dump(res, open(os.path.join(OUTD, 'labelcheck.json'), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines))
