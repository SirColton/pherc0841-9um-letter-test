"""day1-render-labelalign.py - which label carry sits where the 2.4 um ink is? Decided on w00big, applied to 252405.

WHY. day1-render-labelcheck.py found that the three ways of putting the same human labels on a render
grid disagree by 4-21 render px (40-200 um), with near-perfect correlation once shifted, so each is a
shifted copy of the others: (a) ScrollPrize's republished labels read by index (their transfer carries
a canvas_offset_yx of about 30 and 50-80 raster px), (b) the ink-dataset raster by 3D nearest
neighbour at 20 px per cell (labels-on-grid.py, the w00big scoring so far), (c) the same at 1/scale
px per cell. A stroke is about 0.3-0.5 mm wide, so 0.1-0.2 mm decides an AUC.

THE RULE, written before this runs: on w00big (the selection frame, not held out) each label grid is
shift-scanned against two 2.4 um maps that sit on the bucket mesh's canvas by construction - our
Hecate 2.4 rev render (c0841's) and ScrollPrize's w00 ink map (its ds8 jpg, as c0841 maps it). The
carry whose peak is within 1 px of (0, 0) against both is the right one; it is then used for 252405
unchanged. On 252405 ScrollPrize's 2.4 um map is shift-scanned against the delivered labels as a
CONFIRMATION only: no 9 um model is involved and nothing is chosen there.

Usage: day1-render-labelalign.py <out dir>      CPU only; local files; writes only into <out dir>.
"""
import json
import os
import sys
import time

import numpy as np
import tifffile
import zarr
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None
OUTD = sys.argv[1]
os.makedirs(OUTD, exist_ok=True)
OUT = os.path.join(OUTD, 'labelalign-run.txt')
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


def carry_nn(Q, dsdir, prefix, raster, fr, fc, SUB=5):
    X, Y, Z = (tifffile.imread(os.path.join(dsdir, prefix + k + '.tif')).astype(np.float64) for k in 'xyz')
    lo, hi = Q.reshape(-1, 3).min(0) - 60, Q.reshape(-1, 3).max(0) + 60
    ok0 = (X > 0) & (X >= lo[0]) & (X <= hi[0]) & (Y >= lo[1]) & (Y <= hi[1]) & (Z >= lo[2]) & (Z <= hi[2])
    rr, cc = np.nonzero(ok0)
    r0, r1, c0, c1 = max(rr.min() - 2, 0), min(rr.max() + 3, X.shape[0]), max(cc.min() - 2, 0), min(cc.max() + 3, X.shape[1])
    CC, RR = np.meshgrid(np.arange(c0, c1 - 1, 1.0 / SUB), np.arange(r0, r1 - 1, 1.0 / SUB))
    i1, j1 = np.floor(RR).astype(int), np.floor(CC).astype(int)
    ok = (X[i1, j1] > 0) & (X[i1 + 1, j1] > 0) & (X[i1, j1 + 1] > 0) & (X[i1 + 1, j1 + 1] > 0)
    P = np.stack([bil(A, RR, CC)[ok] for A in (X, Y, Z)], 1)
    dist, idx = cKDTree(P).query(Q.reshape(-1, 3))
    dr, dc = RR[ok][idx].reshape(Q.shape[:2]), CC[ok][idx].reshape(Q.shape[:2])
    L = tifffile.imread(raster)
    L = L.reshape(L.shape[-2:]) if L.ndim > 2 else L
    acc = np.zeros(Q.shape[:2], np.float32)
    for a in range(4):
        for b in range(4):
            ri = np.clip(np.rint(dr * fr - 1.5 + a).astype(int), 0, L.shape[0] - 1)
            ci = np.clip(np.rint(dc * fc - 1.5 + b).astype(int), 0, L.shape[1] - 1)
            acc += L[ri, ci] / 255.0
    return acc / 16, dist.reshape(Q.shape[:2])


def r_at(a, b, m, dy, dx):
    H, W = a.shape
    ys, ye = max(0, dy), min(H, H + dy)
    xs, xe = max(0, dx), min(W, W + dx)
    mm = m[ys:ye, xs:xe] & m[ys - dy:ye - dy, xs - dx:xe - dx]
    aa, bb = a[ys:ye, xs:xe][mm], b[ys - dy:ye - dy, xs - dx:xe - dx][mm]
    if aa.std() == 0 or bb.std() == 0:
        return -2.0
    return float(np.corrcoef(aa, bb)[0, 1])


def scan(mp, lab, m, R=30):
    """Shift of the LABEL grid relative to the map: r(map(y, x), label(y - dy, x - dx)); step 3 then step 1."""
    r00 = r_at(mp, lab, m, 0, 0)
    best = max((r_at(mp, lab, m, dy, dx), dy, dx) for dy in range(-R, R + 1, 3) for dx in range(-R, R + 1, 3))
    best = max((r_at(mp, lab, m, dy, dx), dy, dx) for dy in range(best[1] - 3, best[1] + 4) for dx in range(best[2] - 3, best[2] + 4))
    return r00, best


def auc(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]])
    rk = v.argsort().argsort() + 1.0
    n1, n0 = pos.sum(), neg.sum()
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


say('day1-render-labelalign.py  %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
A = 'geom/l0-1203-render/audit/'
E = 40
res = {'w00big': {}, '252405': {}}

# ---- w00big: the decision -------------------------------------------------------------------------
crop = (380, 580, 500, 750)
Q, H, W = grid_points('geom/c0841/mesh-l0/w00', crop)
P4 = 4
mask = np.load(A + 'build-deciding-test/ours-mask.npy')
m24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/mask.png'), np.float32)[:H * P4, :W * P4].reshape(H, P4, W, P4).min((1, 3)) > 0
M = mask & m24
M[:E] = M[-E:] = False
M[:, :E] = M[:, -E:] = False
h24 = np.asarray(Image.open('geom/c0841/mesh0b-render/w00big/hecate-rev.png'), np.float32)[:H * P4, :W * P4].reshape(H, P4, W, P4).mean((1, 3))
ink = np.asarray(Image.open('geom/c0841/ink-20260220213127.jpg').convert('L'), np.float32)
gi = 2.5 * (380 + (np.arange(H) * P4 + (P4 - 1) / 2) / 20.0) - 0.5
gj = 2.5 * (500 + (np.arange(W) * P4 + (P4 - 1) / 2) / 20.0) - 0.5
GJ, GI = np.meshgrid(gj, gi)
spink = map_coordinates(ink, [GI, GJ], order=1)
sc = json.load(open(A + 'evidence-control-training-labels/w00_meta.json'))['scale']
labs = {'republished, by index': by_index('geom/day1-9um/render/labelcheck/w00-20260918/inklabels.zarr/0', crop, H, W),
        'prior grid (labels-on-grid.py, NN x20)': np.load(A + 'verify-nine-um-test-does-it-matter/w00_inklabels_v2-grid.npy')}
labs['NN x1/scale'], _ = carry_nn(Q, A + 'sota-research/w00-dataset/', '', A + 'sota-research/w00-dataset/w00_inklabels_v2.tif', 1 / sc[1], 1 / sc[0])
sup = by_index('geom/day1-9um/render/labelcheck/w00-20260918/supervision.zarr/0', crop, H, W)
say('')
say('  w00big (selection frame): %d px; label grid shift that best matches each 2.4 um map (s2 smoothing), render px of 9.61 um' % M.sum())
for mn, mp in (('2.4 um Hecate rev (ours)', h24), ('ScrollPrize 2.4 um map (ds8 jpg)', spink)):
    mps = gaussian_filter(mp, 2)
    for ln, lg in labs.items():
        r00, best = scan(mps, gaussian_filter(lg, 2), M)
        pos = M & (lg >= 0.75)
        neg = M & (lg <= 0)
        res['w00big']['%s | %s' % (mn, ln)] = {'r0': r00, 'best_r': best[0], 'shift': [best[1], best[2]], 'auc_all_neg': auc(mp, pos, neg)}
        say('    %-34s vs %-40s r(0,0) %+.3f  peak %+.3f at (%+d, %+d)  | AUC of the map at (0,0), all negatives %.3f'
            % (mn, ln, r00, best[0], best[1], best[2], auc(mp, pos, neg)))
ok = {}
for ln in labs:
    sh = [res['w00big']['%s | %s' % (mn, ln)]['shift'] for mn in ('2.4 um Hecate rev (ours)', 'ScrollPrize 2.4 um map (ds8 jpg)')]
    ok[ln] = all(abs(s[0]) <= 1 and abs(s[1]) <= 1 for s in sh)
say('  RULE -> within 1 px of (0, 0) against both maps: %s' % ', '.join('%s: %s' % (k, 'YES' if v else 'no') for k, v in ok.items()))

# ---- 252405: confirmation only --------------------------------------------------------------------
RD = 'geom/day1-9um/render/'
crop5 = (42, 468, 210, 664)
Q5, H5, W5 = grid_points(RD + 'src/mesh-2.403', crop5)
mask5 = np.load(RD + '252405-mask.npy')
M5 = mask5.copy()
M5[:E] = M5[-E:] = False
M5[:, :E] = M5[:, -E:] = False
sp5 = np.load(RD + '252405-spmap-grid.npy')
sc5 = json.load(open(A + 'design-use-what-exists/seg/auto_grown_20260220174252405_meta.json'))['scale']
labs5 = {'republished, by index': np.load(RD + '252405-inklabels-grid.npy')}
labs5['NN x1/scale'], d5 = carry_nn(Q5, A + 'design-use-what-exists/seg/', 'auto_grown_20260220174252405_',
                                    A + 'design-use-what-exists/seg/auto_grown_20260220174252405_inklabels_v2.tif', 1 / sc5[1], 1 / sc5[0])
labs5['NN x20'], _ = carry_nn(Q5, A + 'design-use-what-exists/seg/', 'auto_grown_20260220174252405_',
                              A + 'design-use-what-exists/seg/auto_grown_20260220174252405_inklabels_v2.tif', 20.0, 20.0)
say('')
say('  252405 (held out; CONFIRMATION only, ScrollPrize\'s map is not a model run here): %d px' % M5.sum())
sps = gaussian_filter(sp5, 2)
for ln, lg in labs5.items():
    r00, best = scan(sps, gaussian_filter(lg, 2), M5)
    res['252405']['ScrollPrize 2.4 um map | %s' % ln] = {'r0': r00, 'best_r': best[0], 'shift': [best[1], best[2]]}
    say('    ScrollPrize 2.4 um map (full res) vs %-28s r(0,0) %+.3f  peak %+.3f at (%+d, %+d)' % (ln, r00, best[0], best[1], best[2]))
np.save(os.path.join(OUTD, '252405-inklabels-NNx1scale-grid.npy'), labs5['NN x1/scale'])
json.dump(res, open(os.path.join(OUTD, 'labelalign.json'), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines))
