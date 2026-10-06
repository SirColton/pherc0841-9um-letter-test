"""student-renders-labels.py - the label grids for the student's native 9.366 um training renders
(part renders of geom/student-9um/prereg-student.json, sha256 f3ebf0ab...).

WHY. The renders part carries the frozen teacher and the labels onto each training render grid. The
teacher (ScrollPrize's 2.4 um map) and w00's S3 20260918 inklabels / supervision come onto the grid inside
student-render-9um.py (a byte copy of day1-render-9um.py) by its --canvas rule, exactly as the prereg's
command lines say. What that rule cannot carry is ScrollPrize's HF v2 set: those rasters sit on the HF
dataset's own tifxyz, not on the mesh's 2.403 um canvas. This carries them by day1-labels-hf.py's
convention (labels-on-grid.py's), unchanged:
  render grid point (i, j) = bilinear position on the render mesh at ((4 i + 1.5) / 20, (4 j + 1.5) / 20)
  cells -> 3D nearest point of the HF tifxyz densified 5 sub-samples per cell (a sub-sample kept only when
  its four corner cells are valid) -> raster px = that sub-sample's (row, col) x 20 -> mean of the 4 x 4
  raster px at rint(R - 1.5 + a), rint(C - 1.5 + b), / 255. No distance limit; the distance is saved.
The dataset window is day1-labels-hf.py's: dataset cells inside the bounding box of the masked grid
points +- 60 voxels (+- 2 cells). The one change is mechanical: the grid points are built and queried in
row blocks so a full segment never needs every temporary at once; a nearest-neighbour query and a
per-pixel gather do not depend on how the points are batched.
Also carried BY INDEX with day1-render-9um.py's --canvas rule (same code lines): the S3 20260918 grids the
render command did not already carry (w00 validation; 552896 inklabels, supervision, validation).

CONVENTION CHECK (w00 only): the w00big window of the full-w00 HF grids (render rows [1900, 2895), cols
[2500, 3745)) is compared element by element with verify-nine-um-test-does-it-matter/w00_*_v2-grid.npy,
the grids day1-labels-hf.py rebuilt and matched.

Never opens a 252405 file. CPU only, no network. Usage: student-renders-labels.py <w00|552896>.
Run from <workdir>; writes only into geom/student-9um/renders/.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import tifffile
import zarr
from scipy.spatial import cKDTree

import runfile

SEG = sys.argv[1]
OUTD = 'geom/student-9um/renders'
OUT = os.path.join(OUTD, '%s-labels-run.txt' % SEG)
runfile.claim(OUT)
lines = []
POOL, SUB, BLOCK = 4, 5, 256
A = 'geom/l0-1203-render/audit/'
CFG = {
    'w00': {'mesh': 'geom/c0841/mesh-l0/w00', 'crop': (0, 823, 0, 928),
            'hf_dir': A + 'sota-research/w00-dataset/', 'hf_prefix': 'w00_',
            'hf_xyz': {k: A + 'sota-research/w00-dataset/%s.tif' % k for k in 'xyz'},
            's3_by_index': {'validation': 'geom/day1-9um/render/labelcheck/w00-20260918/validation.zarr/0'}},
    '552896': {'mesh': 'geom/day1-9um/gates/s3/20260220214732-auto_grown_20260220144552896/mesh/20260220214732-on-20260319124803-2.403um.tifxyz',
               'crop': (0, 904, 0, 920),
               'hf_dir': A + 'design-use-what-exists/seg/', 'hf_prefix': 'auto_grown_20260220144552896_',
               'hf_xyz': {k: A + 'design-use-what-exists/seg/auto_grown_20260220144552896_%s.tif' % k for k in 'xyz'},
               's3_by_index': {k: 'geom/day1-9um/gates/s3/20260220214732-auto_grown_20260220144552896/ink-labels/'
                                  '2.403um-volume-20260319124803/20260918/%s.zarr/0' % k for k in ('inklabels', 'supervision', 'validation')}},
}[SEG]
HFR = {'inklabels_v2': 'inklabels_v2.tif', 'supervision_mask_v2': 'supervision_mask_v2.tif', 'validation_mask_v2': 'validation_mask_v2.tif'}


def say(s=''):
    lines.append(s)
    print(s, flush=True)


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def bil(A_, ii, jj):
    i1 = np.clip(np.floor(ii).astype(int), 0, A_.shape[0] - 2)
    j1 = np.clip(np.floor(jj).astype(int), 0, A_.shape[1] - 2)
    di, dj = ii - i1, jj - j1
    return (A_[i1, j1] * (1 - di) * (1 - dj) + A_[i1 + 1, j1] * di * (1 - dj)
            + A_[i1, j1 + 1] * (1 - di) * dj + A_[i1 + 1, j1 + 1] * di * dj)


def tif2(p):
    a = tifffile.imread(p)
    return a.reshape(a.shape[-2:]) if a.ndim > 2 else a


assert not any('252405' in str(v) for v in (CFG['mesh'], CFG['hf_dir'], CFG['hf_prefix']))
say('student-renders-labels.py %s  %s' % (SEG, time.strftime('%Y-%m-%d %H:%M:%S')))
say('  prereg-student.json sha256 %s' % sha256('geom/student-9um/prereg-student.json'))
gj = json.load(open(os.path.join(OUTD, '%s-grid.json' % SEG)))
mask = np.load(os.path.join(OUTD, '%s-mask.npy' % SEG))
H, W = mask.shape
assert (H, W) == (gj['H'], gj['W']) and list(gj['crop_cells']) == list(CFG['crop']) and gj['mesh'] == CFG['mesh'], gj
r0c, r1c, c0c, c1c = CFG['crop']
X, Y, Z = (tifffile.imread(os.path.join(CFG['mesh'], k + '.tif')).astype(np.float64)[r0c:r1c, c0c:c1c] for k in 'xyz')
h, w = X.shape
assert (H, W) == ((h - 1) * 20 // POOL, (w - 1) * 20 // POOL)
ii_all = (np.arange(H) * POOL + (POOL - 1) / 2) / 20.0
jj_all = (np.arange(W) * POOL + (POOL - 1) / 2) / 20.0


def grid_block(i0, i1):
    JJ, II = np.meshgrid(jj_all, ii_all[i0:i1])
    return np.stack([bil(A_, II, JJ) for A_ in (X, Y, Z)], -1)


# ---- HF v2 by the day1-labels-hf.py convention ----------------------------------------------------------
src = {}
for k in 'xyz':
    src[k + '.tif'] = {'path': CFG['hf_xyz'][k], 'sha256': sha256(CFG['hf_xyz'][k])}
ras = {}
for name, short in HFR.items():
    p = CFG['hf_dir'] + CFG['hf_prefix'] + short
    src[short] = {'path': p, 'sha256': sha256(p), 'bytes': os.path.getsize(p)}
    ras[name] = tif2(p)
Xh, Yh, Zh = (tifffile.imread(CFG['hf_xyz'][k]).astype(np.float64) for k in 'xyz')
say('  HF tifxyz %s, rasters %s (%.3f x %.3f px per cell; factor 20, the convention)'
    % (Xh.shape, ras['inklabels_v2'].shape, ras['inklabels_v2'].shape[0] / Xh.shape[0], ras['inklabels_v2'].shape[1] / Xh.shape[1]))
for k, v in src.items():
    say('    %-26s sha256 %s  %s' % (k, v['sha256'][:16], v['path']))

# window: bounding box of the masked grid points +- 60 voxels (day1-labels-hf.py's carry)
lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
for i0 in range(0, H, BLOCK):
    q = grid_block(i0, min(H, i0 + BLOCK))[mask[i0:i0 + BLOCK]]
    if len(q):
        lo, hi = np.minimum(lo, q.min(0)), np.maximum(hi, q.max(0))
lo, hi = lo - 60, hi + 60
ok0 = (Xh > 0) & (Xh >= lo[0]) & (Xh <= hi[0]) & (Yh >= lo[1]) & (Yh <= hi[1]) & (Zh >= lo[2]) & (Zh <= hi[2])
rr, cc = np.nonzero(ok0)
r0, r1, c0, c1 = max(rr.min() - 2, 0), min(rr.max() + 3, Xh.shape[0]), max(cc.min() - 2, 0), min(cc.max() + 3, Xh.shape[1])
RR_, CC_ = np.arange(r0, r1 - 1, 1.0 / SUB), np.arange(c0, c1 - 1, 1.0 / SUB)
CC, RR = np.meshgrid(CC_, RR_)
i1_, j1_ = np.floor(RR).astype(int), np.floor(CC).astype(int)
okd = (Xh[i1_, j1_] > 0) & (Xh[i1_ + 1, j1_] > 0) & (Xh[i1_, j1_ + 1] > 0) & (Xh[i1_ + 1, j1_ + 1] > 0)
del i1_, j1_
P = np.stack([bil(A_, RR, CC)[okd] for A_ in (Xh, Yh, Zh)], 1)
dr, dc = RR[okd], CC[okd]
NPTS = len(P)
del RR, CC, okd
t1 = time.time()
T = cKDTree(P)
say('  dataset window rows %d-%d cols %d-%d of %s, %d densified points; KD tree in %.0f s' % (r0, r1, c0, c1, Xh.shape, len(P), time.time() - t1))
grids = {n: np.zeros((H, W), np.float32) for n in HFR}
dist = np.zeros((H, W), np.float32)
edge = np.zeros((H, W), bool)
t1 = time.time()
for i0 in range(0, H, BLOCK):
    i1 = min(H, i0 + BLOCK)
    Q = grid_block(i0, i1)
    d, idx = T.query(Q.reshape(-1, 3), workers=4)
    shp = Q.shape[:2]
    Rr, Cr = dr[idx].reshape(shp) * 20, dc[idx].reshape(shp) * 20
    edge[i0:i1] = ((dr[idx] <= r0 + 1e-6) | (dr[idx] >= r1 - 1 - 1.0 / SUB - 1e-6) | (dc[idx] <= c0 + 1e-6) | (dc[idx] >= c1 - 1 - 1.0 / SUB - 1e-6)).reshape(shp)
    dist[i0:i1] = d.reshape(shp)
    for name, L in ras.items():
        acc = np.zeros(shp, np.float32)
        for a in range(4):
            for b in range(4):
                ri = np.clip(np.rint(Rr - 1.5 + a).astype(int), 0, L.shape[0] - 1)
                ci = np.clip(np.rint(Cr - 1.5 + b).astype(int), 0, L.shape[1] - 1)
                acc += L[ri, ci] / 255.0
        grids[name][i0:i1] = acc / 16
say('  carried %dx%d grid points in %.0f s' % (H, W, time.time() - t1))
del T, P, dr, dc
dm = dist[mask]
say('  grid -> HF surface nearest distance over the render mask (2.403 um voxels): median %.2f p95 %.2f p99 %.2f max %.2f; > 10 vox %d px, > 20 vox %d px; NN on the window edge %d px'
    % (np.median(dm), np.percentile(dm, 95), np.percentile(dm, 99), dm.max(), (dm > 10).sum(), (dm > 20).sum(), int((edge & mask).sum())))
out_sha = {}
p = os.path.join(OUTD, '%s-hf-nn-dist.npy' % SEG)
np.save(p, dist)
out_sha[os.path.basename(p)] = sha256(p)
cm2 = (2.403 * POOL * 1e-4) ** 2
stats = {}
for name, g in grids.items():
    p = os.path.join(OUTD, '%s-%s-grid.npy' % (SEG, name))
    np.save(p, g)
    out_sha[os.path.basename(p)] = sha256(p)
    stats[name] = {'mean_in_mask': float(g[mask].mean()), 'px_ge_075': int((mask & (g >= 0.75)).sum()), 'px_le_0': int((mask & (g <= 0)).sum())}
    say('    %-20s over the render mask: mean %.4f, >= 0.75 %d px (%.3f cm2), <= 0 %d px' % (name, g[mask].mean(), (mask & (g >= 0.75)).sum(), (mask & (g >= 0.75)).sum() * cm2, (mask & (g <= 0)).sum()))
lab, sup = grids['inklabels_v2'], grids['supervision_mask_v2']
pos, neg = mask & (lab >= 0.75), mask & (lab <= 0) & (sup >= 0.75)
say('  HF v2 classes on the render mask: pos (label >= 0.75) %d px = %.3f cm2; neg(sup) (label <= 0 & supervision >= 0.75) %d px = %.3f cm2; supervised-or-ink %d px = %.3f cm2'
    % (pos.sum(), pos.sum() * cm2, neg.sum(), neg.sum() * cm2, (mask & ((sup >= 0.75) | (lab >= 0.75))).sum(), (mask & ((sup >= 0.75) | (lab >= 0.75))).sum() * cm2))
stats['classes'] = {'pos': int(pos.sum()), 'neg_sup': int(neg.sum()), 'supervised_or_ink': int((mask & ((sup >= 0.75) | (lab >= 0.75))).sum())}

conv = {}
if SEG == 'w00':
    V = A + 'verify-nine-um-test-does-it-matter/'
    WR, WC = (1900, 2895), (2500, 3745)
    for name in ('inklabels_v2', 'supervision_mask_v2', 'validation_mask_v2'):
        refp = V + 'w00_%s-grid.npy' % name
        old = np.load(refp)
        g = grids[name][WR[0]:WR[1], WC[0]:WC[1]]
        d = np.abs(g - old)
        conv[name] = {'ref': refp, 'ref_sha256': sha256(refp), 'max_abs_diff': float(d.max()), 'px_differing_1e-6': int((d > 1e-6).sum()), 'n': int(d.size)}
        say('  CONVENTION CHECK, w00big window rows %d-%d cols %d-%d: %s vs %s: max |diff| %.3g, px differing (> 1e-6) %d of %d'
            % (WR[0], WR[1], WC[0], WC[1], name, refp, d.max(), (d > 1e-6).sum(), d.size))

# ---- S3 20260918 by index, day1-render-9um.py's --canvas rule ----------------------------------------------
R0, C0 = CFG['crop'][0] * 20, CFG['crop'][2] * 20
for name, path in CFG['s3_by_index'].items():
    cv = zarr.open_array(path, mode='r')[:]
    cv = cv.reshape(cv.shape[-2:]) if cv.ndim > 2 else cv
    assert cv.shape == (h * 20, w * 20), (cv.shape, h, w)
    blk = cv[R0:R0 + H * POOL, C0:C0 + W * POOL]
    g = blk.astype(np.float32).reshape(H, POOL, W, POOL).mean((1, 3)) / 255.0
    p = os.path.join(OUTD, '%s-%s-grid.npy' % (SEG, name))
    np.save(p, g)
    out_sha[os.path.basename(p)] = sha256(p)
    sup_canvas = cv > 0
    inside = sup_canvas[:H * POOL, :W * POOL]
    stats['s3_' + name] = {'mean_in_mask': float(g[mask].mean()), 'frac_gt_05': float((g[mask] > 0.5).mean()),
                           'canvas_px_gt0': int(sup_canvas.sum()), 'canvas_px_gt0_in_render_extent': int(inside.sum()),
                           'canvas_px_gt0_in_render_mask': int(inside.reshape(H, POOL, W, POOL).transpose(0, 2, 1, 3)[mask].sum())}
    say('  S3 20260918 %-12s by index (%s, canvas %s): mean over the mask %.4f, frac > 0.5 %.4f; canvas px > 0: %d, of them inside the render extent %d, inside the render mask %d'
        % (name, path, cv.shape, g[mask].mean(), (g[mask] > 0.5).mean(), stats['s3_' + name]['canvas_px_gt0'],
           stats['s3_' + name]['canvas_px_gt0_in_render_extent'], stats['s3_' + name]['canvas_px_gt0_in_render_mask']))
    del cv, blk, sup_canvas, inside

json.dump({'segment': SEG, 'prereg_sha256': sha256('geom/student-9um/prereg-student.json'), 'mesh': CFG['mesh'], 'crop_cells': list(CFG['crop']),
           'H': H, 'W': W, 'hf_sources': src, 'hf_dataset_window': [int(r0), int(r1), int(c0), int(c1)], 'hf_n_points': int(NPTS),
           'hf_nn_dist_render_mask': {'median': float(np.median(dm)), 'p95': float(np.percentile(dm, 95)), 'p99': float(np.percentile(dm, 99)), 'max': float(dm.max()),
                                      'gt10': int((dm > 10).sum()), 'gt20': int((dm > 20).sum()), 'window_edge': int((edge & mask).sum())},
           'convention': "day1-labels-hf.py's carry (labels-on-grid.py), unchanged; grid points batched by %d rows" % BLOCK,
           's3_by_index_rule': "day1-render-9um.py --canvas: render (i, j) = mean of canvas block [R0+4i, +4) x [C0+4j, +4), /255",
           'values_outside_mask': 'carried as computed; meaningful only inside <seg>-mask.npy (day 1 left 252405 the same way)',
           'stats': stats, 'convention_check_w00big': conv, 'outputs_sha256': out_sha},
          open(os.path.join(OUTD, '%s-labels.json' % SEG), 'w'), indent=1)
runfile.done(OUT, '\n'.join(lines) + '\n')
