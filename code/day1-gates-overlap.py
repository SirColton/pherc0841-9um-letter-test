"""day1-gates-overlap.py - plan step 0, gate (c). Does PHerc0841 auto_grown_20260220174252405 (catalog segment
20260221022814, the held-out test text) share any CO-SUPERVISED papyrus within 80 voxels (2.403 um voxels,
192 um) of w00 (20260220213127) or auto_grown_..552896 (20260220214732)? CPU only; local files; prints.

Two label sets, both measured, because they sit on different grids:
  S3  - ScrollPrize open-data, ink-labels/2.403um-volume-20260319124803/20260918 (reviewed set), on the
        published 2.403 um tifxyz grid (x20), the same grid as the 2.4 um ink-detection map;
  HF  - the v2 masks already on disk (inklabels_v2, supervision_mask_v2, validation_mask_v2) on the
        '_transformed' tifxyz grids (x~20.07), the set the 0.731 / 0.762 w00big numbers were scored with.
A grid vertex counts as supervised if ANY label pixel in its cell is supervised (conservative for leakage).
'Labelled' = supervised OR validation OR ink (any label footprint at all), the widest reading.
WHAT WOULD FAIL IT: any 252405 supervised vertex within 80 voxels of a supervised vertex of w00 or 552896.
Also prints: the same distances for all vertices (pure geometry), the HF-grid vs S3-grid surface offset,
and one sanity number: the ScrollPrize 2.4 um map vs the 252405 labels, AUC with supervised negatives
(the reference map only; nothing at 9 um is looked at here).
"""
import os
import numpy as np, tifffile, zarr
from scipy.spatial import cKDTree

G = 'geom/day1-9um/gates/s3'
A = 'geom/l0-1203-render/audit/'
SEG = {'w00': ('20260220213127-w00', '20260220213127'),
       '552896': ('20260220214732-auto_grown_20260220144552896', '20260220214732'),
       '252405': ('20260221022814-auto_grown_20260220174252405', '20260221022814')}
HF = {'w00': (A + 'evidence-control-training-labels/w00_%s.tif', A + 'evidence-control-training-labels/w00_w00_%s.tif'),
      '552896': (A + 'design-use-what-exists/seg/auto_grown_20260220144552896_%s.tif',) * 2,
      '252405': (A + 'design-use-what-exists/seg/auto_grown_20260220174252405_%s.tif',) * 2}
LIMS = (15, 40, 80, 160)

def cell_any(m, shape, f):
    """any-pooling of a full-res mask onto the tifxyz grid; vertex (i, j) owns pixels [i*f, (i+1)*f)"""
    H, W = shape
    b = (m > 0).view(np.uint8)
    si = np.minimum(np.ceil(np.arange(H) * f).astype(int), m.shape[0] - 1)   # f > 1, so no empty cells
    sj = np.minimum(np.ceil(np.arange(W) * f).astype(int), m.shape[1] - 1)
    r = np.maximum.reduceat(b, si, axis=0)
    return np.maximum.reduceat(r, sj, axis=1) > 0

def load_s3(n):
    seg, sid = SEG[n]
    d = os.path.join(G, seg, 'mesh', '%s-on-20260319124803-2.403um.tifxyz' % sid)
    X, Y, Z = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    lb = os.path.join(G, seg, 'ink-labels', '2.403um-volume-20260319124803', '20260918')
    L = {k: zarr.open_group(os.path.join(lb, k + '.zarr'), mode='r')['0'][:] for k in ('inklabels', 'supervision', 'validation')}
    f = L['supervision'].shape[0] / X.shape[0]
    return X, Y, Z, L, f

def load_hf(n):
    fx, fl = HF[n]
    X, Y, Z = (tifffile.imread(fx % k).astype(np.float64) for k in 'xyz')
    L = {}
    for k, name in (('inklabels', 'inklabels_v2'), ('supervision', 'supervision_mask_v2'), ('validation', 'validation_mask_v2')):
        a = tifffile.imread(fl % name)
        L[k] = a.reshape(a.shape[-2:]) if a.ndim > 2 else a
    f = L['supervision'].shape[0] / X.shape[0]
    return X, Y, Z, L, f

def vertices(X, Y, Z, L, f):
    ok = (X > 0) & (Y > 0) & (Z > 0)
    sup = cell_any(L['supervision'], X.shape, f)
    ink = cell_any(L['inklabels'], X.shape, f)
    lab = sup | cell_any(L['validation'], X.shape, f) | ink
    P = np.stack([X, Y, Z], -1)
    return P[ok], sup[ok], lab[ok], ink[ok], ok

for setname, loader in (('S3 20260918 labels, published 2.403 tifxyz grids', load_s3), ('HF v2 labels, _transformed grids', load_hf)):
    print('== %s' % setname)
    V = {}
    for n in SEG:
        X, Y, Z, L, f = loader(n)
        P, sup, lab, ink, ok = vertices(X, Y, Z, L, f)
        V[n] = (P, sup, lab, ink)
        print('   %-7s grid %s  label %s (x%.2f)  vertices %d  supervised %d (%.1f%%)  any-label %d (%.1f%%)  label-pixel fracs sup %.4f val %.4f ink %.4f' % (
            n, X.shape, L['supervision'].shape, f, len(P), sup.sum(), 100 * sup.mean(), lab.sum(), 100 * lab.mean(),
            (L['supervision'] > 0).mean(), (L['validation'] > 0).mean(), (L['inklabels'] > 0).mean()))
    P5, s5, l5, i5 = V['252405']
    print('   252405 ink-labelled vertices %d, of them inside its supervised cells %d' % (i5.sum(), (i5 & s5).sum()))
    for other in ('w00', '552896'):
        Po, so, lo, io = V[other]
        for what, ma, mb in (('supervised vs supervised', s5, so), ('252405 supervised vs all', s5, np.ones(len(Po), bool)),
                             ('252405 ink-labelled vs all', i5, np.ones(len(Po), bool)), ('ink-labelled vs ink-labelled', i5, io),
                             ('any-label vs any-label', l5, lo), ('all vertices (geometry)', np.ones(len(P5), bool), np.ones(len(Po), bool))):
            d, _ = cKDTree(Po[mb]).query(P5[ma])
            print('   252405 -> %-6s %-26s n %6d  min %8.1f vox (%7.0f um)  median %8.1f  ' % (other, what, ma.sum(), d.min(), 2.403 * d.min(), np.median(d)) +
                  '  '.join('<%d: %d' % (l, (d < l).sum()) for l in LIMS))
    print()

# ---- the HF grid vs the published grid, 252405 (how labels move between them) ----------------------------
X, Y, Z, _, _ = load_s3('252405'); Ps = np.stack([X, Y, Z], -1)[(X > 0) & (Y > 0) & (Z > 0)]
X, Y, Z, _, _ = load_hf('252405'); Ph = np.stack([X, Y, Z], -1)[(X > 0) & (Y > 0) & (Z > 0)]
tr = cKDTree(Ps)
q = Ph[::5]
d, k = tr.query(q, k=8)
off = np.array([abs((q[i] - Ps[k[i]].mean(0)) @ np.linalg.svd(Ps[k[i]] - Ps[k[i]].mean(0))[2][2]) for i in range(len(q))])
print('== 252405 HF _transformed grid vs published 2.403 grid: NN median %.2f p95 %.2f vox; point-to-plane median %.3f p95 %.3f vox; frac HF vertices > 40 vox from published mesh %.4f' % (
    np.median(d[:, 0]), np.percentile(d[:, 0], 95), np.median(off), np.percentile(off, 95), (d[:, 0] > 40).mean()))

# ---- sanity: the ScrollPrize 2.4 um map against each segment's labels on its own grid ---------------------
def pool(a, p=4):
    h, w = a.shape[0] // p * p, a.shape[1] // p * p
    return a[:h, :w].reshape(h // p, p, w // p, p).astype(np.float32).mean((1, 3))
def auc(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]); rk = v.argsort(kind='stable').argsort() + 1.0; n1, n0 = pos.sum(), neg.sum()
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
for n in ('252405', 'w00'):
    seg, sid = SEG[n]
    fp = os.path.join(G, seg, 'ink-detection', 'PHerc0841-%s-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif' % sid)
    if not os.path.exists(fp):
        print('== %s: no ScrollPrize map on disk, skipped' % n); continue
    ink = tifffile.imread(fp)
    _, _, _, L, _ = load_s3(n)
    assert ink.shape == L['supervision'].shape, (ink.shape, L['supervision'].shape)
    m, lab, sup, val = pool(ink), pool(L['inklabels']) / 255, pool(L['supervision']) / 255, pool(L['validation']) / 255
    valid = m > 0
    pos = valid & (lab >= 0.75)
    for nm, neg in (('supervised negatives (sup>=0.75, lab<=0)', valid & (lab <= 0) & (sup >= 0.75)),
                    ('validation-mask negatives (val>=0.75, lab<=0)', valid & (lab <= 0) & (val >= 0.75)),
                    ('all unlabelled pixels as negatives', valid & (lab <= 0))):
        print('== ScrollPrize 2.4 um map vs S3 20260918 labels on %-6s 9.6 um pooled: pos %7d neg %8d  AUC %.4f   [%s]' % (n, pos.sum(), neg.sum(), auc(m, pos, neg), nm))
    print('   %s ink-labelled pixels %d -> %.4f cm2 (full-res pixels x 2.403 um^2; assumes the label pixel pitch is 2.403 um)' % (
        n, (L['inklabels'] > 0).sum(), (L['inklabels'] > 0).sum() * 2.403 ** 2 / 1e8))

# ---- the selection frame: w00big (where recipes are chosen) against 252405 (where they are tested) -----------
Xb, Yb, Zb = (tifffile.imread(os.path.join('geom/c0841/mesh0b/w00big', k + '.tif')).astype(np.float64) for k in 'xyz')
okb = (Xb > 0) & (Yb > 0) & (Zb > 0)
Pb = np.stack([Xb[okb], Yb[okb], Zb[okb]], 1)
X, Y, Z, L, f = load_s3('252405')
P5, s5, l5, i5, _ = vertices(X, Y, Z, L, f)
tb = cKDTree(Pb)
for what, msk in (('all 252405 vertices', np.ones(len(P5), bool)), ('252405 supervised', s5), ('252405 ink-labelled', i5)):
    d, _ = tb.query(P5[msk])
    print('== w00big selection frame (%d vertices, grid %s) vs %-20s n %6d  min %8.1f vox (%6.0f um)  ' % (len(Pb), Xb.shape, what, msk.sum(), d.min(), 2.403 * d.min()) +
          '  '.join('<%d: %d' % (l, (d < l).sum()) for l in LIMS))
