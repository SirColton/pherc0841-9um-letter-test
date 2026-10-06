"""day1-sceptic-calib.py - sceptic: the gates part's calibration AUCs used ordinal ranks (stable argsort, positives
first), which break every tie AGAINST the positives. Recompute the same AUCs with midranks (ties = 1/2), same
pixels, same cells. Copies day1-gates-calib.py's cell carry and day1-gates-overlap.py's 4x4 pooling verbatim in
logic. CPU, local files; prints and writes geom/day1-9um/sceptic-numbers/sceptic-calib.txt only."""
import os
import numpy as np, tifffile, zarr
from scipy.spatial import cKDTree

OUT = 'geom/day1-9um/sceptic-numbers/sceptic-calib.txt'
G = 'geom/day1-9um/gates/s3'
A = 'geom/l0-1203-render/audit/'
SEG = {'w00': ('20260220213127-w00', '20260220213127', A + 'evidence-control-training-labels/w00_%s.tif', A + 'evidence-control-training-labels/w00_w00_%s.tif'),
       '252405': ('20260221022814-auto_grown_20260220174252405', '20260221022814', A + 'design-use-what-exists/seg/auto_grown_20260220174252405_%s.tif',
                  A + 'design-use-what-exists/seg/auto_grown_20260220174252405_%s.tif')}
lines = []


def say(s):
    print(s, flush=True); lines.append(s)


def auc_ord(s, pos, neg):
    v = np.concatenate([s[pos], s[neg]]); rk = v.argsort(kind='stable').argsort() + 1.0; n1, n0 = pos.sum(), neg.sum()
    return float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_mid(s, pos, neg):
    p = np.sort(s[pos].astype(np.float64)); n = np.sort(s[neg].astype(np.float64))
    lo = np.searchsorted(n, p, 'left'); hi = np.searchsorted(n, p, 'right')
    return float((lo + 0.5 * (hi - lo)).sum() / (len(p) * len(n)))


def cell_mean(m, shape, f):
    H, W = shape
    si = np.minimum(np.ceil(np.arange(H) * f).astype(int), m.shape[0] - 1)
    sj = np.minimum(np.ceil(np.arange(W) * f).astype(int), m.shape[1] - 1)
    ni = np.diff(np.append(si, m.shape[0])); nj = np.diff(np.append(sj, m.shape[1]))
    r = np.add.reduceat(m.astype(np.float64), si, axis=0)
    r = np.add.reduceat(r, sj, axis=1)
    return r / np.maximum(ni[:, None] * nj[None, :], 1)


def pool(a, p=4):
    h, w = a.shape[0] // p * p, a.shape[1] // p * p
    return a[:h, :w].reshape(h // p, p, w // p, p).astype(np.float32).mean((1, 3))


for n, (seg, sid, fx, fl) in SEG.items():
    ink = tifffile.imread(os.path.join(G, seg, 'ink-detection', 'PHerc0841-%s-2.403um-0.22m-77keV-volume-20260319124803-20260417190342-new_canon_autoresearch_recipe-tile256-stride128.tif' % sid))
    lb = os.path.join(G, seg, 'ink-labels', '2.403um-volume-20260319124803', '20260918')
    L = {k: zarr.open_group(os.path.join(lb, k + '.zarr'), mode='r')['0'][:] for k in ('inklabels', 'supervision', 'validation')}
    L = {k: (v.reshape(v.shape[-2:]) if v.ndim > 2 else v) for k, v in L.items()}
    # 9.6 um pooled (overlap script)
    m, lab, sup = pool(ink), pool(L['inklabels']) / 255, pool(L['supervision']) / 255
    valid = m > 0
    pos = valid & (lab >= 0.75)
    for nm, neg in (('supervised neg', valid & (lab <= 0) & (sup >= 0.75)), ('all unlabelled neg', valid & (lab <= 0))):
        allv = np.concatenate([m[pos], m[neg]])
        say('%-7s 9.6um pooled %-18s pos %7d neg %8d  AUC ordinal %.4f  midrank %.4f  (distinct values %d)' % (
            n, nm, pos.sum(), neg.sum(), auc_ord(m, pos, neg), auc_mid(m, pos, neg), len(np.unique(allv))))
    # 48 um cells (calib script)
    d = os.path.join(G, seg, 'mesh', '%s-on-20260319124803-2.403um.tifxyz' % sid)
    Xs, Ys, Zs = (tifffile.imread(os.path.join(d, k + '.tif')).astype(np.float64) for k in 'xyz')
    oks = (Xs > 0) & (Ys > 0) & (Zs > 0)
    fs = ink.shape[0] / Xs.shape[0]
    m_s = cell_mean(ink, Xs.shape, fs)
    lab_s = cell_mean(L['inklabels'] > 0, Xs.shape, fs); sup_s = cell_mean(L['supervision'] > 0, Xs.shape, fs)
    cov_s = cell_mean(ink > 0, Xs.shape, fs)
    Xh, Yh, Zh = (tifffile.imread(fx % k).astype(np.float64) for k in 'xyz')
    okh = (Xh > 0) & (Yh > 0) & (Zh > 0)
    HL = {}
    for k, nm in (('inklabels', 'inklabels_v2'), ('supervision', 'supervision_mask_v2')):
        a = tifffile.imread(fl % nm); HL[k] = a.reshape(a.shape[-2:]) if a.ndim > 2 else a
    fh = HL['supervision'].shape[0] / Xh.shape[0]
    lab_h = cell_mean(HL['inklabels'] > 0, Xh.shape, fh); sup_h = cell_mean(HL['supervision'] > 0, Xh.shape, fh)
    is_, js_ = np.nonzero(oks)
    tr = cKDTree(np.stack([Xs[oks], Ys[oks], Zs[oks]], 1))
    ih, jh = np.nonzero(okh)
    dd, kk = tr.query(np.stack([Xh[okh], Yh[okh], Zh[okh]], 1))
    near = dd < 20
    mh = m_s[is_[kk], js_[kk]]; covh = cov_s[is_[kk], js_[kk]]
    labsh = lab_s[is_[kk], js_[kk]]; supsh = sup_s[is_[kk], js_[kk]]
    lh, sh = lab_h[ih, jh], sup_h[ih, jh]
    use = near & (covh > 0.99)
    for setname, L_, S_ in (('HF v2', lh, sh), ('S3 20260918', labsh, supsh)):
        pos = use & (L_ >= 0.75); neg = use & (L_ <= 0) & (S_ >= 0.75)
        say('%-7s 48um cells   %-18s pos %7d neg %8d  AUC ordinal %.4f  midrank %.4f' % (n, setname + ' sup', pos.sum(), neg.sum(), auc_ord(mh, pos, neg), auc_mid(mh, pos, neg)))
open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
